#!/usr/bin/env python3
"""
Batch H5 generation script optimized for high-performance systems.

Generates H5 files for both chen11 and bu48 datasets with properly mapped ESM2 embeddings.
Optimized for systems with high CPU count, large RAM, and multiple GPUs.
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
from Bio import PDB
from typing import Dict, List, Tuple, Optional, Set
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import warnings
import time

warnings.filterwarnings('ignore')

# Configuration for feature extraction
FEATURE_EXCLUDES = {'file_name', 'residue_number', 'residue_name', 'chain_id', 'class'}

def load_csv_groups(csv_file: Path) -> Dict[str, pd.DataFrame]:
    """Load CSV and group by protein name."""
    print(f"Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"CSV shape: {df.shape}")
    
    groups = {}
    for fn, grp in df.groupby('file_name'):
        if fn.endswith('.pdb'):
            prot_name = fn[:-4]  # Remove .pdb extension
            groups[prot_name] = grp.drop(columns=['file_name'])
    
    print(f"Found {len(groups)} proteins")
    return groups

def extract_pdb_sequence(pdb_file: Path) -> Tuple[List[int], Dict[int, int]]:
    """Extract PDB sequence mapping."""
    parser = PDB.PDBParser(QUIET=True)
    struct = parser.get_structure('', str(pdb_file))
    
    aa_map = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
              "GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
              "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
              "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    
    pdb_to_seq = {}
    idx = 0
    for model in struct:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ':  # Skip non-standard residues
                    continue
                name = res.resname
                num = res.id[1]
                if name in aa_map:
                    idx += 1
                    pdb_to_seq[num] = idx
            break  # Only first chain
        break  # Only first model
    
    return list(pdb_to_seq.keys()), pdb_to_seq

def create_mapping(esm_len: int, pdb_to_seq: Dict[int,int], csv_res: Dict[int,int]) -> Dict:
    """Create ESM to CSV residue mapping."""
    seq2pdb = {v: k for k, v in pdb_to_seq.items()}
    esm2csv, csv2esm, esm_no = {}, {}, []
    
    for pos in range(1, esm_len + 1):
        if pos in seq2pdb:
            pdbn = seq2pdb[pos]
            if pdbn in csv_res:
                esm2csv[pos] = pdbn
                csv2esm[pdbn] = pos
            else:
                esm_no.append(pos)
    
    return {'esm_to_csv': esm2csv, 'csv_to_esm': csv2esm, 'esm_no_sas': esm_no}

def get_embedding(tok_emb: torch.Tensor, res_num: int, mapping: Dict, fallback: str) -> np.ndarray:
    """Get ESM embedding for a specific residue."""
    csv2esm = mapping['csv_to_esm']
    
    if res_num in csv2esm:
        idx = csv2esm[res_num] - 1  # Convert to 0-based indexing
        return tok_emb[idx].numpy()
    
    # Fallback strategies
    if fallback == 'neighboring' and csv2esm:
        closest = min(csv2esm, key=lambda r: abs(r - res_num))
        idx = csv2esm[closest] - 1
        return tok_emb[idx].numpy()
    elif fallback == 'global_mean':
        return tok_emb.mean(dim=0).numpy()
    else:  # zero fallback
        return tok_emb[0].numpy()

def process_protein(args):
    """Process a single protein."""
    prot, df, pdb_dir, esm_dir, fb = args
    pdb_file = pdb_dir / f"{prot}.pdb"
    esm_file = esm_dir / f"{prot}.pt"
    
    # Check file existence
    if not pdb_file.exists():
        print(f"PDB file missing: {pdb_file}")
        return None
    if not esm_file.exists():
        print(f"ESM file missing: {esm_file}")
        return None
    
    try:
        # Load ESM embeddings
        emb = torch.load(esm_file, map_location='cpu')
        if emb.size(0) > 2:  # Remove BOS/EOS tokens if present
            emb = emb[1:-1]
        
        # Extract PDB sequence
        keys, pdb2seq = extract_pdb_sequence(pdb_file)
        
        # Get residue counts from CSV
        res_counts = df['residue_number'].value_counts().to_dict()
        
        # Create mapping
        mapping = create_mapping(emb.size(0), pdb2seq, res_counts)
        
        # Get feature columns (excluding metadata)
        feat_cols = [c for c in df.columns if c not in FEATURE_EXCLUDES]
        
        # Process each row
        tab, esm_feats, lbls, resnums = [], [], [], []
        for _, row in df.iterrows():
            # Tabular features
            tab_feat = row[feat_cols].values.astype(np.float32)
            tab.append(tab_feat)
            
            # ESM embedding for this residue
            esm_feat = get_embedding(emb, int(row['residue_number']), mapping, fb).astype(np.float32)
            esm_feats.append(esm_feat)
            
            # Labels and metadata
            lbls.append(int(row['class']))
            resnums.append(int(row['residue_number']))
        
        return prot, np.vstack(tab), np.vstack(esm_feats), np.array(lbls), np.array(resnums)
    
    except Exception as e:
        print(f"Error processing {prot}: {e}")
        return None

def create_h5_dataset(csv_file: str, pdb_dir: str, esm_dir: str, out_file: str, 
                     fallback: str = 'neighboring', nproc: int = None):
    """Create H5 dataset with optimizations for high-performance systems."""
    
    print(f"\n=== Creating H5 dataset: {out_file} ===")
    start_time = time.time()
    
    pdb_dir, esm_dir = Path(pdb_dir), Path(esm_dir)
    
    # Determine PDB subdirectory based on dataset
    if 'chen11' in csv_file:
        pdb_subdir = pdb_dir / 'chen11'
        dataset_name = 'chen11'
    elif 'bu48' in csv_file:
        pdb_subdir = pdb_dir / 'bu48'
        dataset_name = 'bu48'
    else:
        pdb_subdir = pdb_dir
        dataset_name = 'unknown'
    
    print(f"Dataset: {dataset_name}")
    print(f"PDB directory: {pdb_subdir}")
    print(f"ESM directory: {esm_dir}")
    
    # Load and group CSV data
    groups = load_csv_groups(Path(csv_file))
    prots = list(groups.keys())
    print(f"Processing {len(prots)} proteins...")
    
    # Prepare arguments for multiprocessing
    args = [(p, groups[p], pdb_subdir, esm_dir, fallback) for p in prots]
    
    # Use all available CPU cores (or specified number)
    if nproc is None:
        nproc = cpu_count()
    print(f"Using {nproc} CPU cores for processing")
    
    # Process proteins in parallel
    with Pool(nproc) as pool:
        results = list(tqdm(pool.imap(process_protein, args), 
                          total=len(args), desc="Processing proteins"))
    
    # Filter successful results
    results = [r for r in results if r is not None]
    print(f"Successfully processed {len(results)} out of {len(prots)} proteins")
    
    if not results:
        print("ERROR: No proteins were successfully processed!")
        return False
    
    # Combine all data
    print("Combining data arrays...")
    tab_all = np.vstack([r[1] for r in results])
    esm_all = np.vstack([r[2] for r in results])
    lbl_all = np.concatenate([r[3] for r in results])
    res_all = np.concatenate([r[4] for r in results])
    
    # Create protein IDs array
    prot_ids = []
    for r in results:
        prot_ids.extend([r[0]] * len(r[3]))
    prot_ids = np.array(prot_ids)
    
    print(f"Final dataset shapes:")
    print(f"  Tabular features: {tab_all.shape}")
    print(f"  ESM embeddings: {esm_all.shape}")
    print(f"  Labels: {lbl_all.shape}")
    print(f"  Residue numbers: {res_all.shape}")
    print(f"  Protein IDs: {prot_ids.shape}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    # Write H5 file with compression
    print(f"Writing H5 file: {out_file}")
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('tabular', data=tab_all, compression='gzip', compression_opts=6)
        f.create_dataset('esm', data=esm_all, compression='gzip', compression_opts=6)
        f.create_dataset('labels', data=lbl_all, compression='gzip', compression_opts=6)
        f.create_dataset('residue_numbers', data=res_all, compression='gzip', compression_opts=6)
        f.create_dataset('protein_ids', data=[x.encode() for x in prot_ids], compression='gzip', compression_opts=6)
        
        # Add metadata
        f.attrs['dataset'] = dataset_name
        f.attrs['fallback_strategy'] = fallback
        f.attrs['n_proteins'] = len(results)
        f.attrs['n_samples'] = len(lbl_all)
        f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    elapsed = time.time() - start_time
    print(f"✅ Successfully created {out_file}")
    print(f"   Processing time: {elapsed:.2f} seconds")
    print(f"   Samples: {len(lbl_all):,}")
    print(f"   File size: {os.path.getsize(out_file) / 1024**3:.2f} GB")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate H5 files with ESM embeddings')
    parser.add_argument('--chen11', action='store_true', help='Generate chen11 dataset')
    parser.add_argument('--bu48', action='store_true', help='Generate bu48 dataset')
    parser.add_argument('--all', action='store_true', help='Generate both datasets')
    parser.add_argument('--pdb', default='data/orginal_pdb', help='PDB directory')
    parser.add_argument('--esm', default='data/esm2', help='ESM directory')
    parser.add_argument('--output-dir', default='data/h5', help='Output directory for H5 files')
    parser.add_argument('--fallback', default='neighboring', choices=['neighboring', 'global_mean', 'zero'],
                       help='Fallback strategy for missing residues')
    parser.add_argument('--nproc', type=int, default=None, help='Number of CPU cores (default: all)')
    
    args = parser.parse_args()
    
    if not (args.chen11 or args.bu48 or args.all):
        print("Please specify --chen11, --bu48, or --all")
        return
    
    success_count = 0
    
    if args.chen11 or args.all:
        if create_h5_dataset(
            csv_file='data/train/chen11.csv',
            pdb_dir=args.pdb,
            esm_dir=args.esm,
            out_file=f'{args.output_dir}/chen11_with_esm.h5',
            fallback=args.fallback,
            nproc=args.nproc
        ):
            success_count += 1
    
    if args.bu48 or args.all:
        if create_h5_dataset(
            csv_file='data/test/bu48.csv',
            pdb_dir=args.pdb,
            esm_dir=args.esm,
            out_file=f'{args.output_dir}/bu48_with_esm.h5',
            fallback=args.fallback,
            nproc=args.nproc
        ):
            success_count += 1
    
    print(f"\n🎉 Generated {success_count} H5 datasets successfully!")

if __name__ == '__main__':
    main()

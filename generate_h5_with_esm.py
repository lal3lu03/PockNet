#!/usr/bin/env python3
"""
Generate H5 files with properly mapped ESM2 embeddings for PockNet datasets.

This script creates optimized H5 files containing:
1. Tabular features from CSV files
2. Residue-specific ESM2 embeddings properly mapped to each SAS point
3. Labels and metadata

The resulting H5 files can be loaded much faster during training.
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

warnings.filterwarnings('ignore')  # suppress BioPython warnings

# Pre-extract feature column names to avoid per-loop recomputation
FEATURE_EXCLUDES = {'file_name', 'residue_number', 'residue_name', 'chain_id', 'class'}

# Load and cache CSV data grouped by protein

def load_csv_groups(csv_file: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_file)
    groups = {
        fn[:-4]: grp.drop(columns=['file_name'])
        for fn, grp in df.groupby('file_name')
        if fn.endswith('.pdb')
    }
    return groups


def extract_pdb_sequence(pdb_file: Path) -> Tuple[List[int], Dict[int, int]]:
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
                if res.id[0] != ' ':
                    continue
                name = res.resname
                num = res.id[1]
                if name in aa_map:
                    idx += 1
                    pdb_to_seq[num] = idx
            break
        break
    return list(pdb_to_seq.keys()), pdb_to_seq


def create_mapping(esm_len:int, pdb_to_seq:Dict[int,int], csv_res:Dict[int,int]) -> Dict:
    seq2pdb = {v:k for k,v in pdb_to_seq.items()}
    esm2csv, csv2esm, esm_no = {}, {}, []
    for pos in range(1, esm_len+1):
        if pos in seq2pdb:
            pdbn = seq2pdb[pos]
            if pdbn in csv_res:
                esm2csv[pos] = pdbn
                csv2esm[pdbn] = pos
            else:
                esm_no.append(pos)
    return {'esm_to_csv': esm2csv, 'csv_to_esm': csv2esm, 'esm_no_sas': esm_no}


def get_embedding(tok_emb: torch.Tensor, res_num:int, mapping:Dict, fallback:str) -> np.ndarray:
    csv2esm = mapping['csv_to_esm']
    if res_num in csv2esm:
        idx = csv2esm[res_num]-1
        return tok_emb[idx].numpy()
    if fallback=='neighboring' and csv2esm:
        closest = min(csv2esm, key=lambda r: abs(r-res_num))
        idx = csv2esm[closest]-1
        return tok_emb[idx].numpy()
    if fallback=='global_mean':
        return tok_emb.mean(dim=0).numpy()
    return tok_emb[0].numpy()


def process_protein(args):
    prot, df, pdb_dir, esm_dir, fb = args
    pdb_file = pdb_dir/ f"{prot}.pdb"
    esm_file = esm_dir/ f"{prot}.pt"
    if not pdb_file.exists():
        print(f"PDB file missing: {pdb_file}")
        return None
    if not esm_file.exists():
        print(f"ESM file missing: {esm_file}")
        return None
    emb = torch.load(esm_file, map_location='cpu')
    if emb.size(0)>2: emb = emb[1:-1]
    keys, pdb2seq = extract_pdb_sequence(pdb_file)
    res_counts = df['residue_number'].value_counts().to_dict()
    mapping = create_mapping(emb.size(0), pdb2seq, res_counts)
    feat_cols = [c for c in df.columns if c not in FEATURE_EXCLUDES]
    tab, esm_feats, lbls, resnums = [], [], [], []
    for _, row in df.iterrows():
        tab.append(row[feat_cols].values.astype(np.float32))
        esm_feats.append(get_embedding(emb, int(row['residue_number']), mapping, fb).astype(np.float32))
        lbls.append(int(row['class'])); resnums.append(int(row['residue_number']))
    return prot, np.vstack(tab), np.vstack(esm_feats), np.array(lbls), np.array(resnums)


def create_h5(csv_file:str,pdb_dir:str,esm_dir:str,out_file:str,fb:str,nproc:int=None):
    pdb_dir, esm_dir = Path(pdb_dir), Path(esm_dir)
    groups = load_csv_groups(Path(csv_file))
    prots = list(groups)
    
    # Determine dataset name for PDB subdirectory
    if 'chen11' in csv_file:
        pdb_subdir = pdb_dir / 'chen11'
    elif 'bu48' in csv_file:
        pdb_subdir = pdb_dir / 'bu48'
    else:
        pdb_subdir = pdb_dir
        
    args = [(p, groups[p], pdb_subdir, esm_dir, fb) for p in prots]
    with Pool(nproc or cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_protein, args), total=len(args)))
    results = [r for r in results if r]
    print(f"Successfully processed {len(results)} out of {len(prots)} proteins")
    if not results:
        print("No proteins were successfully processed!")
        return
    # assemble arrays
    tab_all = np.vstack([r[1] for r in results])
    esm_all = np.vstack([r[2] for r in results])
    lbl_all = np.concatenate([r[3] for r in results])
    res_all = np.concatenate([r[4] for r in results])
    prot_ids = np.concatenate([[r[0]]*len(r[3]) for r in results])
    # write H5
    with h5py.File(out_file,'w') as f:
        f.create_dataset('tabular', data=tab_all, compression='gzip')
        f.create_dataset('esm', data=esm_all, compression='gzip')
        f.create_dataset('labels', data=lbl_all, compression='gzip')
        f.create_dataset('resnums', data=res_all, compression='gzip')
        f.create_dataset('pids', data=[x.encode() for x in prot_ids], compression='gzip')
    print("Saved H5:", out_file)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True);
    p.add_argument('--pdb',required=True);
    p.add_argument('--esm',required=True);
    p.add_argument('--out',required=True);
    p.add_argument('--fb',default='neighboring');
    p.add_argument('--nproc',type=int,default=None);
    args=p.parse_args()
    create_h5(args.csv,args.pdb,args.esm,args.out,args.fb,args.nproc)

if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""
Generate residue mappings for all proteins in the chen11 dataset.

This script creates proper mappings between ESM sequence positions and CSV residue numbers
for all proteins in the training dataset.
"""

import os
import pandas as pd
import torch
from Bio import PDB
from typing import Dict, List, Tuple, Optional, Set
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_pdb_sequence(pdb_file: str) -> Tuple[str, List[int], Dict[int, int]]:
    """Extract protein sequence from PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    sequence = ''
    pdb_residue_numbers = []
    pdb_to_seq_mapping = {}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Standard amino acid
                    residue_name = residue.resname
                    residue_num = residue.id[1]
                    
                    if residue_name in aa_map:
                        sequence += aa_map[residue_name]
                        pdb_residue_numbers.append(residue_num)
                        pdb_to_seq_mapping[residue_num] = len(pdb_residue_numbers)
            break
        break
    
    return sequence, pdb_residue_numbers, pdb_to_seq_mapping


def analyze_csv_residues(csv_file: str, protein_name: str) -> Dict[int, int]:
    """Analyze which residues are present in the CSV file for a given protein."""
    df = pd.read_csv(csv_file)
    protein_data = df[df['file_name'] == protein_name]
    
    residue_counts = protein_data['residue_number'].value_counts().to_dict()
    return residue_counts


def create_residue_mapping(
    esm_length: int,
    pdb_to_seq_mapping: Dict[int, int],
    csv_residue_counts: Dict[int, int]
) -> Dict[str, any]:
    """Create comprehensive mapping between ESM positions and CSV residues."""
    seq_to_pdb_mapping = {v: k for k, v in pdb_to_seq_mapping.items()}
    
    esm_to_csv_residue = {}
    csv_residue_to_esm = {}
    esm_without_sas = []
    
    for esm_pos in range(1, esm_length + 1):
        if esm_pos in seq_to_pdb_mapping:
            pdb_residue_num = seq_to_pdb_mapping[esm_pos]
            
            if pdb_residue_num in csv_residue_counts:
                esm_to_csv_residue[esm_pos] = pdb_residue_num
                csv_residue_to_esm[pdb_residue_num] = esm_pos
            else:
                esm_without_sas.append(esm_pos)
    
    csv_without_esm = [res for res in csv_residue_counts.keys() 
                       if res not in csv_residue_to_esm]
    
    return {
        'esm_length': esm_length,
        'total_pdb_residues': len(pdb_to_seq_mapping),
        'total_csv_residues': len(csv_residue_counts),
        'esm_to_csv_residue': esm_to_csv_residue,
        'csv_residue_to_esm': csv_residue_to_esm,
        'esm_without_sas': esm_without_sas,
        'csv_without_esm': csv_without_esm,
        'pdb_to_seq_mapping': pdb_to_seq_mapping,
        'seq_to_pdb_mapping': seq_to_pdb_mapping,
        'csv_residue_counts': csv_residue_counts
    }


def create_protein_mapping(protein_id: str, csv_file: str, pdb_dir: str, esm_dir: str) -> Optional[Dict[str, any]]:
    """Create complete mapping for a single protein."""
    try:
        # File paths
        pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
        esm_file = os.path.join(esm_dir, f"{protein_id}.pt")
        
        # Check if files exist
        if not os.path.exists(pdb_file):
            print(f"  Warning: PDB file not found: {pdb_file}")
            return None
            
        if not os.path.exists(esm_file):
            print(f"  Warning: ESM file not found: {esm_file}")
            return None
        
        # Extract PDB sequence
        sequence, pdb_residue_numbers, pdb_to_seq_mapping = extract_pdb_sequence(pdb_file)
        
        # Load ESM embedding
        esm_data = torch.load(esm_file, map_location='cpu')
        esm_length = esm_data.shape[0] - 2  # Remove BOS/EOS tokens
        
        # Analyze CSV residues
        csv_residue_counts = analyze_csv_residues(csv_file, f"{protein_id}.pdb")
        
        if len(csv_residue_counts) == 0:
            print(f"  Warning: No CSV data found for {protein_id}")
            return None
        
        # Verify sequence lengths match
        if len(sequence) != esm_length:
            print(f"  Warning: PDB sequence length ({len(sequence)}) != ESM length ({esm_length}) for {protein_id}")
        
        # Create mapping
        mapping = create_residue_mapping(esm_length, pdb_to_seq_mapping, csv_residue_counts)
        
        # Add metadata
        mapping.update({
            'protein_id': protein_id,
            'pdb_sequence': sequence,
            'pdb_file': pdb_file,
            'esm_file': esm_file,
            'csv_file': csv_file
        })
        
        return mapping
        
    except Exception as e:
        print(f"  Error processing {protein_id}: {e}")
        return None


def save_mapping(mapping: Dict[str, any], output_file: str):
    """Save mapping to JSON file."""
    def convert_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj
    
    mapping_serializable = convert_types(mapping)
    
    with open(output_file, 'w') as f:
        json.dump(mapping_serializable, f, indent=2)


def get_proteins_from_csv(csv_file: str) -> Set[str]:
    """Get unique protein IDs from CSV file."""
    df = pd.read_csv(csv_file)
    
    if 'file_name' not in df.columns:
        raise ValueError("CSV file must contain 'file_name' column")
    
    protein_ids = set()
    for file_name in df['file_name'].unique():
        if file_name.endswith('.pdb'):
            protein_id = file_name[:-4]
            protein_ids.add(protein_id)
    
    return protein_ids


def main():
    """Generate mappings for all proteins in the dataset."""
    parser = argparse.ArgumentParser(description='Generate residue mappings for all proteins')
    parser.add_argument('--data_dir', default='.', help='Base data directory')
    parser.add_argument('--dataset', default='chen11', help='Dataset name (chen11 or bu48)')
    parser.add_argument('--output_dir', default='mappings', help='Output directory for mappings')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mappings')
    
    args = parser.parse_args()
    
    # Set up paths
    csv_file = os.path.join(args.data_dir, 'data', 'train', f'{args.dataset}.csv')
    pdb_dir = os.path.join(args.data_dir, 'data', 'orginal_pdb', args.dataset)
    esm_dir = os.path.join(args.data_dir, 'data', 'esm2')
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input files exist
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    if not os.path.exists(pdb_dir):
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")
        
    if not os.path.exists(esm_dir):
        raise FileNotFoundError(f"ESM directory not found: {esm_dir}")
    
    print(f"Generating mappings for dataset: {args.dataset}")
    print(f"CSV file: {csv_file}")
    print(f"PDB directory: {pdb_dir}")
    print(f"ESM directory: {esm_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get protein IDs from CSV
    protein_ids = get_proteins_from_csv(csv_file)
    print(f"Found {len(protein_ids)} unique proteins in CSV")
    
    # Process each protein
    successful_mappings = 0
    failed_mappings = 0
    skipped_mappings = 0
    
    for protein_id in tqdm(protein_ids, desc="Processing proteins"):
        output_file = os.path.join(output_dir, f"{protein_id}_residue_mapping.json")
        
        # Skip if file exists and not overwriting
        if os.path.exists(output_file) and not args.overwrite:
            skipped_mappings += 1
            continue
        
        print(f"Processing {protein_id}...")
        mapping = create_protein_mapping(protein_id, csv_file, pdb_dir, esm_dir)
        
        if mapping is not None:
            save_mapping(mapping, output_file)
            successful_mappings += 1
            print(f"  Success: ESM {mapping['esm_length']}, PDB {mapping['total_pdb_residues']}, CSV {mapping['total_csv_residues']}")
        else:
            failed_mappings += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Total proteins: {len(protein_ids)}")
    print(f"Successful mappings: {successful_mappings}")
    print(f"Failed mappings: {failed_mappings}")
    print(f"Skipped (already exist): {skipped_mappings}")
    print(f"Mappings saved to: {output_dir}")


if __name__ == '__main__':
    main()

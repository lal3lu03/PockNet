#!/usr/bin/env python3
"""
Create residue-specific ESM-CSV mapping for PockNet.

This script creates a proper mapping between ESM sequence positions and CSV residue numbers,
handling cases where some residues may not have SAS points in the CSV data.
"""

import os
import pandas as pd
import torch
from Bio import PDB
from typing import Dict, List, Tuple, Optional
import pickle
import json
from pathlib import Path


def extract_pdb_sequence(pdb_file: str) -> Tuple[str, List[int], Dict[int, int]]:
    """
    Extract protein sequence from PDB file.
    
    Returns:
        sequence: Amino acid sequence string
        pdb_residue_numbers: List of PDB residue numbers in order
        pdb_to_seq_mapping: Dict mapping PDB residue number to sequential position (1-based)
    """
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
                        # Sequential position is 1-based
                        pdb_to_seq_mapping[residue_num] = len(pdb_residue_numbers)
            break  # Only process first chain
        break  # Only process first model
    
    return sequence, pdb_residue_numbers, pdb_to_seq_mapping


def analyze_csv_residues(csv_file: str, protein_name: str) -> Dict[int, int]:
    """
    Analyze which residues are present in the CSV file for a given protein.
    
    Returns:
        Dict mapping residue number to count of SAS points
    """
    df = pd.read_csv(csv_file)
    protein_data = df[df['file_name'] == protein_name]
    
    residue_counts = protein_data['residue_number'].value_counts().to_dict()
    return residue_counts


def create_residue_mapping(
    esm_length: int,
    pdb_to_seq_mapping: Dict[int, int],
    csv_residue_counts: Dict[int, int]
) -> Dict[str, any]:
    """
    Create comprehensive mapping between ESM positions and CSV residues.
    
    Args:
        esm_length: Number of amino acids in ESM sequence (excluding BOS/EOS)
        pdb_to_seq_mapping: PDB residue number -> sequential position
        csv_residue_counts: Residue number -> SAS point count
    
    Returns:
        Dictionary with mapping information
    """
    # Create bidirectional mappings
    seq_to_pdb_mapping = {v: k for k, v in pdb_to_seq_mapping.items()}
    
    # ESM positions that have corresponding residues in CSV
    esm_to_csv_residue = {}
    csv_residue_to_esm = {}
    
    # ESM positions that exist but have no SAS points in CSV
    esm_without_sas = []
    
    for esm_pos in range(1, esm_length + 1):
        if esm_pos in seq_to_pdb_mapping:
            pdb_residue_num = seq_to_pdb_mapping[esm_pos]
            
            if pdb_residue_num in csv_residue_counts:
                # This ESM position has SAS points in CSV
                esm_to_csv_residue[esm_pos] = pdb_residue_num
                csv_residue_to_esm[pdb_residue_num] = esm_pos
            else:
                # This ESM position exists in PDB but has no SAS points
                esm_without_sas.append(esm_pos)
    
    # CSV residues that don't have direct ESM mapping (shouldn't happen if sequences match)
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


def create_protein_mapping(protein_id: str, base_dir: str = '.') -> Dict[str, any]:
    """
    Create complete mapping for a single protein.
    """
    # File paths
    pdb_file = f"{base_dir}/data/orginal_pdb/chen11/{protein_id}.pdb"
    esm_file = f"{base_dir}/data/esm2/{protein_id}.pt"
    csv_file = f"{base_dir}/data/train/chen11.csv"
    
    print(f"Processing protein: {protein_id}")
    
    # Extract PDB sequence
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    sequence, pdb_residue_numbers, pdb_to_seq_mapping = extract_pdb_sequence(pdb_file)
    print(f"  PDB sequence length: {len(sequence)}")
    print(f"  PDB residue range: {min(pdb_residue_numbers)} to {max(pdb_residue_numbers)}")
    
    # Load ESM embedding
    if not os.path.exists(esm_file):
        raise FileNotFoundError(f"ESM file not found: {esm_file}")
    
    esm_data = torch.load(esm_file, map_location='cpu')
    esm_length = esm_data.shape[0] - 2  # Remove BOS/EOS tokens
    print(f"  ESM sequence length: {esm_length}")
    
    # Analyze CSV residues
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    csv_residue_counts = analyze_csv_residues(csv_file, f"{protein_id}.pdb")
    print(f"  CSV residues with SAS points: {len(csv_residue_counts)}")
    
    # Verify sequence lengths match
    if len(sequence) != esm_length:
        print(f"  WARNING: PDB sequence length ({len(sequence)}) != ESM length ({esm_length})")
    
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
    
    print(f"  ESM positions with SAS points: {len(mapping['esm_to_csv_residue'])}")
    print(f"  ESM positions without SAS points: {len(mapping['esm_without_sas'])}")
    
    return mapping


def save_mapping(mapping: Dict[str, any], output_file: str):
    """Save mapping to file."""
    # Convert numpy types to regular Python types for JSON serialization
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
    
    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(mapping_serializable, f, indent=2)
    elif output_file.endswith('.pkl'):
        with open(output_file, 'wb') as f:
            pickle.dump(mapping, f)
    else:
        raise ValueError("Output file must be .json or .pkl")


def main():
    """Create mapping for the test protein."""
    protein_id = "a.001.001.001_1s69a"
    
    # Create mapping
    mapping = create_protein_mapping(protein_id)
    
    # Save mapping
    os.makedirs('mappings', exist_ok=True)
    save_mapping(mapping, f'mappings/{protein_id}_residue_mapping.json')
    save_mapping(mapping, f'mappings/{protein_id}_residue_mapping.pkl')
    
    print(f"\nMapping saved to:")
    print(f"  mappings/{protein_id}_residue_mapping.json")
    print(f"  mappings/{protein_id}_residue_mapping.pkl")
    
    # Print summary
    print(f"\n=== MAPPING SUMMARY ===")
    print(f"Protein: {protein_id}")
    print(f"ESM sequence length: {mapping['esm_length']}")
    print(f"PDB residues: {mapping['total_pdb_residues']}")
    print(f"CSV residues with SAS: {mapping['total_csv_residues']}")
    print(f"ESM->CSV mappable: {len(mapping['esm_to_csv_residue'])}")
    print(f"ESM without SAS: {len(mapping['esm_without_sas'])}")
    
    if mapping['esm_without_sas']:
        print(f"ESM positions without SAS points: {mapping['esm_without_sas']}")
    
    if mapping['csv_without_esm']:
        print(f"CSV residues without ESM: {mapping['csv_without_esm']}")


if __name__ == '__main__':
    main()

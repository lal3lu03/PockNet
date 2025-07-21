#!/usr/bin/env python3
"""
Script to verify ESM embeddings are properly mapped to CSV residue data.

This script examines the mapping between:
1. ESM sequence embeddings (sequential 1-123 after removing BOS/EOS tokens)
2. CSV residue features (PDB numbering with potential gaps)

The goal is to verify if the current data pipeline correctly aligns residue-specific
information between the two data sources.
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_esm_embedding(esm_path):
    """Load ESM embedding and return sequence information."""
    embedding = torch.load(esm_path)
    print(f"ESM embedding shape: {embedding.shape}")
    
    # ESM format: [BOS, residue1, residue2, ..., residueN, EOS]
    # Remove BOS (position 0) and EOS (position -1) tokens
    sequence_embeddings = embedding[1:-1]  # Remove first and last tokens
    
    print(f"Sequence embeddings after removing BOS/EOS: {sequence_embeddings.shape}")
    print(f"Number of amino acid positions: {sequence_embeddings.shape[0]}")
    
    return sequence_embeddings


def analyze_csv_residues(csv_path):
    """Analyze residue numbering in CSV file."""
    df = pd.read_csv(csv_path)
    
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Analyze residue information
    if 'residue_number' in df.columns:
        unique_residues = df['residue_number'].unique()
        print(f"Unique residue numbers: {len(unique_residues)}")
        print(f"Residue range: {min(unique_residues)} to {max(unique_residues)}")
        print(f"First 10 residue numbers: {sorted(unique_residues)[:10]}")
        print(f"Last 10 residue numbers: {sorted(unique_residues)[-10:]}")
        
        # Check for gaps in residue numbering
        sorted_residues = sorted(unique_residues)
        gaps = []
        for i in range(len(sorted_residues) - 1):
            if sorted_residues[i+1] - sorted_residues[i] > 1:
                missing_range = list(range(sorted_residues[i] + 1, sorted_residues[i+1]))
                gaps.extend(missing_range)
        
        if gaps:
            print(f"Missing residue numbers (gaps): {gaps}")
        else:
            print("No gaps in residue numbering")
            
        # Count SAS points per residue
        residue_counts = df['residue_number'].value_counts()
        print(f"SAS points per residue - min: {residue_counts.min()}, max: {residue_counts.max()}, mean: {residue_counts.mean():.2f}")
    
    if 'file_name' in df.columns:
        unique_files = df['file_name'].unique()
        print(f"Unique files in CSV: {len(unique_files)}")
        print(f"First few files: {unique_files[:5]}")
    
    return df


def examine_protein_mapping(protein_id="a.001.001.001_1s69a"):
    """Examine mapping for a specific protein."""
    print(f"\n=== Examining protein {protein_id} ===")
    
    # Paths
    esm_path = f"/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data/esm2/{protein_id}.pt"
    csv_path = "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data/train/chen11.csv"
    
    # Load ESM embedding
    if os.path.exists(esm_path):
        print(f"\n--- ESM Embedding Analysis ---")
        sequence_embeddings = load_esm_embedding(esm_path)
        esm_residue_count = sequence_embeddings.shape[0]
    else:
        print(f"ESM file not found: {esm_path}")
        return
    
    # Load and filter CSV data for this protein
    print(f"\n--- CSV Analysis ---")
    df = analyze_csv_residues(csv_path)
    
    # Filter for this specific protein
    protein_df = df[df['file_name'] == f"{protein_id}.pdb"]
    if len(protein_df) == 0:
        print(f"No data found for {protein_id}.pdb in CSV")
        return
    
    print(f"\nProtein-specific data:")
    print(f"Total SAS points for {protein_id}: {len(protein_df)}")
    
    if 'residue_number' in protein_df.columns:
        unique_residues = protein_df['residue_number'].unique()
        csv_residue_count = len(unique_residues)
        
        print(f"Unique residues in CSV: {csv_residue_count}")
        print(f"ESM residue count: {esm_residue_count}")
        print(f"Difference: {esm_residue_count - csv_residue_count}")
        
        # Show residue mapping
        sorted_residues = sorted(unique_residues)
        print(f"CSV residue range: {min(sorted_residues)} to {max(sorted_residues)}")
        
        # Identify missing residues
        full_range = list(range(min(sorted_residues), max(sorted_residues) + 1))
        missing_residues = [r for r in full_range if r not in unique_residues]
        if missing_residues:
            print(f"Missing residues in CSV: {missing_residues}")
        
        # Show which ESM positions would map to which PDB residues
        print(f"\nMapping analysis:")
        print(f"ESM sequential positions 1-{esm_residue_count}")
        print(f"CSV PDB residue numbers: {sorted_residues[:10]}...{sorted_residues[-10:]}")
        
        return {
            'esm_residue_count': esm_residue_count,
            'csv_residue_count': csv_residue_count,
            'csv_residue_range': (min(sorted_residues), max(sorted_residues)),
            'missing_residues': missing_residues,
            'csv_residues': sorted_residues
        }


def check_current_datamodule_mapping():
    """Check how the current datamodule handles residue mapping."""
    print(f"\n=== Current DataModule Mapping Analysis ===")
    
    print("Current ESM datamodule behavior:")
    print("1. Loads entire ESM sequence embedding (after removing BOS/EOS)")
    print("2. Applies global pooling (mean/max/first) across all residues")
    print("3. Concatenates the same pooled embedding to every SAS point")
    print("4. No residue-specific mapping between ESM positions and CSV residues")
    
    print("\nIssues identified:")
    print("- ESM embeddings are sequence-based (positions 1, 2, 3, ...)")
    print("- CSV residues use PDB numbering (may have gaps, different starting numbers)")
    print("- Current approach loses residue-specific information")
    print("- Missing residues in crystal structure are not handled")


def propose_solution():
    """Propose a solution for proper residue mapping."""
    print(f"\n=== Proposed Solution ===")
    
    print("To properly map ESM embeddings to CSV residue features:")
    print()
    print("1. SEQUENCE ALIGNMENT:")
    print("   - Extract protein sequence from PDB file")
    print("   - Align with ESM sequence to establish position mapping")
    print("   - Handle missing residues in crystal structure")
    print()
    print("2. RESIDUE-SPECIFIC MAPPING:")
    print("   - Map each CSV residue number to corresponding ESM position")
    print("   - Use residue-specific ESM embeddings instead of global pooling")
    print("   - Handle cases where ESM has residues missing from PDB")
    print()
    print("3. FEATURE EXTRACTION:")
    print("   - For each SAS point, use ESM embedding of its nearest residue")
    print("   - Fallback to neighboring residues if exact match not available")
    print("   - Optionally include context from nearby residues")
    print()
    print("4. VALIDATION:")
    print("   - Verify sequence identity between PDB and ESM")
    print("   - Check that all CSV residues have corresponding ESM embeddings")
    print("   - Log any mismatches or missing mappings")


def main():
    """Main analysis function."""
    print("ESM-CSV Residue Mapping Verification")
    print("=" * 50)
    
    # Examine specific protein
    result = examine_protein_mapping("a.001.001.001_1s69a")
    
    # Check current datamodule approach
    check_current_datamodule_mapping()
    
    # Propose solution
    propose_solution()
    
    if result:
        print(f"\n=== Summary for a.001.001.001_1s69a ===")
        print(f"ESM sequence length: {result['esm_residue_count']} residues")
        print(f"CSV unique residues: {result['csv_residue_count']} residues")
        print(f"Missing in PDB: {len(result['missing_residues'])} residues")
        print(f"PDB residue range: {result['csv_residue_range'][0]}-{result['csv_residue_range'][1]}")
        
        if result['missing_residues']:
            print(f"Missing residue numbers: {result['missing_residues']}")
        
        print(f"\nMapping requirement:")
        print(f"Need to map ESM positions 1-{result['esm_residue_count']} to PDB residues {result['csv_residues']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to check if ESM2 embeddings are correctly connected to the residues in CSV data.
This validates that the embedding files match the protein structures in the datasets.
"""

import os
import sys
import pandas as pd
import torch
from glob import glob
from collections import defaultdict

# Add project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

def check_embedding_residue_mapping():
    """Check if ESM embeddings match the residues in CSV data."""
    
    # Paths
    data_dir = "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data"
    esm_dir = os.path.join(data_dir, "esm2")
    
    # Load training data
    train_csv = os.path.join(data_dir, "train", "chen11.csv")
    test_csv = os.path.join(data_dir, "test", "bu48.csv")
    
    print("Checking ESM2 Embedding-Residue Mapping")
    print("=" * 50)
    
    # Check Chen11 training data
    if os.path.exists(train_csv):
        print(f"\nChecking Chen11 training data: {train_csv}")
        df_train = pd.read_csv(train_csv)
        check_dataset_mapping(df_train, esm_dir, "Chen11 Training")
    else:
        print(f"Training CSV not found: {train_csv}")
    
    # Check BU48 test data
    if os.path.exists(test_csv):
        print(f"\nChecking BU48 test data: {test_csv}")
        df_test = pd.read_csv(test_csv)
        check_dataset_mapping(df_test, esm_dir, "BU48 Test")
    else:
        print(f"Test CSV not found: {test_csv}")

def check_dataset_mapping(df, esm_dir, dataset_name):
    """Check mapping for a specific dataset."""
    
    print(f"\n{dataset_name} Dataset Analysis:")
    print("-" * 30)
    
    # Get unique proteins in CSV
    unique_proteins = df['file_name'].str.replace('.pdb', '').unique()
    print(f"Unique proteins in CSV: {len(unique_proteins)}")
    
    # Get available ESM embeddings
    esm_files = glob(os.path.join(esm_dir, "*.pt"))
    esm_proteins = [os.path.basename(f).replace('.pt', '') for f in esm_files]
    
    print(f"Available ESM embeddings: {len(esm_proteins)}")
    
    # Check matches
    missing_embeddings = []
    available_embeddings = []
    
    for protein in unique_proteins:
        if protein in esm_proteins:
            available_embeddings.append(protein)
        else:
            missing_embeddings.append(protein)
    
    print(f"Proteins with embeddings: {len(available_embeddings)}")
    print(f"Proteins missing embeddings: {len(missing_embeddings)}")
    
    if missing_embeddings:
        print(f"\nMissing embeddings for:")
        for protein in sorted(missing_embeddings)[:10]:  # Show first 10
            print(f"  - {protein}")
        if len(missing_embeddings) > 10:
            print(f"  ... and {len(missing_embeddings) - 10} more")
    
    # Check sequence lengths for available embeddings
    if available_embeddings:
        print(f"\nChecking sequence lengths for available embeddings:")
        check_sequence_lengths(df, esm_dir, available_embeddings[:5])  # Check first 5
    
    return len(available_embeddings), len(missing_embeddings)

def check_sequence_lengths(df, esm_dir, proteins_to_check):
    """Check if ESM embedding lengths match expected protein lengths."""
    
    for protein in proteins_to_check:
        # Get residues for this protein from CSV
        protein_data = df[df['file_name'].str.contains(protein)]
        csv_residues = len(protein_data)
        
        # Load ESM embedding
        esm_path = os.path.join(esm_dir, f"{protein}.pt")
        if os.path.exists(esm_path):
            try:
                embedding = torch.load(esm_path, map_location='cpu')
                esm_length = embedding.shape[0] - 2  # Remove BOS/EOS tokens
                
                print(f"  {protein}: CSV residues={csv_residues}, ESM length={esm_length}")
                
                if abs(csv_residues - esm_length) > 5:  # Allow small differences
                    print(f"    ⚠️  Large length difference detected!")
                
            except Exception as e:
                print(f"  {protein}: Error loading embedding - {e}")
        else:
            print(f"  {protein}: ESM file not found")

def check_protein_residue_details():
    """Check specific protein-residue mapping details."""
    
    data_dir = "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data"
    train_csv = os.path.join(data_dir, "train", "chen11.csv")
    esm_dir = os.path.join(data_dir, "esm2")
    
    if not os.path.exists(train_csv):
        print("Training CSV not found for detailed check")
        return
    
    print(f"\nDetailed Protein-Residue Mapping Check:")
    print("-" * 40)
    
    df = pd.read_csv(train_csv)
    
    # Get a sample protein for detailed analysis
    sample_proteins = df['file_name'].str.replace('.pdb', '').unique()[:3]
    
    for protein in sample_proteins:
        print(f"\nAnalyzing {protein}:")
        
        # Get protein data from CSV
        protein_data = df[df['file_name'].str.contains(protein)]
        
        if len(protein_data) > 0:
            print(f"  CSV records: {len(protein_data)}")
            print(f"  Residue range: {protein_data['residue_number'].min()} to {protein_data['residue_number'].max()}")
            print(f"  Chains: {protein_data['chain_id'].unique()}")
            print(f"  Residue types: {len(protein_data['residue_name'].unique())} unique")
            
            # Check ESM embedding
            esm_path = os.path.join(esm_dir, f"{protein}.pt")
            if os.path.exists(esm_path):
                embedding = torch.load(esm_path, map_location='cpu')
                print(f"  ESM shape: {embedding.shape}")
                print(f"  ESM length (without BOS/EOS): {embedding.shape[0] - 2}")
            else:
                print(f"  ESM embedding: Not found")

def main():
    """Main function."""
    print("ESM2 Embedding-Residue Mapping Validation")
    print("=" * 50)
    
    # Check basic mapping
    check_embedding_residue_mapping()
    
    # Check detailed mapping
    check_protein_residue_details()
    
    print(f"\n" + "=" * 50)
    print("Validation complete!")

if __name__ == "__main__":
    main()

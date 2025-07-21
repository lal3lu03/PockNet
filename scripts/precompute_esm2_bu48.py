#!/usr/bin/env python3
"""
Script to precompute ESM-2 embeddings for BU48 dataset.
This script processes PDB files from the BU48 dataset and generates ESM embeddings.
"""

import os
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
import esm
from glob import glob
import sys

# Add project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

# Redirect torch/esm caches to avoid issues
os.environ["TORCH_HOME"] = "/system/user/studentwork/hageneder/cache/torch"
os.environ["TORCH_MODEL_ZOO"] = "/system/user/studentwork/hageneder/cache/torch"
os.environ["TRANSFORMERS_CACHE"] = "/system/user/studentwork/hageneder/cache/transformers"
os.environ["HF_HOME"] = "/system/user/studentwork/hageneder/cache/hf"

# Constants
PDB_DIR = "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data/orginal_pdb/bu48"
OUT_DIR = "/system/user/studentwork/hageneder/MSC/Practical_work/PockNet/data/esm2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Processing PDB files from: {PDB_DIR}")
print(f"Output directory: {OUT_DIR}")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Load ESM-2 model
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")  # Using smaller model for faster processing
model = model.to(DEVICE).eval()
batch_converter = alphabet.get_batch_converter()
print("ESM-2 model loaded successfully!")

# PDB parser
p = PDBParser(QUIET=True)
ppb = PPBuilder()

def pdb_to_seq(pdb_path):
    """Extract protein sequence from PDB file."""
    try:
        structure = p.get_structure("", pdb_path)
        for model in structure:
            for chain in model:
                # Build peptide sequences
                peptides = ppb.build_peptides(chain)
                if peptides:
                    # Take the longest peptide if multiple exist
                    longest_peptide = max(peptides, key=lambda p: len(p.get_sequence()))
                    return str(longest_peptide.get_sequence())
        return None
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return None

def main():
    """Main processing function."""
    pdb_files = glob(os.path.join(PDB_DIR, "*.pdb"))
    print(f"Found {len(pdb_files)} PDB files to process")
    
    processed = 0
    skipped = 0
    
    for pdb_file in sorted(pdb_files):
        pid = os.path.splitext(os.path.basename(pdb_file))[0]
        output_file = os.path.join(OUT_DIR, f"{pid}.pt")
        
        # Skip if already exists
        if os.path.exists(output_file):
            print(f"Skipping {pid} (already exists)")
            continue
            
        print(f"Processing {pid}...")
        
        # Extract sequence
        seq = pdb_to_seq(pdb_file)
        if not seq:
            print(f"  {pid}: no sequence found, skipping")
            skipped += 1
            continue
            
        if len(seq) < 5:  # Skip very short sequences
            print(f"  {pid}: sequence too short ({len(seq)} residues), skipping")
            skipped += 1
            continue
            
        print(f"  {pid}: {len(seq)} residues")
        
        try:
            # Format for ESM batch converter
            batch = [(pid, seq)]
            labels, strs, toks = batch_converter(batch)
            toks = toks.to(DEVICE)
            
            # Extract representations
            with torch.no_grad():
                out = model(toks, repr_layers=[33], return_contacts=False)
            
            # Shape: [batch, seq_len, dim] -> [seq_len, dim]
            reps = out["representations"][33].squeeze(0).cpu()
            
            # Save
            torch.save(reps, output_file)
            print(f"  Saved {pid}.pt with shape {reps.shape}")
            processed += 1
            
        except Exception as e:
            print(f"  Error processing {pid}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed} files")
    print(f"Skipped: {skipped} files")
    print(f"Total ESM embedding files: {len(glob(os.path.join(OUT_DIR, '*.pt')))}")

if __name__ == "__main__":
    main()

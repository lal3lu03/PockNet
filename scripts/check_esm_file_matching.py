# filepath: /system/user/studentwork/hageneder/MSC/Practical_work/PockNet/scripts/check_esm_file_matching.py
import os
import sys
import pandas as pd
import torch
from glob import glob

# Add the project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

# Constants
DATA_DIR = os.path.join(proj_root, "data")
ESM_DIR = os.path.join(DATA_DIR, "esm2")

def check_file_matching():
    """Check if the ESM embeddings can be matched to files in vectorsTrain.csv."""
    print(f"Project root: {proj_root}")
    print(f"Data directory: {DATA_DIR}")
    print(f"ESM directory: {ESM_DIR}")
    
    # Load the CSV file
    train_csv_path = os.path.join(DATA_DIR, "data", "chen11", "vectorsTrain.csv")
    print(f"Loading training data from: {train_csv_path}")
    
    if not os.path.exists(train_csv_path):
        print(f"ERROR: Training file not found at {train_csv_path}")
        return
        
    train_data = pd.read_csv(train_csv_path)
    
    # Check if file_name column exists
    if 'file_name' not in train_data.columns:
        print("ERROR: No file_name column found in the CSV file")
        return
        
    # Get the list of file names from the CSV
    file_names = train_data['file_name'].values
    print(f"Found {len(file_names)} files in the CSV")
    print(f"First 5 files: {file_names[:5]}")
    
    # Get list of ESM embeddings
    esm_files = glob(os.path.join(ESM_DIR, "*.pt"))
    print(f"Found {len(esm_files)} ESM embedding files")
    
    if not esm_files:
        print("ERROR: No ESM embedding files found")
        return
        
    esm_file_ids = [os.path.splitext(os.path.basename(f))[0] for f in esm_files]
    print(f"First 5 ESM files: {esm_file_ids[:5]}")
    
    # Check matching
    matched_count = 0
    unmatched_files = []
    
    for file_name in file_names:
        # Try different versions of the file name
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        found = False
        
        # Try exact match
        if base_name in esm_file_ids:
            found = True
        # Try lowercase
        elif base_name.lower() in esm_file_ids:
            found = True
        # Try without chain identifier (last character)
        elif base_name[:-1] in esm_file_ids:
            found = True
            
        if found:
            matched_count += 1
        else:
            unmatched_files.append(file_name)
    
    print(f"Matched {matched_count} out of {len(file_names)} files ({matched_count/len(file_names)*100:.2f}%)")
    
    if unmatched_files:
        print(f"First 10 unmatched files: {unmatched_files[:10]}")
    
    # Try to determine the pattern for matching
    if unmatched_files:
        print("\nAnalyzing file naming patterns...")
        csv_pattern = analyze_naming_pattern(file_names[:10])
        esm_pattern = analyze_naming_pattern(esm_file_ids[:10])
        
        print(f"CSV file pattern: {csv_pattern}")
        print(f"ESM file pattern: {esm_pattern}")
        
        # Suggest transformation
        print("\nSuggested transformation for matching:")
        if "_" in csv_pattern and "." in esm_pattern:
            print("- Replace underscores with dots in CSV file names")
        elif "." in csv_pattern and "_" in esm_pattern:
            print("- Replace dots with underscores in CSV file names")
        
        # Sample ESM content
        print("\nSample ESM embedding content:")
        sample_embed = torch.load(esm_files[0])
        print(f"Shape: {sample_embed.shape}")
        print(f"Type: {sample_embed.dtype}")

def analyze_naming_pattern(file_names):
    """Analyze the naming pattern of files."""
    has_underscore = any("_" in f for f in file_names)
    has_dot = any("." in f for f in file_names)
    has_chain_id = any(f[-1].isalpha() for f in file_names)
    
    pattern = ""
    if has_underscore:
        pattern += "Contains underscores; "
    if has_dot:
        pattern += "Contains dots; "
    if has_chain_id:
        pattern += "Has chain identifier; "
    
    return pattern

if __name__ == "__main__":
    print("Checking ESM File Matching")
    print("=" * 50)
    check_file_matching()

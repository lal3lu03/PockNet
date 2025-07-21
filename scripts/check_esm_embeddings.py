# filepath: /system/user/studentwork/hageneder/MSC/Practical_work/PockNet/scripts/check_esm_embeddings.py
import os
import sys
import torch
from glob import glob

# Add the project root to the Python path to allow imports
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

# Constants
DATA_DIR = os.path.join(proj_root, "data")
ESM_DIR = os.path.join(DATA_DIR, "esm2")

# Print directory paths for debugging
print(f"Project root: {proj_root}")
print(f"Data directory: {DATA_DIR}")
print(f"ESM directory: {ESM_DIR}")

def list_available_embeddings():
    """Print information about available ESM embeddings."""
    embedding_files = glob(os.path.join(ESM_DIR, "*.pt"))
    print(f"Found {len(embedding_files)} ESM embedding files.")
    
    if embedding_files:
        # List some embedding files
        for i, file in enumerate(embedding_files[:5]):
            print(f"  - {os.path.basename(file)}")
        
        if len(embedding_files) > 5:
            print(f"  - ... and {len(embedding_files) - 5} more")
            
        # Load a sample to get shape information
        sample_embed = torch.load(embedding_files[0])
        print(f"Sample embedding shape: {sample_embed.shape}")
        print(f"Sample embedding dtype: {sample_embed.dtype}")
    return embedding_files

if __name__ == "__main__":
    print("Checking ESM Embeddings")
    print("=" * 50)
    
    # Check available embeddings
    embedding_files = list_available_embeddings()
    
    if not embedding_files:
        print("ERROR: No embedding files found. Please check the ESM directory path.")
        sys.exit(1)
    
    # Try loading and processing some embeddings
    print("\nTesting embedding processing:")
    for file_path in embedding_files[:3]:
        file_id = os.path.basename(file_path).replace('.pt', '')
        embedding = torch.load(file_path)
        
        # Mean pooling
        mean_emb = torch.mean(embedding, dim=0)
        print(f"{file_id}: Original shape {embedding.shape}, Mean-pooled shape {mean_emb.shape}")
        
        # Max pooling
        max_emb = torch.max(embedding, dim=0)[0]
        print(f"{file_id}: Original shape {embedding.shape}, Max-pooled shape {max_emb.shape}")
        
    print("\nESM embedding check completed!")

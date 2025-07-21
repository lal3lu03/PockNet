# filepath: /system/user/studentwork/hageneder/MSC/Practical_work/PockNet/scripts/test_feature_combination.py
import os
import sys
import numpy as np
import pandas as pd
import torch
from glob import glob

# Add project root to Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

# Print Python info
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {proj_root}")
print(f"System path: {sys.path}")

# Set paths
data_dir = os.path.join(proj_root, "data")
esm_dir = os.path.join(data_dir, "esm2")

print(f"Data directory: {data_dir}")
print(f"ESM directory: {esm_dir}")

# List available embeddings
embedding_files = glob(os.path.join(esm_dir, "*.pt"))
print(f"Found {len(embedding_files)} ESM embedding files")
if embedding_files:
    print(f"First 3 embedding files: {[os.path.basename(f) for f in embedding_files[:3]]}")
    
    # Load a sample to check format
    sample_embed = torch.load(embedding_files[0])
    print(f"Sample embedding shape: {sample_embed.shape}")

# Find and read vectors CSV
csv_path = os.path.join(data_dir, "chen11", "vectorsTrain.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(data_dir, "data", "chen11", "vectorsTrain.csv")
    print(f"Trying alternative path {csv_path}...")

if os.path.exists(csv_path):
    print(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV shape: {df.shape}")
    print(f"CSV columns: {df.columns[:5]} ... (truncated)")
    
    if 'file_name' in df.columns:
        print(f"First 3 file names: {df['file_name'].values[:3]}")
else:
    print(f"ERROR: Could not find vectorsTrain.csv file")
        
    def load_embedding(self, file_id):
        """Load ESM embedding for a specific file."""
        # Check if already in cache
        if file_id in self.esm_embeddings:
            return self.esm_embeddings[file_id]
            
        # Clean file ID
        if file_id.endswith('.pdb'):
            base_file_id = file_id[:-4]  # Remove '.pdb'
        else:
            base_file_id = os.path.basename(file_id)
            
        embedding_path = os.path.join(self.esm_dir, f"{base_file_id}.pt")
        
        if not os.path.exists(embedding_path):
            print(f"Warning: No embedding found for {file_id} (tried {embedding_path})")
            sample_files = glob(os.path.join(self.esm_dir, "*.pt"))
            if len(sample_files) > 0:
                sample_embed = torch.load(sample_files[0])
                return torch.zeros(sample_embed.size(-1))
            else:
                return torch.zeros(2560)
                
        # Load embedding
        embedding = torch.load(embedding_path)
        
        # Process according to embedding_type
        if self.embedding_type == "mean":
            processed_embedding = torch.mean(embedding, dim=0)
        elif self.embedding_type == "max":
            processed_embedding = torch.max(embedding, dim=0)[0]
        elif self.embedding_type == "first":
            processed_embedding = embedding[1] if embedding.size(0) > 1 else embedding[0]
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
            
        # Store in cache
        self.esm_embeddings[file_id] = processed_embedding
        return processed_embedding
    
    def combine_features(self, n_samples=10):
        """Combine vector features with ESM embeddings."""
        # Load sample data from CSV
        csv_path = os.path.join(self.data_dir, "chen11", "vectorsTrain.csv")
        print(f"Loading data from {csv_path}...")
        
        if not os.path.exists(csv_path):
            csv_path = os.path.join(self.data_dir, "data", "chen11", "vectorsTrain.csv")
            print(f"Trying alternative path {csv_path}...")
            
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find vectorsTrain.csv file")
            
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Check for file_name column
        if 'file_name' not in df.columns:
            print("ERROR: No file_name column found in CSV")
            return
            
        # Use only a subset for testing
        df_sample = df.head(n_samples)
        file_names = df_sample['file_name'].values
        
        # Get feature vectors without file_name and class
        features = df_sample.drop(['file_name', 'class'], axis=1).values
        labels = df_sample['class'].values
        
        print(f"\nSample data:")
        print(f"- Features shape: {features.shape}")
        print(f"- Labels shape: {labels.shape}")
        print(f"- First 3 file names: {file_names[:3]}")
        
        # Load ESM embeddings for each file
        print("\nLoading ESM embeddings...")
        embeddings = []
        for file_id in file_names:
            print(f"Loading embedding for {file_id}...")
            embedding = self.load_embedding(file_id)
            embeddings.append(embedding)
            print(f"  Embedding shape: {embedding.shape}")
            
        # Convert embeddings to numpy array
        embeddings_np = np.stack([e.numpy() for e in embeddings])
        print(f"Embeddings array shape: {embeddings_np.shape}")
        
        # Combine features and embeddings
        combined_features = np.hstack([features, embeddings_np])
        print(f"Combined features shape: {combined_features.shape}")
        
        return combined_features, labels, file_names
        
def main():
    # Set paths
    data_dir = os.path.join(proj_root, "data")
    esm_dir = os.path.join(data_dir, "esm2")
    
    print(f"Project root: {proj_root}")
    print(f"Data directory: {data_dir}")
    print(f"ESM directory: {esm_dir}")
    
    # List available embeddings
    embedding_files = glob(os.path.join(esm_dir, "*.pt"))
    print(f"Found {len(embedding_files)} ESM embedding files")
    if embedding_files:
        print(f"First 3 embedding files: {[os.path.basename(f) for f in embedding_files[:3]]}")
        
        # Load a sample to check format
        sample_embed = torch.load(embedding_files[0])
        print(f"Sample embedding shape: {sample_embed.shape}")
    
    # Test combining features
    combiner = ESMFeatureCombiner(data_dir, esm_dir, embedding_type="mean")
    combined_features, labels, file_names = combiner.combine_features(n_samples=5)
    
    # Print results
    print("\nCombined Feature Summary:")
    print(f"- Original features + ESM embeddings: {combined_features.shape[1]} dimensions")
    print(f"- First 3 combined feature vectors shape: {combined_features[:3].shape}")
    
if __name__ == "__main__":
    main()

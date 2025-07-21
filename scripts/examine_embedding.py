import torch
import os

# Path to an embedding file
embedding_path = "data/esm2/a.001.001.001_1s69a.pt"

# Check if file exists
if os.path.exists(embedding_path):
    print(f"Loading embedding from {embedding_path}")
    emb = torch.load(embedding_path)
    print(f"Type: {type(emb)}")
    print(f"Shape: {emb.shape if hasattr(emb, 'shape') else 'No shape attribute'}")
    if isinstance(emb, dict):
        print("Dictionary keys:", emb.keys())
        for key, value in emb.items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    Shape: {value.shape}")
    elif isinstance(emb, torch.Tensor):
        print(f"Min: {emb.min().item()}")
        print(f"Max: {emb.max().item()}")
        print(f"Mean: {emb.mean().item()}")
else:
    print(f"File not found: {embedding_path}")

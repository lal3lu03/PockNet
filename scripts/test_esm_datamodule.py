# filepath: /system/user/studentwork/hageneder/MSC/Practical_work/PockNet/scripts/test_esm_datamodule.py
import os
import sys
import torch
from glob import glob

# Add the project root to the Python path to allow imports
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

# Import our custom data module
from src.data.binding_site_esm_datamodule import BindingSiteESMDataModule

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
        # Load a sample to get shape information
        sample_embed = torch.load(embedding_files[0])
        print(f"Sample embedding shape: {sample_embed.shape}")
        print(f"Sample embedding file: {os.path.basename(embedding_files[0])}")
    return embedding_files

def test_datamodule():
    """Test the ESM data module functionality."""
    print("Creating BindingSiteESMDataModule...")
    datamodule = BindingSiteESMDataModule(
        data_dir=DATA_DIR,
        esm_dir=ESM_DIR,
        embedding_type="mean",
        batch_size=32,
        sampling_strategy="none",
    )
    
    print("Setting up datamodule...")
    datamodule.setup()
    
    print("Testing dataloaders...")
    # Get a batch from the train loader
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    x, y = batch
    
    print(f"Batch shapes - X: {x.shape}, y: {y.shape}")
    print(f"X feature dimension: {x.shape[1]}")
    
    # Check val and test loaders
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    val_x, val_y = val_batch
    print(f"Validation batch shapes - X: {val_x.shape}, y: {val_y.shape}")
    
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    test_x, test_y = test_batch
    print(f"Test batch shapes - X: {test_x.shape}, y: {test_y.shape}")
    
    print("DataModule test completed successfully!")

if __name__ == "__main__":
    print("Testing ESM DataModule Integration")
    print("=" * 50)
    
    # Check available embeddings
    print("Checking available ESM embeddings:")
    embedding_files = list_available_embeddings()
    
    if not embedding_files:
        print("ERROR: No embedding files found. Please check the ESM directory path.")
        sys.exit(1)
    
    # Test the data module
    test_datamodule()

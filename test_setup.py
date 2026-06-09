import torch
import sys
from src.dataset import get_dataloaders
import src.config as config

def verify_data_pipeline():
    print("--- Booting up DataLoader Verification ---\n")
    
    try:
        # We'll spin up the loaders using our training CSV and Image directory.
        # This will also test if our train/val split logic is functioning.
        train_loader, val_loader = get_dataloaders(
            train_csv=config.TRAIN_CSV, 
            train_img_dir=config.TRAIN_IMG_DIR, 
            batch_size=config.BATCH_SIZE
        )
    except FileNotFoundError as e:
        print(f"❌ Path Error: {e}")
        print("Double-check that your 'data/train/labels.csv' and 'images' folder exist.")
        sys.exit(1)

    print(f"✓ Loaders initialized.")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}\n")

    print("--- Pulling a single batch to inspect the pipeline ---")
    
    # Grab just the first batch to see what we're actually feeding the beast
    images, labels, label_lengths = next(iter(train_loader))
    
    # 1. Inspect Image Tensor
    print("\n1. Image Tensor:")
    print(f"   Shape: {images.shape}")
    
    expected_img_shape = torch.Size([config.BATCH_SIZE, 1, config.IMG_HEIGHT, config.IMG_WIDTH])
    if images.shape == expected_img_shape:
        print("   ✓ Shape is perfect for the CNN backbone.")
    else:
        print(f"   ❌ Shape mismatch! Expected {expected_img_shape}")

    # Check normalization range (should be roughly between -1.0 and 1.0)
    print(f"   Pixel value range: min={images.min().item():.2f}, max={images.max().item():.2f}")

    # 2. Inspect Label Tensor (CTC expects a flattened 1D array)
    print("\n2. Label Tensor (Flattened for CTC):")
    print(f"   Shape: {labels.shape}")
    
    expected_label_shape = torch.Size([config.BATCH_SIZE * config.MAX_SEQ_LEN])
    if labels.shape == expected_label_shape:
        print("   ✓ Labels are correctly flattened into 1D space.")
    else:
        print(f"   ❌ Label shape mismatch! Expected {expected_label_shape}")

    # 3. Inspect Label Lengths
    print("\n3. Label Lengths Tensor:")
    print(f"   Shape: {label_lengths.shape}")
    print(f"   Sample lengths (first 5): {label_lengths[:5].tolist()}")
    
    print("\n=== DataLoader Verification Complete! ===")

if __name__ == "__main__":
    verify_data_pipeline()
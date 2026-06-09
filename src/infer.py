import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2

import src.config as config
from src.model import CRNN
from src.utils import LabelEncoder, ctc_greedy_decode
from src.transforms import get_val_transforms

class TestDataset(Dataset):
    """A simplified dataset loader just for the unlabelled test images."""
    def __init__(self, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        # Get all PNG files and sort them so they appear in order (test-0.png, test-1.png...)
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, img_name

def generate_submission():
    print("--- Booting Inference Pipeline ---")
    
    # 1. Load the Test Data
    test_dataset = TestDataset(config.TEST_IMG_DIR, get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model and Load the Best Weights
    encoder = LabelEncoder()
    model = CRNN(num_classes=encoder.num_classes).to(config.DEVICE)
    
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Could not find trained weights at {best_model_path}")
        
    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    model.eval()
    print("✓ Best model weights loaded successfully.")

    # 3. Predict on Test Set
    results = []
    print("Generating predictions...")
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Testing"):
            images = images.to(config.DEVICE)
            
            # Forward pass
            log_probs = model(images)
            
            # Decode the sequence
            predictions = ctc_greedy_decode(log_probs, encoder)
            
            # Pair the image name with its prediction
            for name, pred in zip(img_names, predictions):
                results.append({"image": name, "prediction": pred})

    # 4. Save to CSV
    submission_path = os.path.join(config.BASE_DIR, "submission.csv")
    df = pd.DataFrame(results)
    df.to_csv(submission_path, index=False)
    
    print(f"\n🎉 Inference Complete! Submission saved to: {submission_path}")
    print("Format check:")
    print(df.head())

if __name__ == "__main__":
    generate_submission()
import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import src.config as config
from src.utils import LabelEncoder
from src.transforms import get_train_transforms, get_val_transforms

class DistortedTextDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transforms=None):
        """
        Custom Dataset loader for the distorted text sequences.
        
        Args:
            csv_file: Path to the labels CSV.
            img_dir: Path to the directory containing the PNG images.
            transforms: Torchvision transforms pipeline.
        """
        self.df = pd.read_csv(csv_file, dtype=str)
        self.img_dir = img_dir
        self.transforms = transforms
        self.encoder = LabelEncoder()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # The CSV has columns: 'image' and 'text'
        img_name = self.df.iloc[idx]['image']
        raw_text = str(self.df.iloc[idx]['text'])
        
        # --- BULLETPROOF DATA SANITIZATION ---
        # 1. Fix Pandas turning purely numeric labels into floats (e.g. "123456.0" -> "123456")
        if raw_text.endswith(".0"):
            raw_text = raw_text[:-2]
            
        # 2. Force uppercase and violently strip out ANY character not in our official vocab
        # This protects us from hidden spaces, periods, or dirty data traps in the CSV
        text_label = "".join([char for char in raw_text.upper() if char in config.VOCAB])
        # -------------------------------------
        
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image in grayscale using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")
            
        # Resize all images to the fixed CRNN input size (32x128)
        image = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        if self.transforms:
            image = self.transforms(image)

        # Encode the sanitized text string into an integer tensor
        encoded_label = self.encoder.encode(text_label)
        label_length = torch.tensor(len(text_label), dtype=torch.long)

        return image, encoded_label, label_length

def ctc_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    CTC Loss requires the targets to be flattened into a 1D tensor, 
    and it needs a separate tuple of lengths to reconstruct them.
    """
    images, labels, label_lengths = zip(*batch)
    
    # Stack images into a standard 4D tensor: (Batch, Channels, Height, Width)
    images = torch.stack(images, dim=0)
    
    # Concatenate all label tensors into a single 1D tensor
    labels = torch.cat(labels, dim=0)
    
    # Stack the lengths
    label_lengths = torch.stack(label_lengths, dim=0)
    
    return images, labels, label_lengths

def get_dataloaders(train_csv, train_img_dir, batch_size=config.BATCH_SIZE, val_split=0.1):
    """
    Splits the training dataset into train/val and returns DataLoaders.
    """
    # Load the full dataset (temporarily without transforms to handle the split)
    full_dataset = DistortedTextDataset(train_csv, train_img_dir, transforms=None)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Randomly split the indices
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Fixed seed for reproducibility
    )
    
    # Re-apply the specific transforms to the splits
    train_dataset.dataset.transforms = get_train_transforms()
    val_dataset.dataset.transforms = get_val_transforms()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        collate_fn=ctc_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        collate_fn=ctc_collate_fn
    )
    
    return train_loader, val_loader
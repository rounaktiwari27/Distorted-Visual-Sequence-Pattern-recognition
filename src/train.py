import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

import src.config as config
from src.dataset import get_dataloaders
from src.model import CRNN
from src.utils import LabelEncoder, ctc_greedy_decode, compute_cer

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Handles one full pass through the training data."""
    model.train() # Set model to training mode (enables dropout & batchnorm updates)
    epoch_loss = 0.0
    
    # tqdm gives us a nice progress bar in the terminal
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, labels, label_lengths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Forward Pass
        optimizer.zero_grad()
        log_probs = model(images) # Output: [Time=32, Batch, Classes=37]
        
        # CTC Loss requires the lengths of the inputs (Time) and the targets (label_lengths)
        batch_size = images.size(0)
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long)
        
        # 2. Compute Loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        
        # 3. Backward Pass (Calculate gradients)
        loss.backward()
        
        # 4. Gradient Clipping (Crucial for LSTMs to prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # 5. Optimize (Update weights)
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, encoder, device):
    """Evaluates the model on unseen data using CER without updating weights."""
    model.eval() # Set model to evaluation mode (freezes dropout & batchnorm)
    epoch_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad(): # Disable gradient tracking to save memory/speed
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels, label_lengths in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            log_probs = model(images)
            
            # Loss Calculation
            batch_size = images.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            epoch_loss += loss.item()
            
            # Decode Predictions for CER
            decoded_preds = ctc_greedy_decode(log_probs, encoder)
            all_predictions.extend(decoded_preds)
            
            # Reconstruct target strings from flattened tensor
            start = 0
            for length in label_lengths:
                target_tensor = labels[start:start+length].tolist()
                target_string = encoder.decode(target_tensor)
                all_targets.append(target_string)
                start += length
                
    # Calculate global Character Error Rate for the epoch
    val_cer = compute_cer(all_predictions, all_targets)
    return epoch_loss / len(dataloader), val_cer

def main():
    print(f"--- Booting Distorted Text Training Pipeline ---")
    print(f"Device: {config.DEVICE}")
    
    # 1. Initialize Encoders and DataLoaders
    encoder = LabelEncoder()
    train_loader, val_loader = get_dataloaders(
        train_csv=config.TRAIN_CSV, 
        train_img_dir=config.TRAIN_IMG_DIR
    )
    
    # 2. Initialize Model, Loss, and Optimizer
    model = CRNN(num_classes=encoder.num_classes).to(config.DEVICE)
    
    # zero_infinity=True prevents NaN losses on misaligned batches
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) 
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 3. Training State Tracking
    best_val_cer = float('inf')
    patience_counter = 0
    patience_limit = 10
    history = []
    
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    log_file_path = os.path.join(config.LOG_DIR, "training_log.csv")

    # 4. Main Training Loop
    print("\nStarting Training...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_cer = validate_epoch(model, val_loader, criterion, encoder, config.DEVICE)
        
        print(f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val CER: {val_cer:.4f}")
        
        # Log to history
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_cer": val_cer})
        pd.DataFrame(history).to_csv(log_file_path, index=False)
        
        # Save checkpoint if CER improves
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  🌟 New best CER! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs. No improvement for 10 epochs.")
                break

if __name__ == "__main__":
    main()
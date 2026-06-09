import os
import torch

# ==============================================================================
# Path Configurations
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

# Updated to match your exact folder structure!
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train", "train-labels.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs") 
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Ensuring output directories exist so saving weights don't crash mid-training
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# ==============================================================================
# Dataset / Vocabulary Configuration
# ==============================================================================
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Max length observation 
MAX_SEQ_LEN = 6

# ==============================================================================
# Model Architecture & Training Hyperparameters
# ==============================================================================
# Image dimensions ->CRNN network structure
IMG_HEIGHT = 32
IMG_WIDTH = 128
IMG_CHANNELS = 1  # Grayscale processing

# Optimization settings
BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 3e-4
NUM_WORKERS = 4  

# Hardware acc.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
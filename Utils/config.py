# config.py

import torch

# -- DATA SETTINGS --
# Paths for data storage
ORIGINAL_DATA_DIR = '/content/braint_original'
BASE_DATA_DIR = '/content/data'
DATASET_URL = "mrnotalent/braint" # Kaggle dataset identifier

# -- MODEL AND TRAINING SETTINGS --
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.5
HIDDEN_UNITS = 1024

# -- FINE-TUNING SETTINGS --
UNFREEZE_EPOCH = 5 # Epoch at which to unfreeze the backbones
FINETUNE_LR = 1e-4

# -- SCHEDULER SETTINGS --
SCHEDULER_T_MAX = 10
FINETUNE_SCHEDULER_T_MAX = 15

# -- MISC SETTINGS --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BEST_MODEL_PATH = "hybrid_brain_tumor_best.pth"
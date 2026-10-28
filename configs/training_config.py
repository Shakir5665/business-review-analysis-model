"""
Training Configuration

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

# --------------------------------------------------
# Dataset Split
# --------------------------------------------------

TRAIN_SIZE = 0.80

VALIDATION_SIZE = 0.10

TEST_SIZE = 0.10

# --------------------------------------------------
# Random Seed
# --------------------------------------------------

RANDOM_STATE = 42

# --------------------------------------------------
# DataLoader
# --------------------------------------------------

BATCH_SIZE = 32

NUM_WORKERS = 0

PIN_MEMORY = False

SHUFFLE_TRAIN = True

SHUFFLE_VALIDATION = False

SHUFFLE_TEST = False


# --------------------------------------------------
# Training
# --------------------------------------------------

BATCH_SIZE = 64

NUM_EPOCHS = 100

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

# --------------------------------------------------
# Gradient
# --------------------------------------------------

GRADIENT_CLIP = 1.0

# --------------------------------------------------
# Scheduler
# --------------------------------------------------

WARMUP_RATIO = 0.1


# --------------------------------------------------
# Checkpoints
# --------------------------------------------------

CHECKPOINT_DIR = "checkpoints"

BEST_MODEL_NAME = "best_model.pt"

LATEST_CHECKPOINT_NAME = "latest_checkpoint.pt"   

GDRIVE_DIR = "/content/drive/MyDrive/AI-Business-Risk-Analysis-System"

GDRIVE_CHECKPOINT_DIR = f"{GDRIVE_DIR}/checkpoints"

GDRIVE_REPORTS_DIR = f"{GDRIVE_DIR}/outputs/reports"

GDRIVE_PLOTS_DIR = f"{GDRIVE_DIR}/outputs/plots"

# --------------------------------------------------
# Early Stopping
# --------------------------------------------------

EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 5

MIN_DELTA = 0.001
from pathlib import Path
import torch

# ----------------
# Repro & Paths
# ----------------
SEED = 1234

PROJECT_DIR = Path("./kaggle_model")
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"   # fixed path join
DATA_ROOT = Path("../All_files/FIT5215_Datasetv2/FIT5215_Dataset")
TEST_DIR  = Path("../fit-5215-object-detection-s-2-2025/test_set/official_test")
SPLIT_FILE = Path("../split_indices.json")

# ----------------
# Splits
# ----------------
SPLIT_RATIOS = (0.80, 0.10, 0.10)

# ----------------
# Model & Train
# ----------------
MODEL_NAME = "vit_base_patch16_224"

NUM_EPOCHS = 30
FREEZE_EPOCHS = 0
PATIENCE = 10
BATCH_SIZE = 32
NUM_WORKERS = 4

# ViT-friendly defaults (balanced clean/robust)
LR = 1e-4                 # head LR; backbone uses a multiplier (see main.py)
WEIGHT_DECAY = 0.05       # was 0.10; 0.05 tends to help clean acc on ViTs
DROP_RATE = 0.10          # was 0.20; ViTs often don't need extra dropout
DROP_PATH = 0.10
MAX_GRAD_NORM = 1.0

# ----------------
# Adversarial Train (pixel space in [0,1])
# ----------------
EPS   = 8.0 / 255.0       # 0.03137
ALPHA = 0.5 / 255.0       # 0.00196
PGD_STEPS = 20
WARMUP_FGSM_EPOCHS = 5

# Curriculum (PGD ramp)
CURRICULUM = True
CURR_USE_ALPHA_FRACTION = True
CURR_ALPHA_FRAC = 1/4     # α starts at EPS * 1/4 and ramps to ALPHA

# Adversarial-vs-clean loss weight ramp
ADV_MAX_WEIGHT   = 0.5    # final weight on adversarial loss
ADV_RAMP_EPOCHS  = 10     # epochs to ramp lam_adv from 0 → ADV_MAX_WEIGHT

# ----------------
# Augment
# ----------------
USE_MIXUP = False
MIXUP_ALPHA = 0.2

USE_CUTMIX = True
CUTMIX_ALPHA = 0.5        # softened from 1.0
CUTMIX_PROB  = 0.5        # apply CutMix on ~50% batches

LABEL_SMOOTH = 0.1        # label smoothing improves ViT clean acc

# ----------------
# Inference
# ----------------
USE_TTA = True

# ----------------
# Device
# ----------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------
# Fast-mode knobs (optional runtime overrides via CLI)
# ----------------
TARGET_RES = None         # e.g., set at runtime to 192 for cooler runs
CAP_STEPS  = None         # e.g., cap batches per epoch to 100

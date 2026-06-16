"""Enhanced configuration for improved pipeline."""

from pathlib import Path

# Inherit from original config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import *

# Enhanced decomposition parameters
N_COMPONENTS_IMPROVED = 8  # Increased from 6 for better source separation
NMF_MAX_ITER_IMPROVED = 300  # More iterations for convergence
NMF_INIT_IMPROVED = "nndsvda"  # Better initialization

# Improved segmentation
SEG_LEN_IMPROVED = 512  # Longer segments for better context (was 256)
SEG_OVERLAP_IMPROVED = 0.5

# Enhanced GAN training
WGAN_EPOCHS_IMPROVED = 300  # More epochs for better convergence
WGAN_BATCH_IMPROVED = 128  # Larger batch for stability
WGAN_LR_IMPROVED = 5e-5  # Lower LR for fine-tuning
MAX_SEGMENTS_IMPROVED = 8000  # More training data per category

# Improved classification
CLASSIFIER_EPOCHS_IMPROVED = 50  # More training
CLASSIFIER_BATCH_IMPROVED = 32
CLASSIFIER_LR_IMPROVED = 5e-4
CLASSIFIER_DROPOUT_IMPROVED = 0.3  # Regularization

# Weight optimization
WEIGHT_OPT_ITERATIONS = 100
WEIGHT_OPT_LR = 1e-3

# Output directories
OUTPUTS_IMPROVED = OUTPUTS / "improved_pipeline"
OUTPUTS_IMPROVED.mkdir(exist_ok=True, parents=True)

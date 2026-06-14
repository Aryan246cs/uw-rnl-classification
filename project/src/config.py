"""Central configuration for the DeepShip pipeline."""

import os
from pathlib import Path

# Root paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE_ROOT  # WAV files live directly in class folders at repo root

CLASS_DIRS = {
    "Cargo": WORKSPACE_ROOT / "Cargo",
    "Passengership": WORKSPACE_ROOT / "Passengership",
    "Tanker": WORKSPACE_ROOT / "Tanker",
    "Tug": WORKSPACE_ROOT / "Tug",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS_WAVEFORMS = OUTPUTS / "waveforms"
OUTPUTS_FFT = OUTPUTS / "fft"
OUTPUTS_PSD = OUTPUTS / "psd"
OUTPUTS_SPECTROGRAMS = OUTPUTS / "spectrograms"
OUTPUTS_FEATURES = OUTPUTS / "features"
OUTPUTS_REPORTS = OUTPUTS / "reports"

# Preprocessing
TARGET_SR = None  # None = keep native sample rate
BANDPASS_LOW = 10  # Hz
BANDPASS_HIGH = 1000  # Hz
BANDPASS_ORDER = 4

# FFT
FFT_WINDOW = "hann"
N_TOP_PEAKS = 10

# STFT / Spectrogram
N_FFT = 2048
HOP_LEN = 512
N_MELS = 128

# PSD (Welch)
WELCH_NPERSEG = 1024

# Feature extraction
N_MFCC = 13

# ── GenAI / Phase 10-12 configuration (Appendix A, scaled for the 33-file
#    working subset; see outputs/reports/final_report.md for justification).
#    GPU (CUDA, Blackwell sm_120) is available, so epoch counts / sample
#    caps are set closer to the Appendix A reference values than a CPU-only
#    run would allow. ──────────────────────────────────────────────────────

OUTPUTS_GENAI = OUTPUTS / "genai"
OUTPUTS_MAPPING = OUTPUTS / "mapping"
OUTPUTS_CLASSIFICATION = OUTPUTS / "classification"

# Segmentation (Phase 2 spec): 3s analysis frames, 50% overlap
SEG_DURATION = 3.0  # seconds

# 10A — WGAN-GP (residual noise)
RESIDUAL_SEGMENT_LEN = 256
MAX_SEGMENTS_PER_CLASS = 4000  # subsample cap
Z_DIM_GAN = 128
WGAN_CRITIC_ITERS = 5
WGAN_LAMBDA_GP = 10.0
WGAN_EPOCHS = 200
WGAN_BATCH = 64
WGAN_LR = 1e-4

# 10C — Beta-VAE (latent source representation)
Z_DIM_VAE = 64
VAE_BETA = 4.0
VAE_EPOCHS = 150
VAE_BATCH = 64
VAE_LR = 1e-3

# 10D — TimeGAN (temporal sequence synthesis)
TIMEGAN_SEQ_LEN = 64
TIMEGAN_HIDDEN = 128
TIMEGAN_EPOCHS = 100
TIMEGAN_BATCH = 32
MAX_SEQUENCES_PER_CLASS = 800

# 10B — Conditional DDPM (spectrogram patch synthesis)
DIFFUSION_T = 1000
DDIM_STEPS = 50
GUIDANCE_SCALE = 3.0
DIFFUSION_PATCH = 32  # mel x time patch size
DIFFUSION_EPOCHS = 120
DIFFUSION_BATCH = 32
DIFFUSION_LR = 2e-4

# Phase 11 — synthetic corpus generation
SYNTH_WGAN_PER_CLASS = 2000
SYNTH_DDPM_PER_CLASS = 200
SYNTH_TIMEGAN_PER_CLASS = 500
SYNTH_KL_THRESHOLD = 0.15

# Phase 12 — classification
CLASSIFIER_BATCH = 16
CLASSIFIER_LR = 1e-3
CLASSIFIER_EPOCHS = 30
CLASSIFIER_FOLDS = 3
MEL_PATCH_SIZE = 64  # square mel-spectrogram patch fed to classifiers
MAX_SEGMENTS_PER_FILE_CLF = 32  # cap segments/file for training

# Phase 7c — machinery-component classification (GAN-augmented)
OUTPUTS_COMPONENT_CLF = OUTPUTS / "component_classification"
COMPONENT_MAX_SEGMENTS_PER_CATEGORY = 4000  # subsample cap, mirrors MAX_SEGMENTS_PER_CLASS
COMPONENT_MIN_SEGMENTS_FOR_GAN = 16  # below this, skip WGAN-GP for that category
COMPONENT_SYNTH_PER_CATEGORY = 1000  # synthetic segments generated per category

# Phase 12 — RNL-SBN mapping
ASSUMED_RANGES_M = [500, 1000, 2000, 5000]  # hydrophone-to-ship ranges (m)

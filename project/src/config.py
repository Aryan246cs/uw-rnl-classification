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

"""Phase 1: Load and preprocess single ship type."""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from pathlib import Path
from src.config import CLASS_DIRS, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER, TARGET_SR


def run_preprocessing_single(ship_type: str):
    """Load and preprocess all files for a single ship type."""
    class_dir = CLASS_DIRS[ship_type]
    wav_files = sorted(class_dir.glob("*.wav"))
    
    if not wav_files:
        raise ValueError(f"No WAV files found in {class_dir}")
    
    preprocessed = {}
    sample_rate = None
    
    for wav_path in wav_files:
        # Load audio
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        
        if sample_rate is None:
            sample_rate = sr
        
        # Apply bandpass filter
        sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], 
                     btype='band', fs=sr, output='sos')
        y_filt = sosfilt(sos, y)
        
        # Normalize
        if np.max(np.abs(y_filt)) > 0:
            y_filt = y_filt / np.max(np.abs(y_filt))
        
        preprocessed[str(wav_path)] = y_filt
        print(f"  Loaded: {wav_path.name} ({len(y_filt)/sr:.1f}s)")
    
    return preprocessed, sample_rate

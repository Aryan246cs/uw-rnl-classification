"""Phase 2: Enhanced spectral analysis for single ship type."""

import numpy as np
import librosa
from src.config import N_FFT, HOP_LEN, N_MELS, WELCH_NPERSEG


def run_spectral_single(preprocessed: dict, sr: int):
    """Compute enhanced spectral features for all files."""
    spectral = {}
    
    for fpath, signal in preprocessed.items():
        # STFT
        S = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LEN)
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            S=S_mag**2, sr=sr, n_mels=N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # PSD (Welch)
        from scipy.signal import welch
        freqs, psd = welch(signal, fs=sr, nperseg=WELCH_NPERSEG)
        
        spectral[fpath] = {
            'stft_mag': S_mag,
            'stft_phase': S_phase,
            'mel_spec': mel_spec_db,
            'psd_freqs': freqs,
            'psd': psd,
            'signal': signal,
            'sr': sr
        }
    
    return spectral

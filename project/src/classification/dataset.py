"""
Phase 12b - Classifier dataset construction.
Segments each preprocessed signal into 3s / 50% overlap windows (Phase 2
spec) and computes, per segment:
  - a (MEL_PATCH_SIZE x MEL_PATCH_SIZE) log-mel spectrogram patch (CNN/ResNet/CRNN input)
  - a (MEL_PATCH_SIZE x N_MFCC) MFCC sequence (Transformer input)

The "augmented" dataset additionally injects WGAN-GP-synthesised residual
noise into each real segment before feature extraction, implementing the
Phase 11 synthesis formula x_synth(t) = x_real(t) + eps_synth(t).
"""

import numpy as np
import librosa
from pathlib import Path
from scipy.ndimage import zoom

from src.config import (
    CLASS_DIRS,
    N_FFT,
    HOP_LEN,
    N_MELS,
    N_MFCC,
    SEG_DURATION,
    MEL_PATCH_SIZE,
    MAX_SEGMENTS_PER_FILE_CLF,
)

CLASS_NAMES = list(CLASS_DIRS.keys())


def _resize(arr: np.ndarray, shape: tuple) -> np.ndarray:
    factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    return zoom(arr, factors, order=1)


def _segment_signal(signal: np.ndarray, sr: int, max_segments: int = MAX_SEGMENTS_PER_FILE_CLF):
    seg_len = int(SEG_DURATION * sr)
    if seg_len <= 0 or len(signal) < seg_len:
        return [signal] if len(signal) > 0 else []
    hop = seg_len // 2
    segments = [signal[s : s + seg_len] for s in range(0, len(signal) - seg_len + 1, hop)]
    if len(segments) > max_segments:
        idx = np.linspace(0, len(segments) - 1, max_segments, dtype=int)
        segments = [segments[i] for i in idx]
    return segments


def _segment_to_features(segment: np.ndarray, sr: int):
    mel = librosa.feature.melspectrogram(y=segment.astype(np.float32), sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-12)
    mel_patch = _resize(mel_norm, (MEL_PATCH_SIZE, MEL_PATCH_SIZE)).astype(np.float32)

    mfcc = librosa.feature.mfcc(y=segment.astype(np.float32), sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN)
    mfcc_norm = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-12)
    mfcc_seq = _resize(mfcc_norm.T, (MEL_PATCH_SIZE, N_MFCC)).astype(np.float32)  # (T, N_MFCC)

    return mel_patch, mfcc_seq


def build_dataset(preprocessed: dict, wgan_models: dict = None, augment: bool = False, noise_scale: float = 0.15):
    """
    Returns dict with keys: mel (N,P,P), mfcc (N,P,N_MFCC), y (N,), groups (N,), is_synthetic (N,)
    If augment=True and wgan_models provided, each real segment is duplicated
    with WGAN-GP residual noise injected (x_synth = x_real + noise_scale * eps_synth).
    """
    mel_list, mfcc_list, y_list, group_list, synth_list = [], [], [], [], []

    for file_idx, (fpath, (signal, sr)) in enumerate(preprocessed.items()):
        cls = Path(fpath).parent.name
        cls_idx = CLASS_NAMES.index(cls)

        for segment in _segment_signal(signal, sr):
            mel_patch, mfcc_seq = _segment_to_features(segment, sr)
            mel_list.append(mel_patch); mfcc_list.append(mfcc_seq)
            y_list.append(cls_idx); group_list.append(file_idx); synth_list.append(0)

            if augment and wgan_models and cls in wgan_models:
                G = wgan_models[cls]["generator"]
                from src.genai.wgan_gp import generate_samples
                from src.config import Z_DIM_GAN, RESIDUAL_SEGMENT_LEN

                eps = generate_samples(G, n_samples=1, z_dim=Z_DIM_GAN)[0]
                reps = int(np.ceil(len(segment) / len(eps)))
                eps_tiled = np.tile(eps, reps)[: len(segment)]
                seg_std = segment.std() + 1e-12
                eps_scaled = eps_tiled * seg_std * noise_scale
                synth_segment = (segment + eps_scaled).astype(np.float32)

                mel_s, mfcc_s = _segment_to_features(synth_segment, sr)
                mel_list.append(mel_s); mfcc_list.append(mfcc_s)
                y_list.append(cls_idx); group_list.append(file_idx); synth_list.append(1)

    return {
        "mel": np.array(mel_list, dtype=np.float32),
        "mfcc": np.array(mfcc_list, dtype=np.float32),
        "y": np.array(y_list, dtype=np.int64),
        "groups": np.array(group_list, dtype=np.int64),
        "is_synthetic": np.array(synth_list, dtype=np.int64),
    }

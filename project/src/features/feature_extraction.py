"""
Phase 4 – Feature Extraction
Spectral, harmonic, energy and statistical features → features.csv
"""

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis, skew

from src.config import (
    N_FFT,
    HOP_LEN,
    N_MFCC,
    OUTPUTS_FEATURES,
)


# ── Spectral features ────────────────────────────────────────────────────────


def spectral_features(signal: np.ndarray, sr: int) -> dict:
    y = signal.astype(np.float32)
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN
    )[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LEN)[
        0
    ]

    # FFT peak frequencies
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    peak_idx = np.argpartition(spectrum, -5)[-5:]
    peak_freqs = np.sort(freqs[peak_idx])

    feats = {
        "spec_centroid_mean": float(centroid.mean()),
        "spec_centroid_std": float(centroid.std()),
        "spec_bandwidth_mean": float(bandwidth.mean()),
        "spec_bandwidth_std": float(bandwidth.std()),
        "spec_rolloff_mean": float(rolloff.mean()),
        "spec_rolloff_std": float(rolloff.std()),
        "spec_flatness_mean": float(flatness.mean()),
    }
    for i, pf in enumerate(peak_freqs):
        feats[f"peak_freq_{i + 1}"] = float(pf)
    return feats


# ── Harmonic features ────────────────────────────────────────────────────────


def harmonic_features(signal: np.ndarray, sr: int) -> dict:
    y = signal.astype(np.float32)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C1"),
        fmax=librosa.note_to_hz("C8"),
        sr=sr,
        frame_length=N_FFT,
    )
    voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    f0_mean = float(np.nanmean(f0)) if len(f0) > 0 else 0.0

    feats = {
        "f0_mean": f0_mean,
        "f0_voiced_ratio": float(voiced.size / max(len(f0), 1)),
    }
    for h, mult in enumerate([2, 3, 4], start=1):
        feats[f"harmonic_{mult}f_est"] = round(f0_mean * mult, 4)
    return feats


# ── Energy features ──────────────────────────────────────────────────────────


def energy_features(signal: np.ndarray, sr: int, psd_freqs, psd) -> dict:
    rms = float(np.sqrt(np.mean(signal**2)))
    bands = [(0, 100), (100, 500), (500, 1000), (1000, 4000)]
    feats = {"rms_energy": rms}
    for lo, hi in bands:
        mask = (psd_freqs >= lo) & (psd_freqs < hi)
        feats[f"band_energy_{lo}_{hi}Hz"] = (
            float(np.trapezoid(psd[mask], psd_freqs[mask])) if mask.any() else 0.0
        )
    # PSD peaks
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(psd, height=psd.max() * 0.01)
    top_psd = sorted(zip(psd[peaks], psd_freqs[peaks]), reverse=True)[:5]
    for i, (mag, freq) in enumerate(top_psd):
        feats[f"psd_peak_freq_{i + 1}"] = float(freq)
        feats[f"psd_peak_mag_{i + 1}"] = float(mag)
    return feats


# ── Statistical features ─────────────────────────────────────────────────────


def statistical_features(signal: np.ndarray) -> dict:
    return {
        "stat_mean": float(signal.mean()),
        "stat_variance": float(signal.var()),
        "stat_std": float(signal.std()),
        "stat_kurtosis": float(kurtosis(signal)),
        "stat_skewness": float(skew(signal)),
    }


# ── MFCC features ────────────────────────────────────────────────────────────


def mfcc_features(signal: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> dict:
    y = signal.astype(np.float32)
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LEN
    )
    feats = {}
    for i in range(n_mfcc):
        feats[f"mfcc_{i + 1}_mean"] = float(mfccs[i].mean())
        feats[f"mfcc_{i + 1}_std"] = float(mfccs[i].std())
    return feats


# ── Run all ──────────────────────────────────────────────────────────────────


def run_feature_extraction(spectral_results: dict) -> pd.DataFrame:
    rows = []
    for fpath, data in spectral_results.items():
        signal = None  # reload
        import soundfile as sf

        raw, sr = sf.read(fpath, dtype="float32", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        from src.preprocessing.preprocess import preprocess_signal

        signal = preprocess_signal(raw, sr)

        row = {
            "filename": Path(fpath).name,
            "class": data["class"],
        }
        row.update(spectral_features(signal, sr))
        row.update(harmonic_features(signal, sr))
        row.update(energy_features(signal, sr, data["psd_freqs"], data["psd"]))
        row.update(statistical_features(signal))
        row.update(mfcc_features(signal, sr))
        rows.append(row)
        print(f"[features] {row['class']}_{Path(fpath).stem}")

    df = pd.DataFrame(rows)
    out = OUTPUTS_FEATURES / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[features] Saved → {out}  shape={df.shape}")
    return df


if __name__ == "__main__":
    from src.bootstrap import get_spectral

    run_feature_extraction(get_spectral())

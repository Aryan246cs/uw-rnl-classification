"""
Phase 3 – Spectral Analysis
FFT, STFT, Spectrogram, PSD for every signal.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import welch
from scipy.signal import find_peaks
from pathlib import Path

from src.config import (
    N_FFT,
    HOP_LEN,
    N_MELS,
    WELCH_NPERSEG,
    N_TOP_PEAKS,
    OUTPUTS_FFT,
    OUTPUTS_PSD,
    OUTPUTS_SPECTROGRAMS,
)


# ── FFT ─────────────────────────────────────────────────────────────────────


def compute_fft(signal: np.ndarray, sr: int):
    spectrum = np.fft.rfft(signal * np.hanning(len(signal)))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    magnitude = np.abs(spectrum)
    return freqs, magnitude


def extract_peaks(freqs, magnitude, n: int = N_TOP_PEAKS):
    peaks, _ = find_peaks(magnitude, height=magnitude.max() * 0.01)
    order = np.argsort(magnitude[peaks])[::-1][:n]
    return freqs[peaks[order]], magnitude[peaks[order]]


def plot_fft(freqs, magnitude, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(freqs, magnitude + 1e-12, lw=0.6, color="royalblue")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (log)")
    ax.set_title(f"FFT – {label}")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}_fft.png", dpi=100)
    plt.close(fig)


# ── STFT ─────────────────────────────────────────────────────────────────────


def compute_stft(signal: np.ndarray, n_fft: int = N_FFT, hop_len: int = HOP_LEN):
    return librosa.stft(signal.astype(np.float32), n_fft=n_fft, hop_length=hop_len)


def plot_stft(stft_matrix, sr: int, hop_len: int, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max),
        sr=sr,
        hop_length=hop_len,
        x_axis="time",
        y_axis="hz",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"STFT – {label}")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}_stft.png", dpi=100)
    plt.close(fig)


# ── Spectrogram (mel) ────────────────────────────────────────────────────────


def plot_spectrogram(
    signal: np.ndarray,
    sr: int,
    label: str,
    out_dir: Path,
    n_fft: int = N_FFT,
    hop_len: int = HOP_LEN,
    n_mels: int = N_MELS,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    mel = librosa.feature.melspectrogram(
        y=signal.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_len,
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_len, x_axis="time", y_axis="mel", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel Spectrogram – {label}")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}_spectrogram.png", dpi=150)
    plt.close(fig)


# ── PSD (Welch) ──────────────────────────────────────────────────────────────


def compute_psd(signal: np.ndarray, sr: int, nperseg: int = WELCH_NPERSEG):
    freqs, psd = welch(signal, fs=sr, nperseg=nperseg)
    return freqs, psd


def plot_psd(freqs, psd, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(freqs, psd + 1e-30, lw=0.8, color="seagreen")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(f"PSD (Welch) – {label}")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}_psd.png", dpi=100)
    plt.close(fig)


# ── Run all ──────────────────────────────────────────────────────────────────


def run_spectral_analysis(preprocessed: dict):
    """
    preprocessed: {filepath: (signal, sr)}
    Returns: {filepath: {"freqs": ..., "magnitude": ..., "psd_freqs": ..., "psd": ...}}
    """
    from pathlib import Path as _Path

    results = {}
    for fpath, (signal, sr) in preprocessed.items():
        p = _Path(fpath)
        cls = p.parent.name
        stem = p.stem
        label = f"{cls}_{stem}"

        # FFT
        freqs, mag = compute_fft(signal, sr)
        peak_f, peak_m = extract_peaks(freqs, mag)
        plot_fft(freqs, mag, label, OUTPUTS_FFT / cls)

        # STFT
        stft_mat = compute_stft(signal)
        plot_stft(stft_mat, sr, HOP_LEN, label, OUTPUTS_FFT / cls)

        # Spectrogram
        plot_spectrogram(signal, sr, label, OUTPUTS_SPECTROGRAMS / cls)

        # PSD
        psd_f, psd = compute_psd(signal, sr)
        plot_psd(psd_f, psd, label, OUTPUTS_PSD / cls)

        results[fpath] = {
            "freqs": freqs,
            "magnitude": mag,
            "peak_freqs": peak_f,
            "peak_mags": peak_m,
            "psd_freqs": psd_f,
            "psd": psd,
            "sr": sr,
            "class": cls,
            "stem": stem,
        }
        print(f"[spectral] {label}")

    return results


if __name__ == "__main__":
    from src.preprocessing.loader import inventory_dataset
    from src.preprocessing.preprocess import run_preprocessing

    df = inventory_dataset()
    prep = run_preprocessing(df)
    run_spectral_analysis(prep)

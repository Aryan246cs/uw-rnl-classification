"""
Phase 2 – Preprocessing
DC offset removal, amplitude normalisation, optional bandpass filter.
Generates before/after plots.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

from src.config import (
    BANDPASS_LOW,
    BANDPASS_HIGH,
    BANDPASS_ORDER,
    OUTPUTS_WAVEFORMS,
)


def remove_dc(signal: np.ndarray) -> np.ndarray:
    return signal - signal.mean()


def normalize(signal: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal / peak


def bandpass_filter(
    signal: np.ndarray,
    sr: int,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    nyq = sr / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 1 - 1e-6)
    sos = butter(order, [low_n, high_n], btype="band", output="sos")
    return sosfilt(sos, signal)


def preprocess_signal(
    signal: np.ndarray, sr: int, apply_bandpass: bool = True
) -> np.ndarray:
    processed = remove_dc(signal)
    processed = normalize(processed)
    if apply_bandpass:
        processed = bandpass_filter(processed, sr)
        processed = normalize(processed)  # re-normalise after filter
    return processed


def plot_before_after(
    raw: np.ndarray, processed: np.ndarray, sr: int, label: str, out_dir: Path
):
    """Save a waveform comparison PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t_raw = np.linspace(0, len(raw) / sr, len(raw), endpoint=False)
    t_proc = np.linspace(0, len(processed) / sr, len(processed), endpoint=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
    axes[0].plot(t_raw, raw, lw=0.4, color="steelblue")
    axes[0].set_title(f"Raw – {label}")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(t_proc, processed, lw=0.4, color="darkorange")
    axes[1].set_title(f"Processed – {label}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}_waveform.png", dpi=100)
    plt.close(fig)


def run_preprocessing(df, apply_bandpass: bool = True):
    """
    Preprocess every file in inventory DataFrame.
    Returns dict: {filepath: (processed_signal, sr)}
    """
    results = {}
    for _, row in df.iterrows():
        import soundfile as sf_inner

        signal, sr = sf_inner.read(row["filepath"], dtype="float32", always_2d=False)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        processed = preprocess_signal(signal, sr, apply_bandpass=apply_bandpass)

        label = f"{row['class']}_{Path(row['filepath']).stem}"
        out_dir = OUTPUTS_WAVEFORMS / row["class"]
        plot_before_after(signal, processed, sr, label, out_dir)

        results[row["filepath"]] = (processed, sr)
        print(f"[preprocess] {label}  sr={sr}  samples={len(processed)}")

    return results


if __name__ == "__main__":
    from src.preprocessing.loader import inventory_dataset

    df = inventory_dataset()
    run_preprocessing(df)

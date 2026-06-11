"""
Phase 5 – Signature Discovery
Average FFT, PSD, Spectrogram per vessel class.
Visual comparison across all four classes.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.config import (
    N_FFT,
    HOP_LEN,
    N_MELS,
    WELCH_NPERSEG,
    OUTPUTS_REPORTS,
)


def _interp_to_common(arrays):
    """Interpolate all arrays to the length of the shortest one."""
    min_len = min(len(a) for a in arrays)
    return np.array(
        [
            np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(a)), a)
            for a in arrays
        ]
    )


def compute_class_averages(spectral_results: dict) -> dict:
    """
    Group spectral data by class and compute element-wise averages.
    Returns: {class: {"avg_fft": ..., "avg_psd": ..., "freqs": ..., "psd_freqs": ...}}
    """
    by_class = defaultdict(
        lambda: {"magnitudes": [], "psds": [], "freqs": None, "psd_freqs": None}
    )

    for fpath, data in spectral_results.items():
        cls = data["class"]
        by_class[cls]["magnitudes"].append(data["magnitude"])
        by_class[cls]["psds"].append(data["psd"])
        by_class[cls]["freqs"] = data["freqs"]
        by_class[cls]["psd_freqs"] = data["psd_freqs"]

    averages = {}
    for cls, d in by_class.items():
        mags_arr = _interp_to_common(d["magnitudes"])
        psd_arr = _interp_to_common(d["psds"])
        averages[cls] = {
            "avg_fft": mags_arr.mean(axis=0),
            "std_fft": mags_arr.std(axis=0),
            "avg_psd": psd_arr.mean(axis=0),
            "std_psd": psd_arr.std(axis=0),
            "freqs": d["freqs"],
            "psd_freqs": d["psd_freqs"],
        }
    return averages


def plot_class_comparison_fft(averages: dict):
    out_dir = OUTPUTS_REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Cargo": "royalblue",
        "Passengership": "darkorange",
        "Tanker": "seagreen",
        "Tug": "crimson",
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    for cls, data in averages.items():
        f = data["freqs"]
        avg = data["avg_fft"]
        std = data["std_fft"]
        # Interpolate freqs to match avg length if needed
        f_plot = np.linspace(f[0], f[-1], len(avg))
        ax.semilogy(f_plot, avg + 1e-12, label=cls, color=colors.get(cls), lw=1.2)
        ax.fill_between(
            f_plot,
            np.maximum(avg - std, 1e-12),
            avg + std,
            alpha=0.15,
            color=colors.get(cls),
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean Magnitude (log)")
    ax.set_title("Average FFT by Vessel Class")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "class_avg_fft.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[signatures] → {path}")


def plot_class_comparison_psd(averages: dict):
    out_dir = OUTPUTS_REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Cargo": "royalblue",
        "Passengership": "darkorange",
        "Tanker": "seagreen",
        "Tug": "crimson",
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    for cls, data in averages.items():
        f = data["psd_freqs"]
        avg = data["avg_psd"]
        std = data["std_psd"]
        f_plot = np.linspace(f[0], f[-1], len(avg))
        ax.semilogy(f_plot, avg + 1e-30, label=cls, color=colors.get(cls), lw=1.2)
        ax.fill_between(
            f_plot,
            np.maximum(avg - std, 1e-30),
            avg + std,
            alpha=0.15,
            color=colors.get(cls),
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title("Average PSD (Welch) by Vessel Class")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "class_avg_psd.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[signatures] → {path}")


def plot_per_class_grid(averages: dict):
    """2×2 grid with individual class PSD."""
    out_dir = OUTPUTS_REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(averages.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)
    axes = axes.flatten()
    for i, cls in enumerate(classes[:4]):
        data = averages[cls]
        f = np.linspace(
            data["psd_freqs"][0], data["psd_freqs"][-1], len(data["avg_psd"])
        )
        axes[i].semilogy(f, data["avg_psd"] + 1e-30, lw=1.0)
        axes[i].fill_between(
            f,
            np.maximum(data["avg_psd"] - data["std_psd"], 1e-30),
            data["avg_psd"] + data["std_psd"],
            alpha=0.2,
        )
        axes[i].set_title(f"{cls} – Mean PSD ± std")
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("PSD")
        axes[i].grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "class_psd_grid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[signatures] → {path}")


def run_signature_discovery(spectral_results: dict):
    averages = compute_class_averages(spectral_results)
    plot_class_comparison_fft(averages)
    plot_class_comparison_psd(averages)
    plot_per_class_grid(averages)
    return averages


if __name__ == "__main__":
    from src.preprocessing.loader import inventory_dataset
    from src.preprocessing.preprocess import run_preprocessing
    from src.spectral.spectral_analysis import run_spectral_analysis

    df_inv = inventory_dataset()
    prep = run_preprocessing(df_inv)
    spec = run_spectral_analysis(prep)
    run_signature_discovery(spec)

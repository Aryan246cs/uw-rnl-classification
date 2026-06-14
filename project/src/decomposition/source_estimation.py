"""
Phase 7 – Contribution Estimation Framework
RNL = w1*M1 + w2*M2 + w3*M3 + e
Prototypes: Linear Regression + NMF on PSD vectors.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.decomposition import NMF
from pathlib import Path

from src.config import OUTPUTS_REPORTS


def _build_psd_matrix(spectral_results: dict):
    """Stack PSD vectors (interpolated to common length) → matrix X."""
    from src.signatures.signature_discovery import _interp_to_common

    psds = [data["psd"] for data in spectral_results.values()]
    labels = [(data["class"], Path(fp).name) for fp, data in spectral_results.items()]
    X = _interp_to_common(psds)  # (n_files, freq_bins)
    return X, labels


def run_nmf(X: np.ndarray, n_components: int = 3):
    """Non-negative Matrix Factorisation: X ≈ W · H"""
    X_nn = np.maximum(X, 0)
    model = NMF(
        n_components=n_components, init="nndsvda", max_iter=500, random_state=42
    )
    W = model.fit_transform(X_nn)  # (n_files, n_components) — weights
    H = model.components_  # (n_components, freq_bins) — sources
    residual = X_nn - W @ H
    return W, H, residual, model.reconstruction_err_


def run_linear_decomp(X: np.ndarray, n_components: int = 3):
    """
    Approximate each signal as a weighted sum of n_components basis vectors
    derived from the mean spectra of randomly seeded clusters (simple proxy).
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), n_components, replace=False)
    basis = X[idx]  # (n_components, freq_bins)

    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(basis.T, X.T)  # fit each signal
    W = reg.coef_  # (n_files, n_components)
    residual = X - W @ basis
    return W, basis, residual


def plot_nmf_components(H: np.ndarray, psd_freqs, n_components: int):
    out_dir = OUTPUTS_REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_components, 1, figsize=(12, 3 * n_components), sharex=True
    )
    for i in range(n_components):
        f_plot = np.linspace(psd_freqs[0], psd_freqs[-1], len(H[i]))
        axes[i].semilogy(f_plot, H[i] + 1e-30, lw=0.8)
        axes[i].set_title(f"NMF Component {i + 1}")
        axes[i].set_ylabel("Weight")
        axes[i].grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    path = out_dir / "nmf_components.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[decomp] → {path}")


def run_source_estimation(spectral_results: dict):
    X, labels = _build_psd_matrix(spectral_results)
    n_comp = 3

    # NMF
    W_nmf, H_nmf, res_nmf, err = run_nmf(X, n_comp)
    print(f"[decomp] NMF reconstruction error: {err:.6f}")

    # Save weights
    df_weights = pd.DataFrame(W_nmf, columns=[f"w{i + 1}" for i in range(n_comp)])
    df_weights.insert(0, "filename", [l[1] for l in labels])
    df_weights.insert(1, "class", [l[0] for l in labels])
    out = OUTPUTS_REPORTS / "nmf_weights.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_weights.to_csv(out, index=False)
    print(f"[decomp] NMF weights → {out}")

    # Plot components
    sample_psd_freqs = list(spectral_results.values())[0]["psd_freqs"]
    plot_nmf_components(H_nmf, sample_psd_freqs, n_comp)

    # Residual
    residuals = {fp: res_nmf[i] for i, fp in enumerate(spectral_results.keys())}
    return residuals, W_nmf, H_nmf


if __name__ == "__main__":
    from src.bootstrap import get_spectral

    run_source_estimation(get_spectral())

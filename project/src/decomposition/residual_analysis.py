"""
Phase 8 – Residual Noise Analysis
Gaussianity tests, statistical characterisation of residual signals.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest, shapiro, kurtosis, skew, norm
from pathlib import Path

from src.config import OUTPUTS_REPORTS


def analyse_residual(residual: np.ndarray, label: str) -> dict:
    res = residual.flatten().astype(float)

    # Normalise for testing
    res_norm = (res - res.mean()) / (res.std() + 1e-12)

    # KS test vs normal
    ks_stat, ks_p = kstest(res_norm, "norm")

    # Shapiro-Wilk (limited to 5000 samples)
    sample = res_norm[:5000] if len(res_norm) > 5000 else res_norm
    sw_stat, sw_p = shapiro(sample)

    kurt_val = float(kurtosis(res))
    skew_val = float(skew(res))

    # Classification heuristic
    if ks_p > 0.05 and sw_p > 0.05:
        nature = "Gaussian"
    elif abs(kurt_val) > 10:
        nature = "Heavy-tailed / Impulsive"
    elif abs(kurt_val) > 3:
        nature = "Non-Gaussian"
    else:
        nature = "Near-Gaussian"

    return {
        "label": label,
        "ks_stat": round(ks_stat, 6),
        "ks_p": round(ks_p, 6),
        "sw_stat": round(sw_stat, 6),
        "sw_p": round(sw_p, 6),
        "kurtosis": round(kurt_val, 4),
        "skewness": round(skew_val, 4),
        "nature": nature,
    }


def plot_residual_histogram(residual: np.ndarray, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    res = residual.flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(res, bins=80, density=True, alpha=0.7, color="slateblue", label="Residual")
    # Overlay fitted normal
    mu, sigma = res.mean(), res.std()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(x, norm.pdf(x, mu, sigma), "r--", lw=1.5, label="Normal fit")
    ax.set_title(f"Residual Distribution – {label}")
    ax.set_xlabel("Value")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}_residual_hist.png", dpi=100)
    plt.close(fig)


def run_residual_analysis(residuals: dict) -> pd.DataFrame:
    rows = []
    out_dir = OUTPUTS_REPORTS / "residuals"
    for fpath, residual in residuals.items():
        label = Path(fpath).stem
        result = analyse_residual(residual, label)
        plot_residual_histogram(residual, label, out_dir)
        rows.append(result)
        print(f"[residual] {label}  nature={result['nature']}")

    df = pd.DataFrame(rows)
    out = OUTPUTS_REPORTS / "residual_analysis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[residual] Saved → {out}")
    return df


if __name__ == "__main__":
    from src.preprocessing.loader import inventory_dataset
    from src.preprocessing.preprocess import run_preprocessing
    from src.spectral.spectral_analysis import run_spectral_analysis
    from src.decomposition.source_estimation import run_source_estimation

    df_inv = inventory_dataset()
    prep = run_preprocessing(df_inv)
    spec = run_spectral_analysis(prep)
    res, _, _ = run_source_estimation(spec)
    run_residual_analysis(res)

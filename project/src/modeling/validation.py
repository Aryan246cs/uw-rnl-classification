"""
Phase 11 – Validation
MSE, KL Divergence, JS Divergence between real and synthetic noise.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.special import rel_entr
from pathlib import Path

from src.config import OUTPUTS_REPORTS


def _hist_probs(a: np.ndarray, b: np.ndarray, bins: int = 100):
    """Compute normalised histogram over the union range of a and b."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    # Smooth to avoid zeros
    pa = pa + 1e-10
    pb = pb + 1e-10
    pa /= pa.sum()
    pb /= pb.sum()
    return pa, pb


def compute_mse(real: np.ndarray, fake: np.ndarray) -> float:
    n = min(len(real), len(fake))
    return float(np.mean((real[:n] - fake[:n]) ** 2))


def compute_kl(real: np.ndarray, fake: np.ndarray) -> float:
    pa, pb = _hist_probs(real, fake)
    return float(np.sum(rel_entr(pa, pb)))


def compute_js(real: np.ndarray, fake: np.ndarray) -> float:
    pa, pb = _hist_probs(real, fake)
    m = 0.5 * (pa + pb)
    return float(0.5 * np.sum(rel_entr(pa, m)) + 0.5 * np.sum(rel_entr(pb, m)))


def plot_comparison(real: np.ndarray, fake: np.ndarray, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(real, bins=80, density=True, alpha=0.6, label="Real")
    axes[0].hist(fake, bins=80, density=True, alpha=0.6, label="Synthetic")
    axes[0].set_title(f"Distribution Comparison – {label}")
    axes[0].legend()

    # Q-Q style overlay
    q_real = np.quantile(real, np.linspace(0, 1, 200))
    q_fake = np.quantile(fake, np.linspace(0, 1, 200))
    axes[1].scatter(q_real, q_fake, s=5, alpha=0.5)
    lim = (min(q_real.min(), q_fake.min()), max(q_real.max(), q_fake.max()))
    axes[1].plot(lim, lim, "r--", lw=1)
    axes[1].set_xlabel("Real quantiles")
    axes[1].set_ylabel("Synthetic quantiles")
    axes[1].set_title("Q-Q Plot")

    plt.tight_layout()
    path = out_dir / f"{label}_validation.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[validation] → {path}")


def run_validation(residuals: dict, synthetic: np.ndarray) -> pd.DataFrame:
    """
    residuals : {fpath: residual_array}
    synthetic : array from GAN  (n_samples, segment_len)
    """
    if synthetic is None:
        print("[validation] No synthetic data — skipping GAN validation.")
        return pd.DataFrame()

    out_dir = OUTPUTS_REPORTS / "validation"
    real_pool = np.concatenate([r.flatten() for r in residuals.values()])
    real_pool = (real_pool - real_pool.mean()) / (real_pool.std() + 1e-12)
    fake_pool = synthetic.flatten()

    mse = compute_mse(real_pool, fake_pool)
    kl = compute_kl(real_pool, fake_pool)
    js = compute_js(real_pool, fake_pool)

    print(f"[validation] MSE={mse:.6f}  KL={kl:.6f}  JS={js:.6f}")
    plot_comparison(real_pool, fake_pool, "GAN_noise", out_dir)

    df = pd.DataFrame(
        [
            {"metric": "MSE", "value": mse},
            {"metric": "KL_Divergence", "value": kl},
            {"metric": "JS_Divergence", "value": js},
        ]
    )
    out = OUTPUTS_REPORTS / "validation_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[validation] Saved → {out}")
    return df

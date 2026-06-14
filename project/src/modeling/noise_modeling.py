"""
Phase 9 – Noise Modeling
Gaussian Mixture Model and (optional) Alpha-Stable fit on residuals.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from pathlib import Path

from src.config import OUTPUTS_REPORTS


def fit_gmm(samples: np.ndarray, n_components: int = 3) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", max_iter=200, random_state=42
    )
    gmm.fit(samples.reshape(-1, 1))
    return gmm


def gmm_pdf(gmm: GaussianMixture, x: np.ndarray) -> np.ndarray:
    log_prob = gmm.score_samples(x.reshape(-1, 1))
    return np.exp(log_prob)


def try_alpha_stable_fit(samples: np.ndarray):
    """
    Attempt alpha-stable fit via scipy.stats.levy_stable if available.
    Falls back to None gracefully.
    """
    try:
        from scipy.stats import levy_stable

        params = levy_stable.fit(samples)
        return params, levy_stable
    except Exception:
        return None, None


def plot_gmm_fit(
    samples: np.ndarray,
    gmm: GaussianMixture,
    label: str,
    alpha_params=None,
    alpha_dist=None,
    out_dir: Path = None,
):
    out_dir = out_dir or OUTPUTS_REPORTS / "noise_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(samples.min(), samples.max(), 500)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(samples, bins=80, density=True, alpha=0.5, color="steelblue", label="Data")
    ax.plot(x, gmm_pdf(gmm, x), "r-", lw=2, label=f"GMM ({gmm.n_components} comp.)")

    if alpha_params is not None and alpha_dist is not None:
        try:
            ax.plot(
                x, alpha_dist.pdf(x, *alpha_params), "g--", lw=1.5, label="α-Stable"
            )
        except Exception:
            pass

    ax.set_title(f"Noise Model – {label}")
    ax.set_xlabel("Value")
    ax.legend()
    plt.tight_layout()
    path = out_dir / f"{label}_noise_model.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[noise_model] → {path}")


def run_noise_modeling(residuals: dict) -> pd.DataFrame:
    rows = []
    out_dir = OUTPUTS_REPORTS / "noise_models"
    for fpath, residual in residuals.items():
        label = Path(fpath).stem
        samples = residual.flatten().astype(float)

        # Normalise
        samples = (samples - samples.mean()) / (samples.std() + 1e-12)

        for n_comp in [2, 3, 5]:
            gmm = fit_gmm(samples, n_components=n_comp)
            rows.append(
                {
                    "label": label,
                    "n_components": n_comp,
                    "bic": round(gmm.bic(samples.reshape(-1, 1)), 4),
                    "aic": round(gmm.aic(samples.reshape(-1, 1)), 4),
                    "log_likelihood": round(gmm.score(samples.reshape(-1, 1)), 6),
                }
            )

        # Best GMM (lowest BIC)
        best_n = min(
            [2, 3, 5], key=lambda n: fit_gmm(samples, n).bic(samples.reshape(-1, 1))
        )
        best_gmm = fit_gmm(samples, best_n)

        # Alpha-stable
        alpha_params, alpha_dist = try_alpha_stable_fit(samples[:200])
        plot_gmm_fit(samples, best_gmm, label, alpha_params, alpha_dist, out_dir)

        print(f"[noise_model] {label}  best_n_comp={best_n}")

    df = pd.DataFrame(rows)
    out = OUTPUTS_REPORTS / "noise_model_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[noise_model] Saved → {out}")
    return df


if __name__ == "__main__":
    from src.bootstrap import get_spectral
    from src.decomposition.source_estimation import run_source_estimation

    res, _, _ = run_source_estimation(get_spectral())
    run_noise_modeling(res)

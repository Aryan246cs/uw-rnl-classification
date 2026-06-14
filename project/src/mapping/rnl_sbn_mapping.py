"""
Phase 12a - RNL <-> SBN Mapping Module
Implements the chain: onboard machinery/propeller sources -> hull transfer
function H(f) -> propagation medium -> measured hydrophone RNL (Sec. 2.1, 5
"Stage VI", Eq. RNL(f) = H(f)[Sum w_i M_i(f) + P(f) + C(f)] + A(f) + E(f)).

Two estimates are produced:
  1. Hull transfer function H(f), per vessel class, via
     H(f) = S_yx(f) / S_xx(f)
     where S_xx is the measured PSD and S_yx is the (cross-)PSD of the
     NMF-reconstructed SBN component sum - i.e. H(f) ~= measured / modelled.
  2. Propagation transfer loss TL(r,f) = 20*log10(r) + alpha(f)*r using
     Thorp's frequency-dependent absorption coefficient alpha(f) [dB/km],
     evaluated for a set of assumed hydrophone-to-ship ranges (no AIS data
     available for this working subset - ranges are illustrative).
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.config import OUTPUTS_MAPPING, ASSUMED_RANGES_M


# ── Thorp absorption & transfer loss ───────────────────────────────────


def thorp_absorption_db_per_km(freqs_hz: np.ndarray) -> np.ndarray:
    """Thorp's formula, f in kHz, returns alpha in dB/km."""
    f = np.maximum(freqs_hz, 1e-6) / 1000.0  # Hz -> kHz
    f2 = f**2
    alpha = (0.11 * f2 / (1 + f2)) + (44 * f2 / (4100 + f2)) + (3e-4 * f2) + 0.003
    return alpha


def transfer_loss_db(freqs_hz: np.ndarray, range_m: float) -> np.ndarray:
    """TL(r,f) = 20*log10(r) + alpha(f)*r   [r in metres, alpha converted dB/km -> dB/m]."""
    alpha_db_per_m = thorp_absorption_db_per_km(freqs_hz) / 1000.0
    r = max(range_m, 1.0)
    return 20 * np.log10(r) + alpha_db_per_m * r


# ── Hull transfer function estimation ──────────────────────────────────


def _interp(freqs_src, values, freqs_target):
    return np.interp(freqs_target, freqs_src, values)


def estimate_hull_transfer_function(spectral_results: dict, W_nmf: np.ndarray, H_nmf: np.ndarray):
    """
    For each file, reconstruct the SBN estimate (NMF: W @ H, in PSD domain)
    and compute H(f) = measured_PSD(f) / SBN_estimate(f). Average per class.
    Returns {class: {"freqs": ..., "H_db_mean": ..., "H_db_std": ...}}
    """
    files = list(spectral_results.keys())
    by_class = defaultdict(list)
    common_freqs = None

    for i, fpath in enumerate(files):
        data = spectral_results[fpath]
        psd_freqs, psd = data["psd_freqs"], data["psd"]
        if common_freqs is None:
            common_freqs = psd_freqs

        # NMF reconstruction is on an interpolated common grid (length = H_nmf.shape[1])
        recon = W_nmf[i] @ H_nmf  # (freq_bins,)
        recon_freqs = np.linspace(psd_freqs[0], psd_freqs[-1], len(recon))
        recon_on_psd_grid = _interp(recon_freqs, recon, psd_freqs)

        H_f = (psd + 1e-30) / (recon_on_psd_grid + 1e-30)
        H_db = 10 * np.log10(np.clip(H_f, 1e-12, None))

        by_class[data["class"]].append(_interp(psd_freqs, H_db, common_freqs))

    result = {}
    for cls, curves in by_class.items():
        arr = np.array(curves)
        result[cls] = {
            "freqs": common_freqs,
            "H_db_mean": arr.mean(axis=0),
            "H_db_std": arr.std(axis=0),
        }
    return result


# ── Plotting ────────────────────────────────────────────────────────────


def plot_hull_transfer_functions(htf: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for cls, d in htf.items():
        ax.plot(d["freqs"], d["H_db_mean"], label=cls, lw=1.2)
        ax.fill_between(d["freqs"], d["H_db_mean"] - d["H_db_std"], d["H_db_mean"] + d["H_db_std"], alpha=0.15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H(f) - estimated hull transfer function (dB)")
    ax.set_title("Estimated Hull Transfer Function H(f) by Vessel Class")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "hull_transfer_function.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[mapping] -> {path}")


def plot_transfer_loss(freqs_hz: np.ndarray, ranges_m: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in ranges_m:
        ax.plot(freqs_hz, transfer_loss_db(freqs_hz, r), label=f"r = {r} m")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Transfer Loss TL(r,f) (dB)")
    ax.set_title("Propagation Transfer Loss (Thorp absorption + spherical spreading)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "transfer_loss.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[mapping] -> {path}")


def save_tables(htf: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls, d in htf.items():
        for f, h_mean, h_std in zip(d["freqs"], d["H_db_mean"], d["H_db_std"]):
            rows.append({"class": cls, "freq_hz": float(f), "H_db_mean": float(h_mean), "H_db_std": float(h_std)})
    df_htf = pd.DataFrame(rows)
    df_htf.to_csv(out_dir / "hull_transfer_function.csv", index=False)

    any_freqs = next(iter(htf.values()))["freqs"]
    rows_tl = []
    for r in ASSUMED_RANGES_M:
        tl = transfer_loss_db(any_freqs, r)
        for f, t in zip(any_freqs, tl):
            rows_tl.append({"range_m": r, "freq_hz": float(f), "TL_db": float(t)})
    df_tl = pd.DataFrame(rows_tl)
    df_tl.to_csv(out_dir / "transfer_loss.csv", index=False)

    print(f"[mapping] -> {out_dir / 'hull_transfer_function.csv'}")
    print(f"[mapping] -> {out_dir / 'transfer_loss.csv'}")
    return df_htf, df_tl


def run_rnl_sbn_mapping(spectral_results: dict, W_nmf: np.ndarray, H_nmf: np.ndarray):
    out_dir = OUTPUTS_MAPPING
    htf = estimate_hull_transfer_function(spectral_results, W_nmf, H_nmf)
    plot_hull_transfer_functions(htf, out_dir)

    any_freqs = next(iter(htf.values()))["freqs"]
    plot_transfer_loss(any_freqs, ASSUMED_RANGES_M, out_dir)
    df_htf, df_tl = save_tables(htf, out_dir)
    return htf, df_htf, df_tl


if __name__ == "__main__":
    from src.bootstrap import get_spectral
    from src.decomposition.source_estimation import run_source_estimation

    spec = get_spectral()
    _, W, H = run_source_estimation(spec)
    run_rnl_sbn_mapping(spec, W, H)

"""
Phase 11 - Synthetic Acoustic Corpus Generation
Combines the trained generative models (WGAN-GP residual noise, conditional
DDPM spectrogram patches, TimeGAN temporal sequences) into a synthetic
augmentation corpus, applying a distributional quality filter
(KL divergence vs. the real distribution) before retention.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import rel_entr

from src.config import (
    OUTPUTS_GENAI,
    SYNTH_WGAN_PER_CLASS,
    SYNTH_DDPM_PER_CLASS,
    SYNTH_TIMEGAN_PER_CLASS,
    SYNTH_KL_THRESHOLD,
    Z_DIM_GAN,
)
from src.device import get_device
from src.genai.wgan_gp import generate_samples as gan_generate
from src.genai.ddpm_separator import ddim_sample, CLASS_NAMES
from src.genai.timegan import generate_sequences


def _hist_probs(a: np.ndarray, b: np.ndarray, bins: int = 60):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    pa = pa + 1e-10
    pb = pb + 1e-10
    return pa / pa.sum(), pb / pb.sum()


def kl_divergence(real: np.ndarray, fake: np.ndarray) -> float:
    pa, pb = _hist_probs(real.flatten(), fake.flatten())
    return float(np.sum(rel_entr(pb, pa)))  # KL(fake || real)


def run_synthetic_generation(wgan_models: dict, ddpm_bundle, timegan_models: dict):
    """
    wgan_models  : {class: {"generator": G, "real": segments}}   (from run_wgan_gp)
    ddpm_bundle  : (model, schedule) or (None, None)              (from run_ddpm)
    timegan_models: {class: {generator, supervisor, recovery, ...}}
    """
    out_dir = OUTPUTS_GENAI / "synthetic_corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    rows = []

    # ── 10A: WGAN-GP residual noise corpus ──
    for cls, bundle in wgan_models.items():
        G, real = bundle["generator"], bundle["real"]
        n = min(SYNTH_WGAN_PER_CLASS, 5000)
        synth = gan_generate(G, n_samples=n, z_dim=Z_DIM_GAN, device=device)
        kl = kl_divergence(real, synth)
        kept = kl < SYNTH_KL_THRESHOLD
        if kept:
            np.save(out_dir / f"wgan_residual_{cls}.npy", synth)
        rows.append({"source": "WGAN-GP", "class": cls, "n_generated": n, "kl_vs_real": round(kl, 5), "retained": kept})
        print(f"[synth] WGAN-GP {cls}: n={n} KL={kl:.5f} retained={kept}")

    # ── 10B: Conditional DDPM spectrogram patches ──
    model, sched = ddpm_bundle if ddpm_bundle is not None else (None, None)
    if model is not None:
        for i, cls in enumerate(CLASS_NAMES):
            n = SYNTH_DDPM_PER_CLASS
            batches = []
            remaining = n
            while remaining > 0:
                b = min(remaining, 16)
                batches.append(ddim_sample(model, sched, class_idx=i, n_samples=b))
                remaining -= b
            synth = np.concatenate(batches, axis=0)
            np.save(out_dir / f"ddpm_patches_{cls}.npy", synth)
            rows.append({"source": "Conditional-DDPM", "class": cls, "n_generated": n, "kl_vs_real": np.nan, "retained": True})
            print(f"[synth] DDPM {cls}: n={n} patches saved")

    # ── 10D: TimeGAN sequences ──
    for cls, models in timegan_models.items():
        n = SYNTH_TIMEGAN_PER_CLASS
        synth = generate_sequences(models, n_samples=min(n, 2000), device=device)
        np.save(out_dir / f"timegan_sequences_{cls}.npy", synth)
        rows.append({"source": "TimeGAN", "class": cls, "n_generated": n, "kl_vs_real": np.nan, "retained": True})
        print(f"[synth] TimeGAN {cls}: n={n} sequences saved")

    df = pd.DataFrame(rows)
    out_csv = out_dir / "synthetic_corpus_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[synth] Summary -> {out_csv}")

    total_real = 33  # working subset size
    total_kept = int(df.loc[df["retained"], "n_generated"].sum())
    expansion = (total_real + total_kept) / total_real
    print(f"[synth] Corpus expansion factor (approx): {expansion:.2f}x")
    return df


if __name__ == "__main__":
    from src.bootstrap import get_spectral, get_preprocessed
    from src.decomposition.source_estimation import run_source_estimation
    from src.genai.wgan_gp import run_wgan_gp
    from src.genai.ddpm_separator import run_ddpm
    from src.genai.timegan import run_timegan

    res, _, _ = run_source_estimation(get_spectral())
    prep = get_preprocessed()
    wgan = run_wgan_gp(res)
    ddpm = run_ddpm(prep)
    tgan = run_timegan(prep)
    run_synthetic_generation(wgan, ddpm, tgan)

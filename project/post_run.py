"""
Post-run orchestration.

Reuses the cached Phase 1-3 results (preprocessed signals + spectral
analysis) produced by ``run_pipeline.py``, then:

  1. Re-runs NMF source estimation (Phase 7) -> residuals, W, H (fast).
  2. Re-trains the WGAN-GP residual-noise generators (Phase 10A).
  3. Builds the baseline + WGAN-GP-augmented classification datasets
     (Phase 12b) and runs the classifier suite, now with per-class
     precision/recall/F1 reporting.
  4. Runs the WGAN-GP gradient-penalty and Beta-VAE beta-sensitivity
     ablation studies.
  5. Rebuilds outputs/reports/final_report.md covering all 12 phases
     plus the ablation studies, with full provenance.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from src.bootstrap import get_preprocessed, get_spectral
from src.decomposition.machinery_decomposition import run_machinery_decomposition
from src.signatures.machinery_signatures import run_machinery_analysis
from src.decomposition.residual_analysis import run_residual_analysis
from src.modeling.noise_modeling import run_noise_modeling
from src.genai.wgan_gp import run_wgan_gp
from src.genai.vae_latent import run_vae
from src.classification.dataset import build_dataset
from src.classification.train import run_classification
from src.ablation.run_ablations import run_ablations
from src.reporting.generate_report import build_report


def main():
    print("[post_run] Loading cached preprocessed + spectral data ...")
    prep = get_preprocessed()
    spectral = get_spectral()

    print("[post_run] Phase 6 - machinery signature exploration (with category labels) ...")
    run_machinery_analysis(spectral)

    print("[post_run] Phase 7b - machinery source decomposition (named components) ...")
    time_residuals, decomp_df, decomp_summary = run_machinery_decomposition(prep)

    # Down-sample full-length time-domain residuals for KS/Shapiro/GMM (Phases 8-9);
    # Phase 10A (WGAN-GP) uses the full-length residuals for far more segments.
    stat_residuals = {k: v[::100] for k, v in time_residuals.items()}

    print("[post_run] Phase 8 - residual noise analysis ...")
    run_residual_analysis(stat_residuals)

    print("[post_run] Phase 9 - residual noise modeling (GMM baseline) ...")
    run_noise_modeling(stat_residuals)

    print("[post_run] Phase 10A - training WGAN-GP residual generators (time-domain residuals) ...")
    wgan_models = run_wgan_gp(time_residuals)

    print("[post_run] Phase 10C - training Beta-VAE on time-domain residuals ...")
    run_vae(time_residuals)

    print("[post_run] Phase 12b - building classification datasets ...")
    baseline = build_dataset(prep, augment=False)
    augmented = build_dataset(prep, wgan_models=wgan_models, augment=True)

    print("[post_run] Phase 12b - running classifier suite (baseline vs augmented) ...")
    run_classification(baseline, augmented)

    print("[post_run] Running ablation studies (WGAN-GP grad-penalty, Beta-VAE beta) ...")
    run_ablations(time_residuals)

    print("[post_run] Rebuilding final_report.md ...")
    build_report()

    print("[post_run] Done.")


if __name__ == "__main__":
    main()

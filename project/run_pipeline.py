"""
DeepShip Research Pipeline – Master Runner
Executes all 12 phases (Phases 1-9 classical/statistical pipeline,
Phases 10A-12b GenAI modelling, RNL-SBN mapping and classification) in order.
"""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.loader import inventory_dataset
from src.preprocessing.preprocess import run_preprocessing
from src.spectral.spectral_analysis import run_spectral_analysis
from src.features.feature_extraction import run_feature_extraction
from src.signatures.signature_discovery import run_signature_discovery
from src.signatures.machinery_signatures import run_machinery_analysis
from src.decomposition.source_estimation import run_source_estimation
from src.decomposition.machinery_decomposition import run_machinery_decomposition
from src.decomposition.residual_analysis import run_residual_analysis
from src.modeling.noise_modeling import run_noise_modeling
from src.genai.wgan_gp import run_wgan_gp
from src.genai.vae_latent import run_vae
from src.classification.component_pipeline import run_component_classification
from src.genai.timegan import run_timegan
from src.genai.ddpm_separator import run_ddpm
from src.genai.synthetic_generator import run_synthetic_generation
from src.mapping.rnl_sbn_mapping import run_rnl_sbn_mapping
from src.reporting.generate_report import build_report


def main():
    print("=" * 60)
    print("DeepShip Analysis Pipeline")
    print("=" * 60)

    # Phase 1 – Load
    print("\n[Phase 1] Signal Inventory")
    df_inv = inventory_dataset()

    # Phase 2 – Preprocess
    print("\n[Phase 2] Preprocessing")
    from src.cache import (
        load_preprocessed,
        save_preprocessed,
        load_spectral,
        save_spectral,
    )

    preprocessed = load_preprocessed()
    if preprocessed is None:
        preprocessed = run_preprocessing(df_inv, apply_bandpass=True)
        save_preprocessed(preprocessed)

    # Phase 3 – Spectral
    print("\n[Phase 3] Spectral Analysis (FFT / STFT / Spectrogram / PSD)")
    spectral = load_spectral()
    if spectral is None:
        spectral = run_spectral_analysis(preprocessed)
        save_spectral(spectral)

    # Phase 4 – Features
    print("\n[Phase 4] Feature Extraction")
    features_df = run_feature_extraction(spectral)

    # Phase 5 – Signatures
    print("\n[Phase 5] Signature Discovery")
    averages = run_signature_discovery(spectral)

    # Phase 6 – Machinery
    print("\n[Phase 6] Machinery Signature Exploration")
    mach_df = run_machinery_analysis(spectral)

    # Phase 7 – Decomposition (global PSD-domain NMF, feeds Phase 12a SBN estimate)
    print("\n[Phase 7] Source Contribution Estimation (NMF)")
    _, W, H = run_source_estimation(spectral)

    # Phase 7b – Named machinery source decomposition (per-file STFT NMF +
    # Wiener masking + Appendix B labelling). Produces the time-domain
    # residual eps[n] = x[n] - x_hat[n] that feeds Phases 8-10A below.
    print("\n[Phase 7b] Machinery Source Decomposition (named components)")
    time_residuals, decomp_df, decomp_summary = run_machinery_decomposition(preprocessed)

    # Down-sample the full-length time-domain residuals for the statistical
    # tests in Phases 8/9 (KS/Shapiro/GMM are intractable at ~6M samples/file);
    # Phase 10A (WGAN-GP) uses the full-length residuals for more segments.
    stat_residuals = {k: v[::100] for k, v in time_residuals.items()}

    # Phase 8 – Residual analysis
    print("\n[Phase 8] Residual Noise Analysis")
    res_df = run_residual_analysis(stat_residuals)

    # Phase 9 – Noise modeling (GMM baseline)
    print("\n[Phase 9] Residual Noise Modeling (GMM baseline)")
    noise_df = run_noise_modeling(stat_residuals)

    # Phase 10A – WGAN-GP residual noise model
    print("\n[Phase 10A] WGAN-GP Residual Noise Modelling")
    wgan_models = run_wgan_gp(time_residuals)

    # Phase 10C – Beta-VAE latent source representation
    print("\n[Phase 10C] Beta-VAE Latent Source Representation")
    run_vae(time_residuals)

    # Phase 7c – Machinery-component classification (GAN-augmented)
    print("\n[Phase 7c] Machinery-Component Classification (GAN-Augmented)")
    run_component_classification(preprocessed)

    # Phase 10D – TimeGAN temporal sequence synthesis
    print("\n[Phase 10D] TimeGAN Temporal Sequence Synthesis")
    timegan_models = run_timegan(preprocessed)

    # Phase 10B – Conditional DDPM spectrogram patch synthesis
    print("\n[Phase 10B] Conditional DDPM Spectrogram Patch Synthesis")
    ddpm_bundle = run_ddpm(preprocessed)

    # Phase 11 – Synthetic acoustic corpus generation
    print("\n[Phase 11] Synthetic Acoustic Corpus Generation")
    run_synthetic_generation(wgan_models, ddpm_bundle, timegan_models)

    # Phase 12a – RNL-SBN mapping (hull transfer function + Thorp propagation)
    print("\n[Phase 12a] RNL-SBN Mapping")
    run_rnl_sbn_mapping(spectral, W, H)

    # Phase 12b (Vessel Classification) de-emphasized — pipeline focus is
    # source decomposition (Phase 7b) + residual noise modeling (Phases 8-10C).

    # Final report
    print("\n[Report] Building consolidated research report")
    build_report()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

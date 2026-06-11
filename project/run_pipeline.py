"""
DeepShip Research Pipeline – Master Runner
Executes all 11 phases in order.
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
from src.decomposition.residual_analysis import run_residual_analysis
from src.modeling.noise_modeling import run_noise_modeling
from src.modeling.gan_model import run_gan_pipeline
from src.modeling.validation import run_validation


def main():
    print("=" * 60)
    print("DeepShip Analysis Pipeline")
    print("=" * 60)

    # Phase 1 – Load
    print("\n[Phase 1] Signal Inventory")
    df_inv = inventory_dataset()

    # Phase 2 – Preprocess
    print("\n[Phase 2] Preprocessing")
    preprocessed = run_preprocessing(df_inv, apply_bandpass=True)

    # Phase 3 – Spectral
    print("\n[Phase 3] Spectral Analysis (FFT / STFT / Spectrogram / PSD)")
    spectral = run_spectral_analysis(preprocessed)

    # Phase 4 – Features
    print("\n[Phase 4] Feature Extraction")
    features_df = run_feature_extraction(spectral)

    # Phase 5 – Signatures
    print("\n[Phase 5] Signature Discovery")
    averages = run_signature_discovery(spectral)

    # Phase 6 – Machinery
    print("\n[Phase 6] Machinery Signature Exploration")
    mach_df = run_machinery_analysis(spectral)

    # Phase 7 – Decomposition
    print("\n[Phase 7] Source Contribution Estimation (NMF)")
    residuals, W, H = run_source_estimation(spectral)

    # Phase 8 – Residual analysis
    print("\n[Phase 8] Residual Noise Analysis")
    res_df = run_residual_analysis(residuals)

    # Phase 9 – Noise modeling
    print("\n[Phase 9] Noise Modeling (GMM)")
    noise_df = run_noise_modeling(residuals)

    # Phase 10 – GAN
    print("\n[Phase 10] GAN Training")
    G, synthetic = run_gan_pipeline(residuals)

    # Phase 11 – Validation
    print("\n[Phase 11] Validation")
    val_df = run_validation(residuals, synthetic)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

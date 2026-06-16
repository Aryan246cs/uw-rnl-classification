"""
Improved Single-Ship-Type Pipeline for Enhanced Component Classification

Goal: Achieve 80-85%+ accuracy by:
1. Training on single vessel type (Cargo or Passengership)
2. Infusing 40-50% residual noise from previous results into new decomposition
3. Enhanced weight optimization via residual noise modeling
4. Complete GAN augmentation pipeline

Usage:
    python project/run_improved_pipeline.py --ship_type Cargo
    python project/run_improved_pipeline.py --ship_type Passengership
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src_improved.preprocessing_single import run_preprocessing_single
from src_improved.spectral_single import run_spectral_single
from src_improved.decomposition_improved import run_improved_decomposition
from src_improved.residual_modeling_improved import run_improved_residual_modeling
from src_improved.weight_optimization import optimize_machinery_weights
from src_improved.gan_augmentation_improved import run_improved_gan_pipeline
from src_improved.classification_improved import run_improved_classification
from src_improved.report_generator import generate_improvement_report


def main():
    parser = argparse.ArgumentParser(description="Run improved single-ship pipeline")
    parser.add_argument(
        "--ship_type",
        type=str,
        required=True,
        choices=["Cargo", "Passengership"],
        help="Ship type to train on (Cargo or Passengership recommended)"
    )
    parser.add_argument(
        "--residual_infusion",
        type=float,
        default=0.45,
        help="Fraction of previous residual noise to infuse (default: 0.45)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"IMPROVED PIPELINE - Single Ship Type: {args.ship_type}")
    print(f"Residual Infusion: {args.residual_infusion * 100:.0f}%")
    print("=" * 70)

    # Phase 1: Load and preprocess single ship type
    print("\n[Phase 1] Loading and preprocessing single ship type...")
    preprocessed, sr = run_preprocessing_single(args.ship_type)
    print(f"  Loaded {len(preprocessed)} files for {args.ship_type}")

    # Phase 2: Spectral analysis (computed for completeness; not used downstream)
    print("\n[Phase 2] Skipping spectral analysis (unused downstream — saves time).")

    # Phase 3: Improved decomposition with residual infusion
    print("\n[Phase 3] Improved machinery decomposition with residual infusion...")
    components, residuals, decomp_df = run_improved_decomposition(
        preprocessed, sr, args.ship_type, args.residual_infusion
    )

    # Phase 4: Enhanced residual modeling with GAN
    print("\n[Phase 4] Enhanced residual noise modeling...")
    residual_model, residual_characteristics = run_improved_residual_modeling(
        residuals, args.ship_type
    )

    # Phase 5: Weight optimization using residual modeling
    print("\n[Phase 5] Optimizing machinery weights via residual back-calculation...")
    optimized_weights = optimize_machinery_weights(
        preprocessed, components, residuals, residual_model, args.ship_type
    )

    # Phase 6: Improved GAN augmentation
    print("\n[Phase 6] Training improved GANs for all components...")
    gan_generators, synthetic_data = run_improved_gan_pipeline(
        components, residuals, args.ship_type
    )

    # Phase 7: Enhanced classification
    print("\n[Phase 7] Training improved classifier with all enhancements...")
    accuracy, f1_scores, confusion = run_improved_classification(
        components, synthetic_data, optimized_weights, args.ship_type
    )

    # Phase 8: Generate comprehensive report
    print("\n[Phase 8] Generating improvement report...")
    generate_improvement_report(
        args.ship_type,
        decomp_df,
        optimized_weights,
        accuracy,
        f1_scores,
        confusion,
        residual_characteristics
    )

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE - {args.ship_type}")
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"Previous Baseline: 70.7%")
    print(f"Improvement: {(accuracy - 0.707) * 100:+.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

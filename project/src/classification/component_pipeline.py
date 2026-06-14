"""
Phase 7c - Machinery-Component Classification with GAN Augmentation

Workflow:
  1. Build a labeled segment dataset from Phase 7b's per-file decomposition
     (each segment labeled by its Appendix-B machinery category - Engine
     Shaft & Propeller BPF, Generator, Pumps & Compressors, Gearbox /
     Gear-mesh, Hull & High-Frequency Machinery, or the residual
     Cavitation/Flow/Ambient Noise category).
  2. Train a baseline 1D-CNN classifier (Phase 7c) on real segments only.
  3. Train a WGAN-GP generator per category (same architecture as Phase 10A)
     on that category's training segments, until the generator produces
     segments the critic can no longer distinguish from real ones.
  4. Augment the training set with GAN-generated segments per category and
     retrain the classifier from scratch on the same train/val split.
  5. Compare baseline vs. GAN-augmented per-category classification metrics.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import OUTPUTS_COMPONENT_CLF, COMPONENT_MIN_SEGMENTS_FOR_GAN, COMPONENT_SYNTH_PER_CATEGORY
from src.decomposition.component_dataset import build_component_segments
from src.classification.component_model import train_classifier
from src.genai.wgan_gp import train_wgan_gp, generate_samples


def run_component_classification(preprocessed: dict):
    print("[component_clf] Building per-component segment dataset (Phase 7b labels) ...")
    by_category = build_component_segments(preprocessed)

    categories = sorted(by_category.keys())
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    X, y = [], []
    for cat, segs in by_category.items():
        X.append(segs)
        y.append(np.full(len(segs), cat_to_idx[cat]))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    print(f"[component_clf] {len(categories)} categories, {len(X)} total segments:")
    for cat, idx in cat_to_idx.items():
        print(f"  [{idx}] {cat}: {(y == idx).sum()} segments")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[component_clf] Training baseline classifier (real segments only) ...")
    _, _, baseline_report = train_classifier(
        X_train, y_train, X_val, y_val, n_classes=len(categories), label="baseline"
    )

    print("[component_clf] Training per-category WGAN-GP generators on training segments ...")
    synth_X, synth_y = [], []
    for cat, idx in cat_to_idx.items():
        train_segs = X_train[y_train == idx]
        if len(train_segs) < COMPONENT_MIN_SEGMENTS_FOR_GAN:
            print(f"[component_clf] {cat}: too few training segments "
                  f"({len(train_segs)}) - skipping GAN augmentation")
            continue
        short = cat.split(" (")[0].replace(" & ", "_").replace(" / ", "_").replace(" ", "_")
        G, _, _ = train_wgan_gp(train_segs, label=f"component_{short}")
        synth = generate_samples(G, COMPONENT_SYNTH_PER_CATEGORY)
        synth_X.append(synth)
        synth_y.append(np.full(len(synth), idx))
        print(f"[component_clf] {cat}: generated {len(synth)} synthetic segments")

    if synth_X:
        X_aug = np.concatenate([X_train] + synth_X, axis=0)
        y_aug = np.concatenate([y_train] + synth_y, axis=0)
    else:
        X_aug, y_aug = X_train, y_train

    print("[component_clf] Training GAN-augmented classifier ...")
    _, _, augmented_report = train_classifier(
        X_aug, y_aug, X_val, y_val, n_classes=len(categories), label="augmented"
    )

    rows = []
    for cat, idx in cat_to_idx.items():
        key = str(idx)
        base = baseline_report.get(key, {})
        aug = augmented_report.get(key, {})
        rows.append({
            "category": cat,
            "n_segments": int((y == idx).sum()),
            "n_train_segments": int((y_train == idx).sum()),
            "n_val_segments": int((y_val == idx).sum()),
            "n_synthetic_segments": int((y_aug == idx).sum() - (y_train == idx).sum()),
            "baseline_precision": round(base.get("precision", 0.0), 4),
            "baseline_recall": round(base.get("recall", 0.0), 4),
            "baseline_f1": round(base.get("f1-score", 0.0), 4),
            "augmented_precision": round(aug.get("precision", 0.0), 4),
            "augmented_recall": round(aug.get("recall", 0.0), 4),
            "augmented_f1": round(aug.get("f1-score", 0.0), 4),
        })

    df = pd.DataFrame(rows)
    df["baseline_overall_accuracy"] = round(baseline_report["accuracy"], 4)
    df["augmented_overall_accuracy"] = round(augmented_report["accuracy"], 4)

    OUTPUTS_COMPONENT_CLF.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUTS_COMPONENT_CLF / "component_classification.csv"
    df.to_csv(out_csv, index=False)
    print(f"[component_clf] -> {out_csv}")
    print(f"[component_clf] baseline accuracy={baseline_report['accuracy']:.4f}  "
          f"augmented accuracy={augmented_report['accuracy']:.4f}")

    _plot_f1_comparison(df, OUTPUTS_COMPONENT_CLF)

    return df


def _plot_f1_comparison(df: pd.DataFrame, out_dir):
    labels = [c.split(" (")[0] for c in df["category"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, df["baseline_f1"], width, label="Baseline (real only)")
    ax.bar(x + width / 2, df["augmented_f1"], width, label="GAN-augmented")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("Phase 7c: Machinery-Component Classifier — Baseline vs. GAN-Augmented")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "component_classification_f1.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[component_clf] -> {path}")


if __name__ == "__main__":
    from src.bootstrap import get_preprocessed

    run_component_classification(get_preprocessed())

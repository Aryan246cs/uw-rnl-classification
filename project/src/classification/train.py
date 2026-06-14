"""
Phase 12b - Vessel Classification, Baseline vs. Augmented
Trains CNN, ResNet-lite, CRNN and Transformer classifiers (Sec. 5) using
GroupKFold cross-validation (grouped by source WAV file to prevent
segment-level leakage), once on the real-only ("baseline") dataset and once
on the WGAN-GP-augmented dataset (Phase 11). Reports Top-1 accuracy, macro
F1, ROC-AUC (OvR), confusion matrices, an ensemble (confidence-weighted
voting) and the delta-accuracy attributable to GenAI augmentation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

from src.config import (
    OUTPUTS_CLASSIFICATION,
    CLASSIFIER_BATCH,
    CLASSIFIER_LR,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_FOLDS,
)
from src.device import get_device
from src.classification.models import MODEL_REGISTRY, N_CLASSES
from src.classification.dataset import CLASS_NAMES


def _train_one_fold(model_cls, X_train, y_train, X_val, y_val, device, input_kind):
    model = model_cls().to(device)
    opt = optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    n = len(Xt)
    batch_size = min(CLASSIFIER_BATCH, max(4, n))

    for epoch in range(CLASSIFIER_EPOCHS):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            b_idx = idx[start : start + batch_size]
            if len(b_idx) < 2:
                continue
            xb, yb = Xt[b_idx], yt[b_idx]
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xv)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    model.train()
    return probs


def evaluate_model(model_name: str, data: dict, device, n_splits: int = CLASSIFIER_FOLDS):
    model_cls, input_kind = MODEL_REGISTRY[model_name]
    X = data[input_kind]
    y = data["y"]
    groups = data["groups"]

    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    all_y_true, all_y_pred, all_probs = [], [], []
    for train_idx, val_idx in gkf.split(X, y, groups):
        probs = _train_one_fold(model_cls, X[train_idx], y[train_idx], X[val_idx], y[val_idx], device, input_kind)
        all_probs.append(probs)
        all_y_true.append(y[val_idx])
        all_y_pred.append(probs.argmax(axis=1))

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    probs = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        roc_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    return {"accuracy": acc, "macro_f1": f1, "roc_auc": roc_auc, "confusion_matrix": cm,
            "y_true": y_true, "y_pred": y_pred, "probs": probs}


def plot_confusion_matrix(cm: np.ndarray, title: str, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_classification(baseline_data: dict, augmented_data: dict):
    """
    baseline_data / augmented_data: dicts from src.classification.dataset.build_dataset
    Returns a results DataFrame with one row per (model, dataset).
    """
    device = get_device()
    out_dir = OUTPUTS_CLASSIFICATION
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_class_rows = []
    ensemble_probs = {"baseline": [], "augmented": []}
    y_true_ref = {"baseline": None, "augmented": None}

    for dataset_name, data in [("baseline", baseline_data), ("augmented", augmented_data)]:
        for model_name in MODEL_REGISTRY:
            print(f"[classify] {model_name} on {dataset_name} dataset "
                  f"({len(data['y'])} segments, {len(np.unique(data['groups']))} files)")
            result = evaluate_model(model_name, data, device)
            rows.append({
                "model": model_name, "dataset": dataset_name,
                "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "roc_auc": result["roc_auc"],
            })
            plot_confusion_matrix(result["confusion_matrix"], f"{model_name} ({dataset_name})",
                                   out_dir / f"confusion_{model_name.replace(' ', '_')}_{dataset_name}.png")
            ensemble_probs[dataset_name].append(result["probs"])
            y_true_ref[dataset_name] = result["y_true"]

            precision, recall, f1c, support = precision_recall_fscore_support(
                result["y_true"], result["y_pred"], labels=list(range(N_CLASSES)), zero_division=0
            )
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                per_class_rows.append({
                    "model": model_name, "dataset": dataset_name, "class": cls_name,
                    "precision": precision[cls_idx], "recall": recall[cls_idx],
                    "f1": f1c[cls_idx], "support": int(support[cls_idx]),
                })

    # Confidence-weighted ensemble (simple average of softmax probabilities)
    for dataset_name in ["baseline", "augmented"]:
        probs_list = ensemble_probs[dataset_name]
        min_len = min(p.shape[0] for p in probs_list)
        avg_probs = np.mean([p[:min_len] for p in probs_list], axis=0)
        y_true = y_true_ref[dataset_name][:min_len]
        y_pred = avg_probs.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        try:
            y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
            roc_auc = roc_auc_score(y_bin, avg_probs, average="macro", multi_class="ovr")
        except ValueError:
            roc_auc = float("nan")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
        rows.append({"model": "Ensemble", "dataset": dataset_name, "accuracy": acc, "macro_f1": f1, "roc_auc": roc_auc})
        plot_confusion_matrix(cm, f"Ensemble ({dataset_name})", out_dir / f"confusion_Ensemble_{dataset_name}.png")

        precision, recall, f1c, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0
        )
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            per_class_rows.append({
                "model": "Ensemble", "dataset": dataset_name, "class": cls_name,
                "precision": precision[cls_idx], "recall": recall[cls_idx],
                "f1": f1c[cls_idx], "support": int(support[cls_idx]),
            })

    df = pd.DataFrame(rows)
    df_per_class = pd.DataFrame(per_class_rows)

    # Delta accuracy table
    pivot = df.pivot(index="model", columns="dataset", values="accuracy")
    pivot["delta_accuracy"] = pivot.get("augmented", np.nan) - pivot.get("baseline", np.nan)
    pivot = pivot.reset_index()

    out_csv = out_dir / "classification_results.csv"
    out_delta = out_dir / "classification_delta.csv"
    out_per_class = out_dir / "classification_per_class.csv"
    df.to_csv(out_csv, index=False)
    pivot.to_csv(out_delta, index=False)
    df_per_class.to_csv(out_per_class, index=False)
    print(f"[classify] -> {out_csv}")
    print(f"[classify] -> {out_delta}")
    print(f"[classify] -> {out_per_class}")
    return df, pivot


if __name__ == "__main__":
    from src.bootstrap import get_preprocessed, get_spectral
    from src.decomposition.source_estimation import run_source_estimation
    from src.genai.wgan_gp import run_wgan_gp
    from src.classification.dataset import build_dataset

    prep = get_preprocessed()
    res, _, _ = run_source_estimation(get_spectral())
    wgan = run_wgan_gp(res)

    baseline = build_dataset(prep, augment=False)
    augmented = build_dataset(prep, wgan_models=wgan, augment=True)
    run_classification(baseline, augmented)

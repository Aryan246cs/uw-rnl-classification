"""Phase 7: Improved classification with log-FFT features and enhanced training."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src_improved.config_improved import (
    CLASSIFIER_EPOCHS_IMPROVED, CLASSIFIER_BATCH_IMPROVED,
    CLASSIFIER_LR_IMPROVED, CLASSIFIER_DROPOUT_IMPROVED, SEG_LEN_IMPROVED
)

N_FFT_FEATS = SEG_LEN_IMPROVED // 2 + 1  # 257 for seg_len=512


def extract_log_fft(segments: np.ndarray) -> np.ndarray:
    """Convert time-domain segments to log-magnitude FFT features.

    Components are already separated by frequency band — giving the classifier
    the frequency spectrum rather than raw samples is far more discriminative.
    """
    fft_mag = np.abs(np.fft.rfft(segments, axis=1))  # (N, N_FFT_FEATS)
    return np.log1p(fft_mag).astype(np.float32)


class ImprovedClassifier(nn.Module):
    """1D-CNN operating on log-magnitude FFT features."""
    def __init__(self, n_classes=6, n_feats=N_FFT_FEATS, dropout=CLASSIFIER_DROPOUT_IMPROVED):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, n_feats)
        return self.classifier(self.conv(x))


def run_improved_classification(components_all: dict, synthetic_data: dict,
                                 optimized_weights: dict, ship_type: str):
    """
    Train improved classifier using:
    1. Log-FFT features (more discriminative than raw samples for freq-band components)
    2. Real segments weighted by optimized weights
    3. Synthetic GAN segments
    4. Data shuffling, cosine LR schedule, best-model restore
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training on {device}")

    # Build category index from all observed categories
    cat_to_idx = {}
    idx_to_cat = {}
    for fpath, components in components_all.items():
        for cat in components.keys():
            if cat not in cat_to_idx:
                idx = len(cat_to_idx)
                cat_to_idx[cat] = idx
                idx_to_cat[idx] = cat

    # Collect real segments with optimized importance weights
    X_real, y_real, weights_real = [], [], []
    for fpath, components in components_all.items():
        opt_weights = optimized_weights[fpath]
        for cat, comp in components.items():
            signal = comp["signal"]
            n_segs = len(signal) // SEG_LEN_IMPROVED
            if n_segs == 0:
                continue
            segs = signal[:n_segs * SEG_LEN_IMPROVED].reshape(-1, SEG_LEN_IMPROVED)
            X_real.append(segs)
            y_real.append(np.full(len(segs), cat_to_idx[cat]))
            weights_real.append(np.full(len(segs), opt_weights.get(cat, 1.0)))

    X_real = np.concatenate(X_real, axis=0)
    y_real = np.concatenate(y_real, axis=0)
    weights_real = np.concatenate(weights_real, axis=0)

    # Collect synthetic GAN segments
    X_synth_list, y_synth_list = [], []
    for cat, synth_segs in synthetic_data.items():
        if cat in cat_to_idx:
            X_synth_list.append(synth_segs)
            y_synth_list.append(np.full(len(synth_segs), cat_to_idx[cat]))

    if X_synth_list:
        X_synth = np.concatenate(X_synth_list, axis=0)
        y_synth = np.concatenate(y_synth_list, axis=0)
        weights_synth = np.ones(len(X_synth), dtype=np.float32)
        X_all = np.concatenate([X_real, X_synth], axis=0)
        y_all = np.concatenate([y_real, y_synth], axis=0)
        weights_all = np.concatenate([weights_real, weights_synth], axis=0)
    else:
        X_all, y_all, weights_all = X_real, y_real, weights_real

    n_synth_total = len(X_synth_list[0]) if X_synth_list else 0
    print(f"  Dataset: {len(X_real)} real + {len(X_all) - len(X_real)} synthetic = {len(X_all)} total")

    # Extract log-FFT features (applied to both real and synthetic time-domain segments)
    X_all_feats = extract_log_fft(X_all)

    # Train/val split (stratified so each class is represented in both)
    X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
        X_all_feats, y_all, weights_all,
        test_size=0.2, random_state=42, stratify=y_all
    )

    # Normalize per-frequency-bin (fit on train, apply to val to avoid leakage)
    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std = X_train.std(axis=0, keepdims=True) + 1e-12
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    w_train_t = torch.FloatTensor(w_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    # Model, optimizer, cosine LR scheduler
    model = ImprovedClassifier(n_classes=len(cat_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LR_IMPROVED, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CLASSIFIER_EPOCHS_IMPROVED, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_acc = 0.0
    best_state = None

    for epoch in range(CLASSIFIER_EPOCHS_IMPROVED):
        model.train()

        # Shuffle training data each epoch
        perm = torch.randperm(len(X_train_t), device=device)
        X_ep = X_train_t[perm]
        y_ep = y_train_t[perm]
        w_ep = w_train_t[perm]

        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_ep), CLASSIFIER_BATCH_IMPROVED):
            batch_X = X_ep[i:i + CLASSIFIER_BATCH_IMPROVED]
            batch_y = y_ep[i:i + CLASSIFIER_BATCH_IMPROVED]
            batch_w = w_ep[i:i + CLASSIFIER_BATCH_IMPROVED]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = (criterion(outputs, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    Epoch {epoch+1}/{CLASSIFIER_EPOCHS_IMPROVED}, "
                  f"Loss={total_loss/n_batches:.4f}, "
                  f"Val={val_acc:.4f} (best={best_acc:.4f})")
            model.train()

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val_acc={best_acc:.4f})")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()

    accuracy = accuracy_score(y_val, val_preds)

    # Per-class F1 (correct: computed over all samples, not filtered subsets)
    n_classes = len(cat_to_idx)
    f1_all = f1_score(y_val, val_preds, average=None, zero_division=0,
                      labels=list(range(n_classes)))
    f1_scores = {cat: float(f1_all[idx]) for cat, idx in cat_to_idx.items()
                 if idx < len(f1_all)}

    confusion = confusion_matrix(y_val, val_preds)

    print(f"\n  Final Accuracy: {accuracy:.4f}")
    print(f"  F1 Scores by category:")
    for cat, f1 in sorted(f1_scores.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {f1:.4f}")

    return accuracy, f1_scores, confusion

"""
Phase 7c - Machinery-Component Classifier

A 1D-CNN over fixed-length waveform segments (the same
``RESIDUAL_SEGMENT_LEN``-sample representation used by Phase 10A's
WGAN-GP), predicting which Appendix-B machinery category a segment
belongs to. Labels come from Phase 7b's per-file decomposition
(``component_dataset.build_component_segments``).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

from src.config import RESIDUAL_SEGMENT_LEN, CLASSIFIER_LR, CLASSIFIER_EPOCHS, CLASSIFIER_BATCH
from src.device import get_device


class SegmentClassifier(nn.Module):
    """Conv1D(64) -> Conv1D(128) -> Conv1D(256) -> FC -> n_classes logits."""

    def __init__(self, n_classes: int, in_dim: int = RESIDUAL_SEGMENT_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        flat_dim = 256 * (in_dim // 8)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))
        h = h.flatten(1)
        return self.fc(h)


def train_classifier(
    X_train, y_train, X_val, y_val, n_classes: int,
    epochs: int = CLASSIFIER_EPOCHS, batch_size: int = CLASSIFIER_BATCH,
    lr: float = CLASSIFIER_LR, label: str = "component",
):
    """Trains a fresh SegmentClassifier and returns (model, history, val_report)."""
    device = get_device()
    model = SegmentClassifier(n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    history = {"train_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        losses = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_preds)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_acc"].append(float(val_acc))

        if (epoch + 1) % 10 == 0:
            print(f"[component_clf:{label}] epoch {epoch + 1}/{epochs}  "
                  f"loss={history['train_loss'][-1]:.4f}  val_acc={val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(1).cpu().numpy()
    report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    return model, history, report

"""
Phase 12b - Vessel classification model architectures.
Four architecture families per Sec. 5 "Vessel Classification Models" table,
scaled down for a (MEL_PATCH_SIZE x MEL_PATCH_SIZE) input and 4-class output.
"""

import math
import torch
import torch.nn as nn

from src.config import MEL_PATCH_SIZE, N_MFCC

N_CLASSES = 4


# ── CNN: Mel Spectrogram -> Conv2D x4 -> Dense -> Softmax ─────────────────


class CNNClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, P, P) -> (B, 1, P, P)
        h = self.net(x.unsqueeze(1)).flatten(1)
        return self.fc(h)


# ── ResNet-lite: residual blocks + GAP + FC ───────────────────────────────


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return torch.relu(h + self.shortcut(x))


class ResNetLite(nn.Module):
    """Residual blocks operating on the Welch-PSD-derived mel patch + GAP + FC."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.stem(x.unsqueeze(1))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)


# ── CRNN: Conv2D x3 + Bidirectional GRU + Dense ──────────────────────────


class CRNNClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        freq_after = MEL_PATCH_SIZE // 8
        self.gru = nn.GRU(64 * freq_after, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        # x: (B, P, P) treated as (freq, time) -> conv -> (B, C, F', T')
        h = self.conv(x.unsqueeze(1))
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F')
        out, _ = self.gru(h)
        return self.fc(out.mean(dim=1))


# ── Transformer: MFCC sequence + multi-head self-attention + class token ──


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, d_model: int = 64, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(N_MFCC, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=MEL_PATCH_SIZE + 1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                                            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, T, N_MFCC)
        h = self.input_proj(x)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.fc(h[:, 0, :])


MODEL_REGISTRY = {
    "CNN": (CNNClassifier, "mel"),
    "ResNet-lite": (ResNetLite, "mel"),
    "CRNN": (CRNNClassifier, "mel"),
    "Transformer": (TransformerClassifier, "mfcc"),
}

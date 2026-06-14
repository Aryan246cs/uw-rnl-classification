"""
Phase 10C - Beta-VAE for Latent Acoustic Source Representation
Learns a compact, continuous latent space (z_dim=64) over residual-noise /
source segments, enabling controlled interpolation and class-discriminative
structure analysis via t-SNE (Sec. 10C).

Encoder : Conv1D(32,8) -> Conv1D(64,8) -> Conv1D(128,4) -> Flatten -> Dense(512) -> [mu, logvar]
Decoder : Dense(512) -> Reshape -> ConvT1D(128,4) -> ConvT1D(64,8) -> ConvT1D(1,8)
Loss    : reconstruction + beta * KL(q(z|x) || N(0,I)),  beta=4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import (
    OUTPUTS_GENAI,
    RESIDUAL_SEGMENT_LEN,
    Z_DIM_VAE,
    VAE_BETA,
    VAE_EPOCHS,
    VAE_BATCH,
    VAE_LR,
)
from src.device import get_device
from src.genai.wgan_gp import prepare_segments


class BetaVAE(nn.Module):
    def __init__(self, seg_len: int = RESIDUAL_SEGMENT_LEN, z_dim: int = Z_DIM_VAE):
        super().__init__()
        self.seg_len = seg_len
        self.z_dim = z_dim

        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3),   # L/2
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),  # L/4
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),  # L/8
            nn.ReLU(),
        )
        self.flat_len = seg_len // 8
        self.enc_fc = nn.Linear(128 * self.flat_len, 512)
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)

        self.dec_fc = nn.Sequential(nn.Linear(z_dim, 512), nn.ReLU(), nn.Linear(512, 128 * self.flat_len), nn.ReLU())
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=8, stride=2, padding=3),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc_conv(x.unsqueeze(1)).flatten(1)
        h = torch.relu(self.enc_fc(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, self.flat_len)
        return self.dec_conv(h).squeeze(1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = VAE_BETA):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


def train_vae(data: np.ndarray, n_epochs: int = VAE_EPOCHS, batch_size: int = VAE_BATCH, lr: float = VAE_LR):
    device = get_device()
    model = BetaVAE(seg_len=data.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    tensor = torch.tensor(data, dtype=torch.float32, device=device)
    batch_size = min(batch_size, max(4, len(tensor) // 2))

    history = {"loss": [], "recon": [], "kld": []}
    for epoch in range(n_epochs):
        idx = torch.randperm(len(tensor))
        losses, recons, klds = [], [], []
        for start in range(0, max(len(tensor) - batch_size, 1), batch_size):
            batch = tensor[idx[start : start + batch_size]]
            if batch.size(0) < 2:
                continue
            recon, mu, logvar = model(batch)
            loss, rl, kl = vae_loss(recon, batch, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            recons.append(rl.item())
            klds.append(kl.item())
        if losses:
            history["loss"].append(float(np.mean(losses)))
            history["recon"].append(float(np.mean(recons)))
            history["kld"].append(float(np.mean(klds)))
        if (epoch + 1) % 20 == 0 and history["loss"]:
            print(
                f"[VAE] epoch {epoch + 1}/{n_epochs}  loss={history['loss'][-1]:.4f}  "
                f"recon={history['recon'][-1]:.4f}  kld={history['kld'][-1]:.4f}"
            )
    return model, history


def plot_training(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="Total")
    ax.plot(history["recon"], label="Reconstruction")
    ax.plot(history["kld"], label="KL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Beta-VAE Training")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "vae_training_loss.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[VAE] -> {path}")


def plot_latent_tsne(model, data: np.ndarray, labels: list, out_dir: Path):
    """Encode all segments to mu and project with t-SNE, coloured by class."""
    from sklearn.manifold import TSNE

    device = get_device()
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(data, dtype=torch.float32, device=device)
        mu, _ = model.encode(tensor)
        z = mu.cpu().numpy()
    model.train()

    n = len(z)
    perplexity = max(5, min(30, n // 4))
    if n < 10:
        print("[VAE] Too few samples for t-SNE - skipping plot")
        return

    z2 = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca").fit_transform(z)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    classes = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for c, color in zip(classes, colors):
        mask = np.array(labels) == c
        ax.scatter(z2[mask, 0], z2[mask, 1], s=8, alpha=0.6, label=c, color=color)
    ax.set_title("Beta-VAE Latent Space (t-SNE projection)")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "vae_latent_tsne.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[VAE] -> {path}")


def run_vae(residuals: dict):
    by_class = prepare_segments(residuals)
    if not by_class:
        print("[VAE] No segments available - skipping")
        return None

    all_segs, all_labels = [], []
    for cls, segs in by_class.items():
        all_segs.append(segs)
        all_labels += [cls] * len(segs)
    data = np.concatenate(all_segs, axis=0)

    print(f"[VAE] Training Beta-VAE on {len(data)} segments (z_dim={Z_DIM_VAE}, beta={VAE_BETA})")
    model, history = train_vae(data)

    out_dir = OUTPUTS_GENAI / "vae"
    plot_training(history, out_dir)
    plot_latent_tsne(model, data, all_labels, out_dir)
    return model, history


if __name__ == "__main__":
    from src.bootstrap import get_spectral
    from src.decomposition.source_estimation import run_source_estimation

    res, _, _ = run_source_estimation(get_spectral())
    run_vae(res)

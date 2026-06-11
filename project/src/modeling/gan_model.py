"""
Phase 10 – GAN-Based Synthetic Noise Generation
Simple Dense Generator + CNN Discriminator.
Train only after residual pipeline is complete.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import OUTPUTS_REPORTS


# ── Architecture ─────────────────────────────────────────────────────────────


class Generator(nn.Module):
    def __init__(self, z_dim: int = 64, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ─────────────────────────────────────────────────────────────────


def prepare_real_data(residuals: dict, segment_len: int = 256) -> np.ndarray:
    """Slice residuals into fixed-length segments for GAN training."""
    segs = []
    for residual in residuals.values():
        r = residual.flatten().astype(np.float32)
        r = (r - r.mean()) / (r.std() + 1e-12)
        for start in range(0, len(r) - segment_len, segment_len):
            segs.append(r[start : start + segment_len])
    return np.array(segs)


def train_gan(
    real_data: np.ndarray,
    z_dim: int = 64,
    n_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 2e-4,
) -> tuple:
    """
    Train GAN and return (generator, discriminator, loss_history).
    Designed to run quickly on CPU with the reduced dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = real_data.shape[1]

    G = Generator(z_dim, out_dim).to(device)
    D = Discriminator(out_dim).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    tensor_data = torch.tensor(real_data, dtype=torch.float32).to(device)

    g_losses, d_losses = [], []

    for epoch in range(n_epochs):
        # Shuffle
        idx = torch.randperm(len(tensor_data))
        epoch_d, epoch_g = [], []

        for start in range(0, len(tensor_data) - batch_size, batch_size):
            batch = tensor_data[idx[start : start + batch_size]]
            bsz = batch.size(0)

            # ── Train Discriminator ──
            z = torch.randn(bsz, z_dim, device=device)
            fake = G(z).detach()
            real_lbl = torch.ones(bsz, 1, device=device)
            fake_lbl = torch.zeros(bsz, 1, device=device)

            opt_D.zero_grad()
            d_loss = criterion(D(batch), real_lbl) + criterion(D(fake), fake_lbl)
            d_loss.backward()
            opt_D.step()

            # ── Train Generator ──
            z = torch.randn(bsz, z_dim, device=device)
            opt_G.zero_grad()
            g_loss = criterion(D(G(z)), real_lbl)
            g_loss.backward()
            opt_G.step()

            epoch_d.append(d_loss.item())
            epoch_g.append(g_loss.item())

        g_losses.append(np.mean(epoch_g))
        d_losses.append(np.mean(epoch_d))

        if (epoch + 1) % 50 == 0:
            print(
                f"[GAN] epoch {epoch + 1}/{n_epochs}  "
                f"G={g_losses[-1]:.4f}  D={d_losses[-1]:.4f}"
            )

    return G, D, {"g_loss": g_losses, "d_loss": d_losses}


def plot_gan_losses(loss_history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(loss_history["g_loss"], label="Generator")
    ax.plot(loss_history["d_loss"], label="Discriminator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "gan_training_loss.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[GAN] → {path}")


def generate_samples(
    G, z_dim: int = 64, n_samples: int = 10, device=None
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        fake = G(z).cpu().numpy()
    return fake


def run_gan_pipeline(residuals: dict):
    real_data = prepare_real_data(residuals)
    if len(real_data) < 10:
        print("[GAN] Not enough segments to train — skipping.")
        return None, None

    print(f"[GAN] Training on {len(real_data)} segments of length {real_data.shape[1]}")
    G, D, losses = train_gan(real_data)
    out_dir = OUTPUTS_REPORTS / "gan"
    plot_gan_losses(losses, out_dir)

    # Save a few synthetic samples
    synth = generate_samples(G)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "synthetic_noise_samples.npy", synth)
    print(f"[GAN] Synthetic samples saved → {out_dir / 'synthetic_noise_samples.npy'}")
    return G, synth


if __name__ == "__main__":
    from src.preprocessing.loader import inventory_dataset
    from src.preprocessing.preprocess import run_preprocessing
    from src.spectral.spectral_analysis import run_spectral_analysis
    from src.decomposition.source_estimation import run_source_estimation

    df_inv = inventory_dataset()
    prep = run_preprocessing(df_inv)
    spec = run_spectral_analysis(prep)
    res, _, _ = run_source_estimation(spec)
    run_gan_pipeline(res)

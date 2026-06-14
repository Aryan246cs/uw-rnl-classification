"""
Phase 10A - Wasserstein GAN with Gradient Penalty (WGAN-GP)
Learns the distribution of non-Gaussian residual noise segments
(per the residual epsilon[n] left after NMF source subtraction, Phase 7-8).

Architecture (Sec. 10A of the research framework):
  Generator : z ~ N(0, I_128) -> Dense(256) -> Dense(512) -> Dense(512) -> Dense(256, Tanh)
  Critic    : Conv1D(64,4) -> Conv1D(128,4) -> Conv1D(256,4) -> Flatten -> Dense(1)  (no sigmoid)
  Loss      : Wasserstein-1 distance + gradient penalty (lambda_gp = 10)
  Training  : critic updated 5x per generator step, Adam(lr=1e-4, b1=0.5, b2=0.9)
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
    MAX_SEGMENTS_PER_CLASS,
    Z_DIM_GAN,
    WGAN_CRITIC_ITERS,
    WGAN_LAMBDA_GP,
    WGAN_EPOCHS,
    WGAN_BATCH,
    WGAN_LR,
)
from src.device import get_device


# ── Architecture ──────────────────────────────────────────────────────────


class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM_GAN, out_dim: int = RESIDUAL_SEGMENT_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    """1D-convolutional critic. Outputs a raw Wasserstein score (no sigmoid)."""

    def __init__(self, in_dim: int = RESIDUAL_SEGMENT_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        flat_dim = 256 * (in_dim // 8)
        self.fc = nn.Linear(flat_dim, 1)

    def forward(self, x):
        # x: (batch, length) -> (batch, 1, length)
        h = self.conv(x.unsqueeze(1))
        h = h.flatten(1)
        return self.fc(h)


# ── Data preparation ─────────────────────────────────────────────────────


def prepare_segments(residuals: dict, segment_len: int = RESIDUAL_SEGMENT_LEN, max_per_class: int = MAX_SEGMENTS_PER_CLASS):
    """Slice residuals into fixed-length, normalised segments, grouped by class.

    Each class is randomly subsampled to ``max_per_class`` segments so that
    WGAN-GP training (5 critic steps/iteration) stays CPU-feasible.
    """
    by_class = {}
    for fpath, residual in residuals.items():
        cls = Path(fpath).parent.name
        r = residual.flatten().astype(np.float32)
        r = (r - r.mean()) / (r.std() + 1e-12)
        segs = [
            r[start : start + segment_len]
            for start in range(0, len(r) - segment_len, segment_len)
        ]
        if segs:
            by_class.setdefault(cls, []).extend(segs)

    rng = np.random.default_rng(42)
    out = {}
    for cls, segs in by_class.items():
        arr = np.array(segs, dtype=np.float32)
        if len(arr) > max_per_class:
            idx = rng.choice(len(arr), max_per_class, replace=False)
            arr = arr[idx]
        out[cls] = arr
    return out


# ── Gradient penalty ─────────────────────────────────────────────────────


def gradient_penalty(critic, real, fake, device):
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(bsz, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ── Training ──────────────────────────────────────────────────────────────


def train_wgan_gp(
    real_data: np.ndarray,
    z_dim: int = Z_DIM_GAN,
    n_epochs: int = WGAN_EPOCHS,
    batch_size: int = WGAN_BATCH,
    lr: float = WGAN_LR,
    label: str = "global",
):
    device = get_device()
    out_dim = real_data.shape[1]
    batch_size = min(batch_size, max(4, len(real_data) // 2))

    G = Generator(z_dim, out_dim).to(device)
    D = Critic(out_dim).to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    data = torch.tensor(real_data, dtype=torch.float32, device=device)
    history = {"critic_loss": [], "wasserstein": []}

    for epoch in range(n_epochs):
        idx = torch.randperm(len(data))
        epoch_w, epoch_c = [], []
        for start in range(0, max(len(data) - batch_size, 1), batch_size):
            real = data[idx[start : start + batch_size]]
            bsz = real.size(0)
            if bsz < 2:
                continue

            # ── Critic update (x5) ──
            for _ in range(WGAN_CRITIC_ITERS):
                z = torch.randn(bsz, z_dim, device=device)
                fake = G(z).detach()
                opt_D.zero_grad()
                d_real = D(real).mean()
                d_fake = D(fake).mean()
                gp = gradient_penalty(D, real, fake, device)
                d_loss = d_fake - d_real + WGAN_LAMBDA_GP * gp
                d_loss.backward()
                opt_D.step()

            # ── Generator update ──
            z = torch.randn(bsz, z_dim, device=device)
            opt_G.zero_grad()
            g_loss = -D(G(z)).mean()
            g_loss.backward()
            opt_G.step()

            epoch_c.append(d_loss.item())
            epoch_w.append((d_real - d_fake).item())

        if epoch_c:
            history["critic_loss"].append(float(np.mean(epoch_c)))
            history["wasserstein"].append(float(np.mean(epoch_w)))

        if (epoch + 1) % 20 == 0 and history["wasserstein"]:
            print(
                f"[WGAN-GP:{label}] epoch {epoch + 1}/{n_epochs}  "
                f"W={history['wasserstein'][-1]:.4f}  D_loss={history['critic_loss'][-1]:.4f}"
            )

    return G, D, history


def plot_training(history: dict, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["critic_loss"])
    axes[0].set_title(f"Critic Loss - {label}")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["wasserstein"])
    axes[1].set_title(f"Wasserstein Estimate - {label}")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"wgan_gp_training_{label}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[WGAN-GP] -> {path}")


def generate_samples(G, n_samples: int, z_dim: int = Z_DIM_GAN, device=None) -> np.ndarray:
    device = device or get_device()
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        out = G(z).cpu().numpy()
    G.train()
    return out


def run_wgan_gp(residuals: dict):
    """
    Train one WGAN-GP per vessel class on residual-noise segments.
    Returns: {class: {"generator": G, "history": history, "real": real_segments}}
    """
    out_dir = OUTPUTS_GENAI / "wgan_gp"
    by_class = prepare_segments(residuals)

    models = {}
    for cls, segs in by_class.items():
        if len(segs) < 8:
            print(f"[WGAN-GP] {cls}: too few segments ({len(segs)}) - skipping")
            continue
        print(f"[WGAN-GP] {cls}: training on {len(segs)} segments of length {segs.shape[1]}")
        G, D, history = train_wgan_gp(segs, label=cls)
        plot_training(history, cls, out_dir)
        models[cls] = {"generator": G, "history": history, "real": segs}

    return models


if __name__ == "__main__":
    from src.bootstrap import get_spectral
    from src.decomposition.source_estimation import run_source_estimation

    res, _, _ = run_source_estimation(get_spectral())
    run_wgan_gp(res)

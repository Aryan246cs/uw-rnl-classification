"""
Phase 10B - Conditional Denoising Diffusion Probabilistic Model (DDPM)
State-of-the-art generative model used here for source/spectrogram-patch
synthesis (Sec. 10B), conditioned on vessel-class identity. Scaled down from
the full research specification (T=1000 -> T=200, 64x64 -> 32x32 patches,
U-Net depth reduced) so that training completes on CPU for the 33-file
working subset, while preserving the architectural principles:

  - Linear beta schedule, T diffusion steps
  - U-Net score network with sinusoidal time embedding + class conditioning
  - Classifier-free guidance (p_uncond = 0.1 during training)
  - DDIM sampler for fast deterministic inference
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

from src.config import (
    OUTPUTS_GENAI,
    CLASS_DIRS,
    DIFFUSION_T,
    DDIM_STEPS,
    GUIDANCE_SCALE,
    DIFFUSION_PATCH,
    DIFFUSION_EPOCHS,
    DIFFUSION_BATCH,
    DIFFUSION_LR,
    N_FFT,
    HOP_LEN,
    N_MELS,
)
from src.device import get_device

CLASS_NAMES = list(CLASS_DIRS.keys())
N_CLASSES = len(CLASS_NAMES)
NULL_CLASS = N_CLASSES  # extra "unconditional" embedding slot


# ── Data preparation: random mel-spectrogram patches ─────────────────────


def extract_patches(preprocessed: dict, patch: int = DIFFUSION_PATCH, per_file: int = 12, rng_seed: int = 42):
    """Compute log-mel spectrograms and extract random (patch x patch) crops."""
    rng = np.random.default_rng(rng_seed)
    patches, labels = [], []
    for fpath, (signal, sr) in preprocessed.items():
        cls = Path(fpath).parent.name
        cls_idx = CLASS_NAMES.index(cls)
        mel = librosa.feature.melspectrogram(
            y=signal.astype(np.float32), sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalise to [-1, 1]
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-12)
        mel_norm = mel_norm * 2 - 1

        n_mel, n_t = mel_norm.shape
        if n_mel < patch or n_t < patch:
            continue
        for _ in range(per_file):
            i = rng.integers(0, n_mel - patch + 1)
            j = rng.integers(0, n_t - patch + 1)
            patches.append(mel_norm[i : i + patch, j : j + patch].astype(np.float32))
            labels.append(cls_idx)

    return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int64)


# ── U-Net score network ────────────────────────────────────────────────


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int = 64):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm(self.conv(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        return h


class TinyUNet(nn.Module):
    """3-level U-Net for 32x32 single-channel patches with FiLM-style conditioning."""

    def __init__(self, n_classes: int = N_CLASSES, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(64, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(n_classes + 1, emb_dim)

        self.down1 = ConvBlock(1, 32, emb_dim)
        self.pool1 = nn.AvgPool2d(2)  # 32 -> 16
        self.down2 = ConvBlock(32, 64, emb_dim)
        self.pool2 = nn.AvgPool2d(2)  # 16 -> 8

        self.bottleneck = ConvBlock(64, 128, emb_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 8 -> 16
        self.up_block2 = ConvBlock(128, 64, emb_dim)  # concat with down2 skip (64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 16 -> 32
        self.up_block1 = ConvBlock(64, 32, emb_dim)  # concat with down1 skip (32)

        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t, c):
        t_emb = self.time_mlp(sinusoidal_embedding(t, 64))
        c_emb = self.class_emb(c)
        emb = t_emb + c_emb

        d1 = self.down1(x, emb)            # (B,32,32,32)
        d2 = self.down2(self.pool1(d1), emb)  # (B,64,16,16)
        b = self.bottleneck(self.pool2(d2), emb)  # (B,128,8,8)

        u2 = self.up2(b)                    # (B,64,16,16)
        u2 = self.up_block2(torch.cat([u2, d2], dim=1), emb)
        u1 = self.up1(u2)                   # (B,32,32,32)
        u1 = self.up_block1(torch.cat([u1, d1], dim=1), emb)

        return self.out(u1)


# ── Diffusion schedule ──────────────────────────────────────────────────


class DiffusionSchedule:
    def __init__(self, T: int = DIFFUSION_T, device=None):
        self.T = T
        self.device = device or get_device()
        self.betas = torch.linspace(1e-4, 0.02, T, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        a = self.alphas_cumprod[t][:, None, None, None]
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise


# ── Training ──────────────────────────────────────────────────────────────


def train_ddpm(patches: np.ndarray, labels: np.ndarray, n_epochs: int = DIFFUSION_EPOCHS,
                batch_size: int = DIFFUSION_BATCH, lr: float = DIFFUSION_LR, p_uncond: float = 0.1):
    device = get_device()
    model = TinyUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = DiffusionSchedule(device=device)

    x = torch.tensor(patches, dtype=torch.float32, device=device).unsqueeze(1)  # (N,1,P,P)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    batch_size = min(batch_size, max(4, len(x) // 2))

    history = {"loss": []}
    for epoch in range(n_epochs):
        idx = torch.randperm(len(x))
        losses = []
        for start in range(0, max(len(x) - batch_size, 1), batch_size):
            xb = x[idx[start : start + batch_size]]
            yb = y[idx[start : start + batch_size]].clone()
            if xb.size(0) < 2:
                continue

            # classifier-free guidance: randomly drop conditioning
            drop_mask = torch.rand(yb.shape, device=device) < p_uncond
            yb[drop_mask] = NULL_CLASS

            t = torch.randint(0, sched.T, (xb.size(0),), device=device)
            noise = torch.randn_like(xb)
            x_t = sched.q_sample(xb, t, noise)
            pred = model(x_t, t, yb)
            loss = nn.functional.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if losses:
            history["loss"].append(float(np.mean(losses)))
        if (epoch + 1) % 10 == 0 and history["loss"]:
            print(f"[DDPM] epoch {epoch + 1}/{n_epochs}  loss={history['loss'][-1]:.5f}")

    return model, sched, history


# ── DDIM sampling ─────────────────────────────────────────────────────────


@torch.no_grad()
def ddim_sample(model, sched: DiffusionSchedule, class_idx: int, n_samples: int = 4,
                 steps: int = DDIM_STEPS, guidance: float = GUIDANCE_SCALE, patch: int = DIFFUSION_PATCH):
    device = sched.device
    model.eval()
    x = torch.randn(n_samples, 1, patch, patch, device=device)

    t_seq = torch.linspace(sched.T - 1, 0, steps, device=device).long()
    y_cond = torch.full((n_samples,), class_idx, dtype=torch.long, device=device)
    y_uncond = torch.full((n_samples,), NULL_CLASS, dtype=torch.long, device=device)

    for i, t in enumerate(t_seq):
        t_batch = t.repeat(n_samples)
        eps_cond = model(x, t_batch, y_cond)
        eps_uncond = model(x, t_batch, y_uncond)
        eps = eps_uncond + guidance * (eps_cond - eps_uncond)

        a_t = sched.alphas_cumprod[t]
        a_prev = sched.alphas_cumprod[t_seq[i + 1]] if i + 1 < len(t_seq) else torch.tensor(1.0, device=device)

        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps

    model.train()
    return x.squeeze(1).cpu().numpy()


# ── Plotting / orchestration ────────────────────────────────────────────


def plot_training(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"])
    ax.set_title("Conditional DDPM Training Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (noise prediction)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "ddpm_training_loss.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[DDPM] -> {path}")


def plot_generated_patches(model, sched, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(4 * N_CLASSES, 4))
    for i, cls in enumerate(CLASS_NAMES):
        samples = ddim_sample(model, sched, class_idx=i, n_samples=1)
        axes[i].imshow(samples[0], origin="lower", aspect="auto", cmap="magma")
        axes[i].set_title(f"{cls} (DDIM)")
        axes[i].axis("off")
    plt.tight_layout()
    path = out_dir / "ddpm_generated_patches.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[DDPM] -> {path}")


def run_ddpm(preprocessed: dict):
    patches, labels = extract_patches(preprocessed)
    if len(patches) < 16:
        print("[DDPM] Not enough patches - skipping")
        return None, None

    print(f"[DDPM] Training conditional DDPM on {len(patches)} patches ({DIFFUSION_PATCH}x{DIFFUSION_PATCH}, T={DIFFUSION_T})")
    model, sched, history = train_ddpm(patches, labels)

    out_dir = OUTPUTS_GENAI / "ddpm"
    plot_training(history, out_dir)
    plot_generated_patches(model, sched, out_dir)
    return model, sched


if __name__ == "__main__":
    from src.bootstrap import get_preprocessed

    run_ddpm(get_preprocessed())

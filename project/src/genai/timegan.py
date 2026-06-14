"""
Phase 10D - TimeGAN (lite) for Temporal Acoustic Sequence Synthesis
Preserves marginal distributions AND temporal dynamics of short acoustic
sequences (Sec. 10D), via the classic Embedder/Recovery/Generator/
Discriminator/Supervisor architecture (Yoon et al., 2019), scaled down
(hidden=24, seq_len=64) for CPU training on the 33-file working subset.

Training proceeds in three stages:
  1. Autoencoder (embedder + recovery) - reconstruction loss
  2. Supervised (next-latent prediction) - joint with autoencoder
  3. Joint adversarial training of generator/discriminator/supervisor
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample

from src.config import (
    OUTPUTS_GENAI,
    TIMEGAN_SEQ_LEN,
    TIMEGAN_HIDDEN,
    TIMEGAN_EPOCHS,
    TIMEGAN_BATCH,
    SEG_DURATION,
    MAX_SEQUENCES_PER_CLASS,
)
from src.device import get_device


# ── Data preparation ────────────────────────────────────────────────────


def prepare_sequences(preprocessed: dict, seq_len: int = TIMEGAN_SEQ_LEN):
    """
    Segment each preprocessed signal into 3s / 50% overlap windows, resample
    each window to seq_len points, and min-max normalise to [0, 1].
    Returns {class: array(n_seqs, seq_len, 1)}.
    """
    by_class = {}
    for fpath, (signal, sr) in preprocessed.items():
        cls = Path(fpath).parent.name
        seg_len = int(SEG_DURATION * sr)
        if seg_len <= 0 or len(signal) < seg_len:
            seg_len = len(signal)
            hop = seg_len
        else:
            hop = seg_len // 2

        seqs = []
        for start in range(0, max(len(signal) - seg_len, 0) + 1, max(hop, 1)):
            window = signal[start : start + seg_len]
            if len(window) < 8:
                continue
            resampled = resample(window, seq_len)
            lo, hi = resampled.min(), resampled.max()
            norm = (resampled - lo) / (hi - lo + 1e-12)
            seqs.append(norm.astype(np.float32))

        if seqs:
            by_class.setdefault(cls, []).extend(seqs)

    rng = np.random.default_rng(42)
    out = {}
    for cls, seqs_list in by_class.items():
        arr = np.array(seqs_list, dtype=np.float32)
        if len(arr) > MAX_SEQUENCES_PER_CLASS:
            idx = rng.choice(len(arr), MAX_SEQUENCES_PER_CLASS, replace=False)
            arr = arr[idx]
        out[cls] = arr[..., None]
    return out


# ── Architecture ─────────────────────────────────────────────────────────


class Embedder(nn.Module):
    def __init__(self, in_dim=1, hidden=TIMEGAN_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=3, batch_first=True)

    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(h)


class Recovery(nn.Module):
    def __init__(self, hidden=TIMEGAN_HIDDEN, out_dim=1):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, h):
        out, _ = self.gru(h)
        return torch.sigmoid(self.fc(out))


class GeneratorRNN(nn.Module):
    def __init__(self, in_dim=1, hidden=TIMEGAN_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=3, batch_first=True)

    def forward(self, z):
        h, _ = self.gru(z)
        return torch.sigmoid(h)


class Supervisor(nn.Module):
    def __init__(self, hidden=TIMEGAN_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)

    def forward(self, h):
        out, _ = self.gru(h)
        return torch.sigmoid(out)


class DiscriminatorRNN(nn.Module):
    def __init__(self, hidden=TIMEGAN_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, h):
        out, _ = self.gru(h)
        return self.fc(out[:, -1, :])


# ── Training ──────────────────────────────────────────────────────────────


def train_timegan(data: np.ndarray, n_epochs: int = TIMEGAN_EPOCHS, batch_size: int = TIMEGAN_BATCH, label: str = "global"):
    device = get_device()
    seq_len, feat_dim = data.shape[1], data.shape[2]
    batch_size = min(batch_size, max(4, len(data) // 2))

    emb, rec = Embedder(feat_dim).to(device), Recovery(out_dim=feat_dim).to(device)
    gen, sup, disc = GeneratorRNN(feat_dim).to(device), Supervisor().to(device), DiscriminatorRNN().to(device)

    opt_er = optim.Adam(list(emb.parameters()) + list(rec.parameters()), lr=1e-3)
    opt_sup = optim.Adam(list(sup.parameters()) + list(emb.parameters()), lr=1e-3)
    opt_g = optim.Adam(list(gen.parameters()) + list(sup.parameters()), lr=1e-3)
    opt_d = optim.Adam(disc.parameters(), lr=1e-3)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    tensor = torch.tensor(data, dtype=torch.float32, device=device)
    history = {"recon": [], "supervised": [], "g_loss": [], "d_loss": []}

    def batches():
        idx = torch.randperm(len(tensor))
        for start in range(0, max(len(tensor) - batch_size, 1), batch_size):
            b = tensor[idx[start : start + batch_size]]
            if b.size(0) >= 2:
                yield b

    # Stage 1: autoencoder
    for epoch in range(n_epochs):
        losses = []
        for x in batches():
            h = emb(x)
            x_hat = rec(h)
            loss = mse(x_hat, x)
            opt_er.zero_grad()
            loss.backward()
            opt_er.step()
            losses.append(loss.item())
        if losses:
            history["recon"].append(float(np.mean(losses)))

    # Stage 2: supervisor (next-step latent prediction)
    for epoch in range(n_epochs):
        losses = []
        for x in batches():
            h = emb(x)
            h_hat_sup = sup(h)
            loss = mse(h_hat_sup[:, :-1, :], h[:, 1:, :])
            opt_sup.zero_grad()
            loss.backward()
            opt_sup.step()
            losses.append(loss.item())
        if losses:
            history["supervised"].append(float(np.mean(losses)))

    # Stage 3: joint adversarial training
    for epoch in range(n_epochs):
        g_losses, d_losses = [], []
        for x in batches():
            bsz, sl, fd = x.shape
            z = torch.rand(bsz, sl, fd, device=device)

            # Discriminator
            with torch.no_grad():
                h_real = emb(x)
                e_hat = gen(z)
                h_fake = sup(e_hat)
            opt_d.zero_grad()
            d_real = disc(h_real)
            d_fake = disc(h_fake)
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            opt_d.step()
            d_losses.append(d_loss.item())

            # Generator + supervisor
            e_hat = gen(z)
            h_fake = sup(e_hat)
            d_fake = disc(h_fake)
            with torch.no_grad():
                h_real2 = emb(x)
            g_loss_adv = bce(d_fake, torch.ones_like(d_fake))
            g_loss_sup = mse(sup(h_real2)[:, :-1, :], h_real2[:, 1:, :])
            g_loss = g_loss_adv + 100 * g_loss_sup
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g_losses.append(g_loss.item())

        if g_losses:
            history["g_loss"].append(float(np.mean(g_losses)))
            history["d_loss"].append(float(np.mean(d_losses)))
        if (epoch + 1) % 25 == 0 and g_losses:
            print(
                f"[TimeGAN:{label}] joint epoch {epoch + 1}/{n_epochs}  "
                f"G={history['g_loss'][-1]:.4f}  D={history['d_loss'][-1]:.4f}"
            )

    return {"embedder": emb, "recovery": rec, "generator": gen, "supervisor": sup, "discriminator": disc}, history


def generate_sequences(models: dict, n_samples: int, seq_len: int = TIMEGAN_SEQ_LEN, feat_dim: int = 1, device=None) -> np.ndarray:
    device = device or get_device()
    gen, sup, rec = models["generator"], models["supervisor"], models["recovery"]
    gen.eval(); sup.eval(); rec.eval()
    with torch.no_grad():
        z = torch.rand(n_samples, seq_len, feat_dim, device=device)
        e_hat = gen(z)
        h_hat = sup(e_hat)
        x_hat = rec(h_hat)
    gen.train(); sup.train(); rec.train()
    return x_hat.cpu().numpy()


def plot_training(history: dict, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["recon"]); axes[0].set_title("Stage 1: Reconstruction")
    axes[1].plot(history["supervised"]); axes[1].set_title("Stage 2: Supervised")
    axes[2].plot(history["g_loss"], label="G"); axes[2].plot(history["d_loss"], label="D")
    axes[2].set_title("Stage 3: Joint Adversarial"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"timegan_training_{label}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[TimeGAN] -> {path}")


def plot_sample_sequences(real: np.ndarray, synth: np.ndarray, label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i in range(min(5, len(real))):
        axes[0].plot(real[i, :, 0], alpha=0.7)
    axes[0].set_title(f"Real sequences - {label}")
    for i in range(min(5, len(synth))):
        axes[1].plot(synth[i, :, 0], alpha=0.7)
    axes[1].set_title(f"TimeGAN-synthetic - {label}")
    for ax in axes:
        ax.set_xlabel("Timestep"); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"timegan_samples_{label}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[TimeGAN] -> {path}")


def run_timegan(preprocessed: dict):
    """Train one TimeGAN per vessel class on resampled 3s acoustic windows."""
    by_class = prepare_sequences(preprocessed)
    out_dir = OUTPUTS_GENAI / "timegan"

    models_by_class = {}
    for cls, seqs in by_class.items():
        if len(seqs) < 8:
            print(f"[TimeGAN] {cls}: too few sequences ({len(seqs)}) - skipping")
            continue
        print(f"[TimeGAN] {cls}: training on {len(seqs)} sequences of length {seqs.shape[1]}")
        models, history = train_timegan(seqs, label=cls)
        plot_training(history, cls, out_dir)
        synth = generate_sequences(models, n_samples=min(5, len(seqs)))
        plot_sample_sequences(seqs, synth, cls, out_dir)
        models_by_class[cls] = models

    return models_by_class


if __name__ == "__main__":
    from src.bootstrap import get_preprocessed

    run_timegan(get_preprocessed())

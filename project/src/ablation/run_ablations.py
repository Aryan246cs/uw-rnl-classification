"""
Ablation Studies
================
Two controlled experiments that justify the architectural choices made in
Phase 10A (WGAN-GP) and Phase 10C (Beta-VAE), run on the same residual-noise
segments produced by Phase 7-8 (NMF residuals -> fixed-length segments via
``src.genai.wgan_gp.prepare_segments``).

A. WGAN-GP: gradient penalty ON (lambda_gp=10, as used in the main pipeline)
   vs. OFF (lambda_gp=0, a plain Wasserstein critic with weight clipping
   disabled). Demonstrates *why* the gradient penalty term is required for
   stable critic training and for the generator to match the real
   residual-noise distribution (lower KL divergence vs. real).

B. Beta-VAE: beta=1 (a standard VAE / no extra disentanglement pressure)
   vs. beta=4 (the value used in the main pipeline). Demonstrates the
   reconstruction-vs-disentanglement trade-off: higher beta increases the
   KL term and typically improves class separability of the latent space
   (measured via silhouette score on the encoder means), at some cost to
   reconstruction fidelity.

Outputs:
  outputs/ablation/ablation_wgan_gp.csv
  outputs/ablation/ablation_wgan_gp.png
  outputs/ablation/ablation_vae.csv
  outputs/ablation/ablation_vae.png
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    OUTPUTS,
    Z_DIM_GAN,
    WGAN_CRITIC_ITERS,
    WGAN_EPOCHS,
    WGAN_BATCH,
    WGAN_LR,
    Z_DIM_VAE,
    VAE_EPOCHS,
    VAE_BATCH,
    VAE_LR,
)
from src.device import get_device
from src.genai.wgan_gp import Generator, Critic, gradient_penalty, prepare_segments, generate_samples
from src.genai.vae_latent import BetaVAE, vae_loss
from src.genai.synthetic_generator import kl_divergence

OUT_DIR = OUTPUTS / "ablation"


# ── A. WGAN-GP: gradient penalty ON vs OFF ─────────────────────────────────


def _train_wgan_variant(real_data: np.ndarray, lambda_gp: float, n_epochs: int, label: str):
    device = get_device()
    torch.manual_seed(42)
    out_dim = real_data.shape[1]
    batch_size = min(WGAN_BATCH, max(4, len(real_data) // 2))

    G = Generator(Z_DIM_GAN, out_dim).to(device)
    D = Critic(out_dim).to(device)
    opt_G = optim.Adam(G.parameters(), lr=WGAN_LR, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=WGAN_LR, betas=(0.5, 0.9))

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

            for _ in range(WGAN_CRITIC_ITERS):
                z = torch.randn(bsz, Z_DIM_GAN, device=device)
                fake = G(z).detach()
                opt_D.zero_grad()
                d_real = D(real).mean()
                d_fake = D(fake).mean()
                gp = gradient_penalty(D, real, fake, device) if lambda_gp > 0 else torch.tensor(0.0, device=device)
                d_loss = d_fake - d_real + lambda_gp * gp
                d_loss.backward()
                opt_D.step()
                if lambda_gp == 0:
                    # Plain WGAN requires weight clipping for a Lipschitz critic.
                    for p in D.parameters():
                        p.data.clamp_(-0.01, 0.01)

            z = torch.randn(bsz, Z_DIM_GAN, device=device)
            opt_G.zero_grad()
            g_loss = -D(G(z)).mean()
            g_loss.backward()
            opt_G.step()

            epoch_c.append(d_loss.item())
            epoch_w.append((d_real - d_fake).item())

        if epoch_c:
            history["critic_loss"].append(float(np.mean(epoch_c)))
            history["wasserstein"].append(float(np.mean(epoch_w)))

    return G, history


def run_wgan_gp_ablation(residuals: dict):
    by_class = prepare_segments(residuals)
    if not by_class:
        print("[ablation] No residual segments available - skipping WGAN-GP ablation")
        return None

    cls = max(by_class, key=lambda c: len(by_class[c]))
    real = by_class[cls]
    print(f"[ablation] WGAN-GP gradient-penalty ablation on class '{cls}' ({len(real)} segments)")

    device = get_device()
    rows, histories = [], {}
    for label, lam in [("with_GP (lambda=10)", 10.0), ("without_GP (lambda=0)", 0.0)]:
        G, history = _train_wgan_variant(real, lambda_gp=lam, n_epochs=WGAN_EPOCHS, label=label)
        synth = generate_samples(G, n_samples=min(2000, len(real) * 50), z_dim=Z_DIM_GAN, device=device)
        kl = kl_divergence(real, synth)
        tail = max(1, WGAN_EPOCHS // 10)
        w_final = float(np.mean(history["wasserstein"][-tail:])) if history["wasserstein"] else float("nan")
        d_stability = float(np.std(history["critic_loss"][-tail:])) if history["critic_loss"] else float("nan")
        rows.append({
            "variant": label, "class": cls, "kl_vs_real": round(kl, 5),
            "final_wasserstein_estimate": round(w_final, 4),
            "critic_loss_std_last10pct": round(d_stability, 4),
        })
        histories[label] = history
        print(f"[ablation] {label}: KL={kl:.5f}  W(final)={w_final:.4f}  D_loss_std={d_stability:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, history in histories.items():
        axes[0].plot(history["wasserstein"], label=label)
        axes[1].plot(history["critic_loss"], label=label)
    axes[0].set_title(f"Wasserstein Estimate - {cls}")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title(f"Critic Loss - {cls}")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ablation_wgan_gp.png", dpi=110)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ablation_wgan_gp.csv", index=False)
    print(f"[ablation] -> {OUT_DIR / 'ablation_wgan_gp.csv'}")
    print(f"[ablation] -> {OUT_DIR / 'ablation_wgan_gp.png'}")
    return df


# ── B. Beta-VAE: beta=1 vs beta=4 ───────────────────────────────────────────


def _train_vae_variant(data: np.ndarray, beta: float, n_epochs: int):
    device = get_device()
    torch.manual_seed(42)
    model = BetaVAE(seg_len=data.shape[1], z_dim=Z_DIM_VAE).to(device)
    opt = optim.Adam(model.parameters(), lr=VAE_LR)
    tensor = torch.tensor(data, dtype=torch.float32, device=device)
    batch_size = min(VAE_BATCH, max(4, len(tensor) // 2))

    history = {"loss": [], "recon": [], "kld": []}
    for epoch in range(n_epochs):
        idx = torch.randperm(len(tensor))
        losses, recons, klds = [], [], []
        for start in range(0, max(len(tensor) - batch_size, 1), batch_size):
            batch = tensor[idx[start : start + batch_size]]
            if batch.size(0) < 2:
                continue
            recon, mu, logvar = model(batch)
            loss, rl, kl = vae_loss(recon, batch, mu, logvar, beta=beta)
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

    return model, history


def run_vae_ablation(residuals: dict):
    by_class = prepare_segments(residuals)
    if not by_class:
        print("[ablation] No residual segments available - skipping Beta-VAE ablation")
        return None

    all_segs, all_labels = [], []
    for cls, segs in by_class.items():
        all_segs.append(segs)
        all_labels += [cls] * len(segs)
    data = np.concatenate(all_segs, axis=0)
    print(f"[ablation] Beta-VAE beta ablation on {len(data)} segments across {len(by_class)} classes")

    device = get_device()
    rows, histories = [], {}
    for beta in (1.0, 4.0):
        label = f"beta={beta:.0f}"
        model, history = _train_vae_variant(data, beta=beta, n_epochs=VAE_EPOCHS)

        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(data, dtype=torch.float32, device=device)
            mu, _ = model.encode(tensor)
            z = mu.cpu().numpy()
        model.train()

        try:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(z, all_labels))
        except ValueError:
            sil = float("nan")

        tail = max(1, VAE_EPOCHS // 10)
        rows.append({
            "variant": label,
            "final_recon_loss": round(float(np.mean(history["recon"][-tail:])), 5),
            "final_kld": round(float(np.mean(history["kld"][-tail:])), 5),
            "final_total_loss": round(float(np.mean(history["loss"][-tail:])), 5),
            "latent_silhouette_by_class": round(sil, 4),
        })
        histories[label] = history
        print(f"[ablation] {label}: recon={rows[-1]['final_recon_loss']}  "
              f"kld={rows[-1]['final_kld']}  silhouette={rows[-1]['latent_silhouette_by_class']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for label, history in histories.items():
        axes[0].plot(history["recon"], label=label)
        axes[1].plot(history["kld"], label=label)
        axes[2].plot(history["loss"], label=label)
    axes[0].set_title("Reconstruction Loss")
    axes[1].set_title("KL Term")
    axes[2].set_title("Total Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ablation_vae.png", dpi=110)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ablation_vae.csv", index=False)
    print(f"[ablation] -> {OUT_DIR / 'ablation_vae.csv'}")
    print(f"[ablation] -> {OUT_DIR / 'ablation_vae.png'}")
    return df


def run_ablations(residuals: dict):
    print("\n[ablation] === A. WGAN-GP gradient-penalty ablation ===")
    df_wgan = run_wgan_gp_ablation(residuals)
    print("\n[ablation] === B. Beta-VAE beta sensitivity ablation ===")
    df_vae = run_vae_ablation(residuals)
    return df_wgan, df_vae


if __name__ == "__main__":
    from src.bootstrap import get_spectral
    from src.decomposition.source_estimation import run_source_estimation

    res, _, _ = run_source_estimation(get_spectral())
    run_ablations(res)

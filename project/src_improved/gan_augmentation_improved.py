"""Phase 6: Improved GAN augmentation with longer training."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src_improved.config_improved import (
    WGAN_EPOCHS_IMPROVED, WGAN_BATCH_IMPROVED, WGAN_LR_IMPROVED,
    MAX_SEGMENTS_IMPROVED, SEG_LEN_IMPROVED, WGAN_LAMBDA_GP, Z_DIM_GAN
)


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM_GAN, seg_len=SEG_LEN_IMPROVED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, seg_len),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, seg_len=SEG_LEN_IMPROVED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seg_len, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake, device):
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_wgan_for_category(segments: np.ndarray, category_name: str):
    """Train WGAN-GP for a single category."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    segments = segments[:MAX_SEGMENTS_IMPROVED]
    segments = (segments - segments.mean()) / (segments.std() + 1e-12)
    X = torch.FloatTensor(segments).to(device)
    
    # Models
    G = Generator().to(device)
    C = Critic().to(device)
    
    opt_G = optim.Adam(G.parameters(), lr=WGAN_LR_IMPROVED, betas=(0.5, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=WGAN_LR_IMPROVED, betas=(0.5, 0.9))
    
    print(f"    Training WGAN for {category_name} ({len(segments)} segments)...")
    
    for epoch in range(WGAN_EPOCHS_IMPROVED):
        # Train Critic
        for _ in range(5):
            idx = torch.randint(0, len(X), (WGAN_BATCH_IMPROVED,))
            real = X[idx]
            
            z = torch.randn(WGAN_BATCH_IMPROVED, Z_DIM_GAN).to(device)
            fake = G(z).detach()
            
            opt_C.zero_grad()
            loss_C = -C(real).mean() + C(fake).mean()
            gp = gradient_penalty(C, real, fake, device)
            (loss_C + WGAN_LAMBDA_GP * gp).backward()
            opt_C.step()
        
        # Train Generator
        z = torch.randn(WGAN_BATCH_IMPROVED, Z_DIM_GAN).to(device)
        fake = G(z)
        opt_G.zero_grad()
        loss_G = -C(fake).mean()
        loss_G.backward()
        opt_G.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}/{WGAN_EPOCHS_IMPROVED}, "
                  f"C_loss={loss_C.item():.4f}, G_loss={loss_G.item():.4f}")
    
    return G


def run_improved_gan_pipeline(components_all: dict, residuals: dict, ship_type: str):
    """Train improved GANs for all component categories."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training GANs on {device}")
    
    # Collect segments by category
    by_category = {}
    for fpath, components in components_all.items():
        for cat, comp in components.items():
            signal = comp["signal"]
            # Create segments
            n_segs = len(signal) // SEG_LEN_IMPROVED
            segs = signal[:n_segs * SEG_LEN_IMPROVED].reshape(-1, SEG_LEN_IMPROVED)
            
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(segs)
    
    # Add residual category
    residual_segs = []
    for res in residuals.values():
        n_segs = len(res) // SEG_LEN_IMPROVED
        segs = res[:n_segs * SEG_LEN_IMPROVED].reshape(-1, SEG_LEN_IMPROVED)
        residual_segs.append(segs)
    by_category["Residual"] = residual_segs
    
    # Train GAN for each category
    gan_generators = {}
    synthetic_data = {}
    
    for cat, seg_list in by_category.items():
        segments = np.concatenate(seg_list, axis=0)
        
        if len(segments) < 100:
            print(f"  Skipping {cat}: insufficient segments ({len(segments)})")
            continue
        
        G = train_wgan_for_category(segments, cat)
        gan_generators[cat] = G
        
        # Generate synthetic samples
        G.eval()
        with torch.no_grad():
            n_synth = min(5000, len(segments))
            z = torch.randn(n_synth, Z_DIM_GAN).to(next(G.parameters()).device)
            synthetic = G(z).cpu().numpy()
        synthetic_data[cat] = synthetic
        print(f"  Generated {len(synthetic)} synthetic samples for {cat}")
    
    return gan_generators, synthetic_data

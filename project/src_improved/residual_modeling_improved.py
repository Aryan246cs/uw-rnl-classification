"""Phase 4: Enhanced residual noise modeling with deep learning."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src_improved.config_improved import OUTPUTS_IMPROVED


class ResidualAutoencoder(nn.Module):
    """Autoencoder for residual noise modeling."""
    def __init__(self, input_dim=512, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def run_improved_residual_modeling(residuals: dict, ship_type: str):
    """
    Model residual noise distribution using autoencoder.
    Returns model and characteristics for weight optimization.
    """
    print("  Training residual noise autoencoder...")
    
    # Pool all residuals and create segments
    all_residuals = []
    for res in residuals.values():
        all_residuals.append(res)
    pooled = np.concatenate(all_residuals)
    
    # Create segments
    seg_len = 512
    n_segs = len(pooled) // seg_len
    segments = pooled[:n_segs * seg_len].reshape(-1, seg_len)
    
    # Normalize
    segments = (segments - np.mean(segments, axis=1, keepdims=True)) / (np.std(segments, axis=1, keepdims=True) + 1e-12)
    
    # Train autoencoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Using device: {device}")
    
    model = ResidualAutoencoder(input_dim=seg_len, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X = torch.FloatTensor(segments).to(device)
    batch_size = 256
    epochs = 50
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X)*batch_size:.6f}")
    
    # Extract residual characteristics
    model.eval()
    with torch.no_grad():
        _, latent = model(X)
        latent_np = latent.cpu().numpy()
    
    characteristics = {
        'mean': np.mean(pooled),
        'std': np.std(pooled),
        'latent_mean': np.mean(latent_np, axis=0),
        'latent_std': np.std(latent_np, axis=0),
        'model': model
    }
    
    print(f"  Residual characteristics: mean={characteristics['mean']:.6f}, "
          f"std={characteristics['std']:.6f}")
    
    return model, characteristics

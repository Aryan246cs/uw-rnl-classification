"""Phase 5: Optimize machinery weights using residual back-calculation.

Implements your idea:
    RNL = SBN - residual_noise
    SBN = M * weights
    
Given:
- RNL (measured signal)
- M (machinery component signals)  
- residual_noise (modeled)

Optimize weights to minimize: ||RNL - (M @ weights + residual_noise)||^2
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from src_improved.config_improved import OUTPUTS_IMPROVED, WEIGHT_OPT_ITERATIONS, WEIGHT_OPT_LR


def optimize_machinery_weights(preprocessed: dict, components_all: dict, 
                                residuals: dict, residual_model, ship_type: str):
    """
    Optimize component weights via residual-aware back-calculation.
    
    For each file:
        x = Sum_i(w_i * m_i) + eps
        
    Optimize w_i to minimize reconstruction error given modeled eps.
    """
    print("  Optimizing weights using residual-constrained least squares...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimized_weights_all = {}
    rows = []
    
    for fpath in preprocessed.keys():
        fname = Path(fpath).name
        signal = preprocessed[fpath]
        components = components_all[fpath]
        residual = residuals[fpath]
        
        # Stack component signals as matrix M
        categories = sorted(components.keys())
        M = np.stack([components[cat]["signal"] for cat in categories], axis=1)  # (time, n_components)
        
        # Start from uniform weights — energy fractions are the wrong scale
        # for M@w where M contains full-amplitude component signals
        n_cats = len(categories)
        w_init = np.ones(n_cats, dtype=np.float32) / n_cats
        
        # Convert to torch
        M_torch = torch.FloatTensor(M).to(device)
        x_torch = torch.FloatTensor(signal).to(device)
        residual_torch = torch.FloatTensor(residual).to(device)
        
        # Optimizable weights (constrained to be positive and sum to ~1)
        w = torch.nn.Parameter(torch.FloatTensor(w_init).to(device))
        optimizer = optim.Adam([w], lr=WEIGHT_OPT_LR)
        
        # Optimize
        for it in range(WEIGHT_OPT_ITERATIONS):
            optimizer.zero_grad()
            
            # Weighted reconstruction
            x_recon = M_torch @ w.abs()  # abs() ensures positivity
            
            # Loss: reconstruction error + residual consistency
            recon_loss = torch.mean((x_torch - x_recon - residual_torch)**2)
            
            # Regularization: weights should sum to reasonable value
            weight_reg = torch.abs(w.abs().sum() - 1.0) * 0.1
            
            loss = recon_loss + weight_reg
            loss.backward()
            optimizer.step()
        
        # Extract optimized weights
        w_opt = w.abs().detach().cpu().numpy()
        w_opt = w_opt / (w_opt.sum() + 1e-12)  # Normalize
        
        optimized_weights = {}
        for cat, weight in zip(categories, w_opt):
            optimized_weights[cat] = float(weight)
            rows.append({
                "filename": fname,
                "category": cat,
                "initial_weight": components[cat]["weight"],
                "optimized_weight": float(weight),
                "improvement": float(weight - components[cat]["weight"])
            })
        
        optimized_weights_all[fpath] = optimized_weights
        print(f"    {fname}: optimized {len(categories)} weights")
    
    # Save results
    df = pd.DataFrame(rows)
    out_csv = OUTPUTS_IMPROVED / f"optimized_weights_{ship_type}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    
    # Show average improvement
    avg_improvement = df.groupby("category")["improvement"].mean()
    print(f"  Average weight improvements:")
    for cat, imp in avg_improvement.items():
        print(f"    {cat}: {imp:+.4f}")
    
    return optimized_weights_all

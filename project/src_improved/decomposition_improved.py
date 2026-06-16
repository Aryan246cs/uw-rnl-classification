"""Phase 3: Improved decomposition with residual infusion."""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from sklearn.decomposition import NMF
from src_improved.config_improved import (
    N_COMPONENTS_IMPROVED, NMF_MAX_ITER_IMPROVED, NMF_INIT_IMPROVED, 
    OUTPUTS_IMPROVED, N_FFT, HOP_LEN
)


def load_previous_residuals(ship_type: str):
    """Load residual noise from previous run."""
    residual_dir = Path("project/outputs/decomposition/separated_audio") / ship_type
    residuals = []
    
    if residual_dir.exists():
        for wav_file in residual_dir.glob("*_residual.wav"):
            try:
                y, sr = librosa.load(str(wav_file), sr=None)
                residuals.append(y)
                print(f"    Loaded previous residual: {wav_file.name}")
            except:
                pass
    
    if residuals:
        # Pool and normalize
        pooled = np.concatenate(residuals)
        return pooled / (np.std(pooled) + 1e-12)
    return None


def classify_frequency_band(freq_hz: float) -> str:
    """Map frequency to machinery category."""
    if freq_hz < 40:
        return "Engine Shaft & Propeller BPF"
    elif freq_hz < 100:
        return "Generator"
    elif freq_hz < 300:
        return "Pumps & Compressors"
    elif freq_hz < 800:
        return "Gearbox / Gear-mesh"
    else:
        return "Hull & High-Frequency Machinery"


def run_improved_decomposition(preprocessed: dict, sr: int, 
                                ship_type: str, residual_infusion: float):
    """
    Enhanced decomposition with:
    1. Higher component count (8 vs 6)
    2. Better NMF initialization
    3. Residual noise infusion from previous run
    """
    print(f"  Using {N_COMPONENTS_IMPROVED} NMF components (up from 6)")
    
    # Load previous residuals
    prev_residual = load_previous_residuals(ship_type)
    if prev_residual is not None:
        print(f"  Infusing {residual_infusion*100:.0f}% previous residual noise")
    else:
        print("  No previous residuals found, proceeding without infusion")
        residual_infusion = 0.0
    
    components_all = {}
    residuals = {}
    rows = []
    
    for fpath, signal in preprocessed.items():
        fname = Path(fpath).name
        
        # Infuse residual noise if available
        if prev_residual is not None and len(prev_residual) > 0:
            # Tile previous residual to match signal length
            n_reps = int(np.ceil(len(signal) / len(prev_residual)))
            residual_tile = np.tile(prev_residual, n_reps)[:len(signal)]
            residual_tile = residual_tile / (np.std(residual_tile) + 1e-12) * np.std(signal)
            signal_infused = signal + residual_infusion * residual_tile
        else:
            signal_infused = signal
        
        # STFT
        S = librosa.stft(signal_infused, n_fft=N_FFT, hop_length=HOP_LEN)
        phase = np.exp(1j * np.angle(S))
        mag = np.abs(S)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        
        # Enhanced NMF decomposition
        X = mag.T + 1e-12
        model = NMF(
            n_components=N_COMPONENTS_IMPROVED,
            init=NMF_INIT_IMPROVED,
            max_iter=NMF_MAX_ITER_IMPROVED,
            random_state=42,
            l1_ratio=0.5  # Sparsity for cleaner separation
        )
        H_act = model.fit_transform(X)
        W_spec = model.components_
        recon = H_act @ W_spec
        recon_T = recon.T + 1e-12
        
        # Reconstruct components
        explained_S = recon_T * phase
        x_hat = librosa.istft(explained_S, hop_length=HOP_LEN, length=len(signal))
        residual = signal - x_hat  # Residual from ORIGINAL signal
        
        # Energy normalization
        total_energy = float(np.sum(signal**2)) + 1e-12
        
        # Extract components by frequency bands
        grouped = {}
        for r in range(N_COMPONENTS_IMPROVED):
            # Component mask
            mask = np.outer(W_spec[r, :], H_act[:, r]) / recon_T
            comp_S = mask * explained_S
            comp_signal = librosa.istft(comp_S, hop_length=HOP_LEN, length=len(signal))
            
            # Classify by dominant frequency
            dom_idx = int(np.argmax(W_spec[r, :]))
            dom_freq = float(freqs[dom_idx])
            label = classify_frequency_band(dom_freq)
            
            if label not in grouped:
                grouped[label] = {"signal": np.zeros_like(comp_signal), "dom_freqs": []}
            grouped[label]["signal"] += comp_signal
            grouped[label]["dom_freqs"].append(round(dom_freq, 2))
        
        # Store components
        components_all[fpath] = {}
        for label, g in grouped.items():
            weight = float(np.sum(g["signal"]**2) / total_energy)
            components_all[fpath][label] = {
                "signal": g["signal"],
                "weight": weight,
                "dom_freqs": g["dom_freqs"]
            }
            rows.append({
                "filename": fname,
                "category": label,
                "weight": weight,
                "dominant_freqs": str(g["dom_freqs"])
            })
        
        # Store residual
        residual_weight = float(np.sum(residual**2) / total_energy)
        residuals[fpath] = residual
        rows.append({
            "filename": fname,
            "category": "Residual",
            "weight": residual_weight,
            "dominant_freqs": "[]"
        })
        
        print(f"    {fname}: {len(grouped)} components, residual={residual_weight:.3f}")
    
    # Save decomposition summary
    df = pd.DataFrame(rows)
    out_csv = OUTPUTS_IMPROVED / f"decomposition_{ship_type}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    
    return components_all, residuals, df

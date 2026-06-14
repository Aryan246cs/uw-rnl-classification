"""
Phase 7b - Machinery Source Decomposition (Source Decomposition / CWE objectives)

Decomposes each preprocessed hydrophone recording x[n] into named
ship-machinery noise components, following the additive mixture model
(Sec. 4.1 of the research framework):

    x[n] = Sum_i w_i * m_i[n]  +  eps[n]

For each file, a per-file NMF is run on the STFT magnitude spectrogram
V (freq x time):  V ~= W_spec @ H_act  (R=6 components). NMF's low-rank
reconstruction recon = H_act @ W_spec captures the *structured* (repeating,
quasi-periodic) part of the spectrogram - i.e. the deterministic machinery
tonals (shaft/propeller rotation harmonics, generator mains-lock, pump and
compressor blade-rate tones, gearbox mesh whine). Re-attaching the original
phase gives the "explained" signal:

    x_hat[n] = ISTFT(recon.T * exp(j*angle(S)))

The genuinely unmodeled part is the residual:

    eps[n] = x[n] - x_hat[n]

This is the broadband/stochastic content an R=6 low-rank spectral model
cannot represent - propeller cavitation, hydrodynamic flow noise and
Knudsen ambient ocean noise - and is exactly the non-Gaussian content
Phase 9 characterises and Phase 10A (WGAN-GP) models generatively.

x_hat is further split into named components m_i via Wiener-style soft
masks (which sum to 1 by construction, so Sum_i m_i[n] = x_hat[n] exactly):

    mask_r(f,t) = (W_spec[r,f] * H_act[t,r]) / recon.T(f,t)
    m_r[n]      = ISTFT(mask_r * x_hat_STFT)

Each component r is labelled against the Appendix B acoustic frequency map
by its dominant frequency, into one of 5 non-overlapping machinery bands.
Components sharing a label are summed, giving the final {label: m_i[n]}.

The contribution weight for component i is its energy fraction:
    w_i = ||m_i||^2 / ||x||^2

so that x[n] = Sum_i m_i[n] + eps[n], with Sum_i w_i + ||eps||^2/||x||^2 ~= 1.

The residual eps[n] (full recording length, time-domain) is consumed by
Phase 9 (residual characterisation) and Phase 10A (WGAN-GP).
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import NMF

from src.config import OUTPUTS, N_FFT

OUTPUTS_DECOMP = OUTPUTS / "decomposition"

N_COMPONENTS = 6
DECOMP_HOP = 1024          # coarser hop than Phase-3 STFT -> tractable per-file NMF
NMF_MAX_ITER = 100

# Appendix B - Acoustic Frequency Map for Naval Vessels (used for labelling)
# Non-overlapping bands assigned by each NMF component's dominant frequency.
CATEGORY_DESCRIPTIONS = {
    "Engine Shaft & Propeller BPF (tonal, <40 Hz)":
        "Engine shaft rotation fundamental (5-30 Hz) and propeller "
        "blade-passing-frequency (3-50 Hz) and their low harmonics.",
    "Generator (tonal, 40-100 Hz)":
        "Generator fundamental, often locked to 50/60 Hz mains frequency.",
    "Pumps & Compressors (tonal, 100-300 Hz)":
        "Pump and compressor blade/vane-rate tonals.",
    "Gearbox / Gear-mesh (tonal, 300-800 Hz)":
        "Gearbox gear-mesh whine (gear_teeth x RPM / 60).",
    "Hull & High-Frequency Machinery (tonal, >800 Hz)":
        "Higher-order machinery harmonics and structural/hull resonances "
        "excited at higher frequency.",
    "Cavitation, Flow & Ambient Noise (broadband residual)":
        "Propeller cavitation, hydrodynamic flow noise and Knudsen ambient "
        "ocean noise - the broadband content not captured by the low-rank "
        "(structured/tonal) NMF model.",
}

RESIDUAL_LABEL = "Cavitation, Flow & Ambient Noise (broadband residual)"


def classify_frequency_band(freq_hz: float) -> str:
    """Map a single frequency (Hz) to its Appendix B machinery category."""
    if freq_hz < 40:
        return "Engine Shaft & Propeller BPF (tonal, <40 Hz)"
    elif freq_hz < 100:
        return "Generator (tonal, 40-100 Hz)"
    elif freq_hz < 300:
        return "Pumps & Compressors (tonal, 100-300 Hz)"
    elif freq_hz < 800:
        return "Gearbox / Gear-mesh (tonal, 300-800 Hz)"
    else:
        return "Hull & High-Frequency Machinery (tonal, >800 Hz)"


def _classify_component(freqs: np.ndarray, spectrum: np.ndarray):
    """Label an NMF component's spectral template by its dominant frequency."""
    spec = spectrum + 1e-12
    dom_freq = float(freqs[int(np.argmax(spec))])
    return classify_frequency_band(dom_freq), dom_freq


def decompose_file(signal: np.ndarray, sr: int, n_components: int = N_COMPONENTS):
    """
    Returns:
      components: {label: {"signal": m_i[n], "weight": w_i, "dom_freqs": [...],
                            "n_sources": int}}
                   - 5 named machinery bands, Sum_i m_i[n] = x_hat[n]
      residual:    eps[n] = x[n] - x_hat[n]  (unmodeled broadband/stochastic content)
    """
    signal = signal.astype(np.float32)
    S = librosa.stft(signal, n_fft=N_FFT, hop_length=DECOMP_HOP)
    phase = np.exp(1j * np.angle(S))
    mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    X = mag.T  # (time, freq) - sklearn convention: rows = samples
    model = NMF(n_components=n_components, init="nndsvda", max_iter=NMF_MAX_ITER, random_state=42)
    H_act = model.fit_transform(X)   # (time, R) activations
    W_spec = model.components_       # (R, freq) spectral templates
    recon = H_act @ W_spec            # (time, freq)
    recon_T = recon.T + 1e-12         # (freq, time)

    explained_S = recon_T * phase
    x_hat = librosa.istft(explained_S, hop_length=DECOMP_HOP, length=len(signal))
    residual = signal - x_hat

    total_energy = float(np.sum(signal.astype(np.float64) ** 2)) + 1e-12

    grouped: dict = {}
    for r in range(n_components):
        mask = np.outer(W_spec[r, :], H_act[:, r]) / recon_T  # (freq, time)
        comp_S = mask * explained_S
        comp_signal = librosa.istft(comp_S, hop_length=DECOMP_HOP, length=len(signal))
        label, dom = _classify_component(freqs, W_spec[r, :])
        grouped.setdefault(label, {"signal": np.zeros_like(comp_signal), "dom_freqs": [], "n_sources": 0})
        grouped[label]["signal"] += comp_signal
        grouped[label]["dom_freqs"].append(round(dom, 2))
        grouped[label]["n_sources"] += 1

    components = {}
    for label, g in grouped.items():
        weight = float(np.sum(g["signal"].astype(np.float64) ** 2) / total_energy)
        components[label] = {
            "signal": g["signal"],
            "weight": weight,
            "dom_freqs": g["dom_freqs"],
            "n_sources": g["n_sources"],
        }

    return components, residual


def _plot_decomposition(signal, sr, components, residual, label, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(components) + 2
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True)
    t = np.arange(len(signal)) / sr

    axes[0].plot(t, signal, lw=0.4, color="black")
    axes[0].set_title(f"Original signal - {label}")

    for i, (cat, comp) in enumerate(components.items(), start=1):
        axes[i].plot(t, comp["signal"], lw=0.4)
        axes[i].set_title(f"{cat}  (w={comp['weight']:.4f})")

    axes[-1].plot(t, residual, lw=0.4, color="firebrick")
    axes[-1].set_title("Residual eps[n] = x[n] - x_hat[n] (unmodeled broadband/stochastic content)")
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    fig.savefig(out_dir / f"{label}_decomposition.png", dpi=100)
    plt.close(fig)


def run_machinery_decomposition(preprocessed: dict, save_audio_per_class: bool = True):
    """
    preprocessed: {filepath: (signal, sr)}
    Returns:
      residuals: {filepath: eps[n]}  (time-domain, full length)
      df:        per-file, per-category contribution weight table
      summary:   per-class mean contribution weight table
    """
    rows = []
    residuals = {}
    saved_audio_classes = set()

    plot_dir = OUTPUTS_DECOMP / "plots"
    audio_dir = OUTPUTS_DECOMP / "separated_audio"

    for fpath, (signal, sr) in preprocessed.items():
        p = Path(fpath)
        cls = p.parent.name
        label = f"{cls}_{p.stem}"

        components, residual = decompose_file(signal, sr)
        residuals[fpath] = residual

        for cat, comp in components.items():
            rows.append({
                "filename": p.name,
                "class": cls,
                "category": cat,
                "role": "machinery_source (w_i*m_i)",
                "contribution_weight": round(comp["weight"], 6),
                "dominant_freqs_Hz": str(comp["dom_freqs"]),
                "n_nmf_components": comp["n_sources"],
            })
        residual_weight = float(np.sum(residual.astype(np.float64) ** 2) /
                                 (np.sum(signal.astype(np.float64) ** 2) + 1e-12))
        rows.append({
            "filename": p.name, "class": cls, "category": RESIDUAL_LABEL,
            "role": "residual (Phase 9/10A input)",
            "contribution_weight": round(residual_weight, 6),
            "dominant_freqs_Hz": "[]", "n_nmf_components": 0,
        })

        print(f"[decomp7b] {label}: " +
              ", ".join(f"{cat.split(' ')[0]}={c['weight']:.3f}" for cat, c in components.items()) +
              f", residual={residual_weight:.3f}")

        # One illustrative plot + separated-audio set per vessel class
        if cls not in saved_audio_classes:
            _plot_decomposition(signal, sr, components, residual, label, plot_dir)
            if save_audio_per_class:
                out = audio_dir / cls
                out.mkdir(parents=True, exist_ok=True)
                sf.write(out / f"{p.stem}_original.wav", signal, sr)
                for cat, comp in components.items():
                    safe = cat.split(" (")[0].replace(" & ", "_").replace(" / ", "_").replace(" ", "_").replace("/", "_")
                    sf.write(out / f"{p.stem}_{safe}.wav", comp["signal"].astype(np.float32), sr)
                sf.write(out / f"{p.stem}_residual.wav", residual.astype(np.float32), sr)
            saved_audio_classes.add(cls)

    df = pd.DataFrame(rows)
    OUTPUTS_DECOMP.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUTS_DECOMP / "machinery_decomposition.csv"
    df.to_csv(out_csv, index=False)
    print(f"[decomp7b] -> {out_csv}")

    # Per-class mean contribution weights (the w_i of Eq. 4.1, averaged over files)
    summary = df.groupby(["class", "category", "role"])["contribution_weight"].mean().reset_index()
    out_summary = OUTPUTS_DECOMP / "machinery_decomposition_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[decomp7b] -> {out_summary}")

    return residuals, df, summary


if __name__ == "__main__":
    from src.bootstrap import get_preprocessed

    run_machinery_decomposition(get_preprocessed())

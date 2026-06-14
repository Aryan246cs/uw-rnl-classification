"""
Phase 6 – Machinery Signature Exploration
Candidate engine, propeller, and rotational harmonic frequencies
detected from PSD peaks — no hardcoded conclusions.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path

from src.config import OUTPUTS_REPORTS
from src.decomposition.machinery_decomposition import classify_frequency_band


def detect_harmonic_series(
    freqs, psd, min_freq=5.0, max_freq=500.0, n_harmonics=6, tol_hz=5.0
):
    """
    Sweep candidate fundamental frequencies and check how many harmonics
    align with PSD peaks.  Returns list of (f0, score, [matched harmonics]).
    """
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    f_sub = freqs[mask]
    p_sub = psd[mask]

    peaks, props = find_peaks(p_sub, height=p_sub.max() * 0.02, distance=3)
    peak_freqs = f_sub[peaks]

    candidates = []
    for f0 in peak_freqs:
        matched = []
        for h in range(1, n_harmonics + 1):
            target = f0 * h
            diffs = np.abs(peak_freqs - target)
            if diffs.min() <= tol_hz:
                matched.append(round(target, 2))
        score = len(matched)
        candidates.append((round(float(f0), 2), score, matched))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:10]  # top 10


def run_machinery_analysis(spectral_results: dict) -> pd.DataFrame:
    rows = []
    for fpath, data in spectral_results.items():
        cands = detect_harmonic_series(data["psd_freqs"], data["psd"])
        for f0, score, harmonics in cands[:3]:  # top 3 per file
            rows.append(
                {
                    "filename": Path(fpath).name,
                    "class": data["class"],
                    "candidate_f0_Hz": f0,
                    "harmonic_score": score,
                    "harmonics_found": str(harmonics),
                    "machinery_category": classify_frequency_band(f0),
                }
            )

    df = pd.DataFrame(rows)
    out = OUTPUTS_REPORTS / "machinery_signatures.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[machinery] Saved → {out}")
    return df


if __name__ == "__main__":
    from src.bootstrap import get_spectral

    run_machinery_analysis(get_spectral())

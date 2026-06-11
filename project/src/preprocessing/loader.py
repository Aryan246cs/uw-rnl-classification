"""
Phase 1 – Signal Loading
Load every WAV file, extract metadata, save CSV inventory.
"""

import soundfile as sf
import pandas as pd
from pathlib import Path
from src.config import CLASS_DIRS, OUTPUTS_FEATURES


def load_wav(filepath: Path):
    """Return (signal_array, sample_rate) for a WAV file."""
    signal, sr = sf.read(str(filepath), dtype="float32", always_2d=False)
    if signal.ndim > 1:  # stereo → mono
        signal = signal.mean(axis=1)
    return signal, sr


def inventory_dataset() -> pd.DataFrame:
    """Walk all class directories and build a metadata DataFrame."""
    records = []
    for cls, folder in CLASS_DIRS.items():
        for wav in sorted(folder.glob("*.wav"), key=lambda p: int(p.stem)):
            signal, sr = load_wav(wav)
            n_samples = len(signal)
            duration = n_samples / sr
            records.append(
                {
                    "filename": wav.name,
                    "class": cls,
                    "filepath": str(wav),
                    "duration_s": round(duration, 4),
                    "sample_rate": sr,
                    "n_samples": n_samples,
                }
            )
    df = pd.DataFrame(records)
    out = OUTPUTS_FEATURES / "inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[loader] Inventory saved → {out}  ({len(df)} files)")
    return df


if __name__ == "__main__":
    df = inventory_dataset()
    print(df.to_string(index=False))

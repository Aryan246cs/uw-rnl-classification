"""
Phase 7c - Machinery-Component Segment Dataset

Builds a labeled dataset of fixed-length waveform segments for the
machinery-component classifier (Phase 7c) and its per-category WGAN-GP
augmentation, using Phase 7b's per-file decomposition (``decompose_file``)
to assign each segment a machinery-category label:

    {label: m_i[n]}  +  {RESIDUAL_LABEL: eps[n]}

Segments are sliced the same way as Phase 10A's WGAN-GP residual segments
(length ``RESIDUAL_SEGMENT_LEN``, z-score normalised per signal) so the same
generator/critic/classifier architectures apply directly.
"""

import numpy as np

from src.decomposition.machinery_decomposition import decompose_file, RESIDUAL_LABEL
from src.config import RESIDUAL_SEGMENT_LEN, COMPONENT_MAX_SEGMENTS_PER_CATEGORY


def _slice_segments(signal: np.ndarray, segment_len: int = RESIDUAL_SEGMENT_LEN):
    sig = signal.flatten().astype(np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-12)
    return [
        sig[start:start + segment_len]
        for start in range(0, len(sig) - segment_len, segment_len)
    ]


def build_component_segments(preprocessed: dict, max_per_category: int = COMPONENT_MAX_SEGMENTS_PER_CATEGORY):
    """
    preprocessed: {filepath: (signal, sr)}

    Returns: {category_label: np.ndarray of shape (N, RESIDUAL_SEGMENT_LEN)}
             covering the 5 named Appendix-B machinery categories plus the
             residual (cavitation/flow/ambient) label, each capped at
             ``max_per_category`` segments.
    """
    by_category: dict = {}

    for fpath, (signal, sr) in preprocessed.items():
        components, residual = decompose_file(signal, sr)
        for cat, comp in components.items():
            by_category.setdefault(cat, []).extend(_slice_segments(comp["signal"]))
        by_category.setdefault(RESIDUAL_LABEL, []).extend(_slice_segments(residual))

    rng = np.random.default_rng(42)
    out = {}
    for cat, segs in by_category.items():
        arr = np.array(segs, dtype=np.float32)
        if len(arr) > max_per_category:
            idx = rng.choice(len(arr), max_per_category, replace=False)
            arr = arr[idx]
        out[cat] = arr
    return out

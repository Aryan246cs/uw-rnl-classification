"""
Shared bootstrap: load preprocessed + spectral results from cache if available,
otherwise run phases 1-3 and cache the results.
"""

from src.cache import load_preprocessed, save_preprocessed, load_spectral, save_spectral


def get_preprocessed(apply_bandpass=True):
    data = load_preprocessed()
    if data is not None:
        return data
    from src.preprocessing.loader import inventory_dataset
    from src.preprocessing.preprocess import run_preprocessing

    df = inventory_dataset()
    data = run_preprocessing(df, apply_bandpass=apply_bandpass)
    save_preprocessed(data)
    return data


def get_spectral():
    data = load_spectral()
    if data is not None:
        return data
    prep = get_preprocessed()
    from src.spectral.spectral_analysis import run_spectral_analysis

    data = run_spectral_analysis(prep)
    save_spectral(data)
    return data

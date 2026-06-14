# DeepShip – Underwater Acoustic Signal Analysis Pipeline

---

## About the Dataset

DeepShip is a real-world underwater acoustic dataset recorded in the Strait of Georgia
delta node between 02 May 2016 and 04 October 2018. It contains 47 hours and 04 minutes
of recordings from 265 different ships across four commercial vessel classes:

  - Cargo
  - Passengership
  - Tanker
  - Tug

Recordings capture real sea conditions across different seasons, including natural
background noise, marine mammal noise, and other ambient activity. The dataset is
intended to support research in underwater acoustic signal analysis, vessel
identification, and noise characterization.

Full dataset access: contact mirfan@mail.nwpu.edu.cn

---

## Dataset Reduction (Local Working Subset)

To keep the repository lightweight and focused, the dataset was trimmed to a working
subset. WAV files were sorted numerically and the first N files per class were retained:

  Class           Files Retained
  --------------- --------------
  Cargo           10
  Passengership   10
  Tanker          10
  Tug              3  (only 3 existed)
  --------------- --------------
  Total            33 WAV files

The dataset folders (Cargo/, Passengership/, Tanker/, Tug/) are excluded from version
control via .gitignore because the WAV files are large. To reproduce results, obtain
the dataset and place the appropriate WAV files in those folders before running the
pipeline.

---

## Project Goal

This is NOT a simple classification project.

**Core deliverable**: given a hydrophone recording, decompose it into named
ship-machinery noise components — engine/propeller, generator, pumps &
compressors, gearbox, hull/high-frequency machinery — plus a residual
(cavitation, flow, ambient ocean noise), with a quantified energy contribution
per component:

```
x[n] = Sum_i m_i[n] + eps[n]
```

Around that core decomposition, the pipeline also builds:

  1. Signal understanding & feature extraction
  2. Machinery signature / harmonic analysis
  3. Per-file source decomposition (the core deliverable, Phase 7b)
  4. Residual noise statistical modeling
  5. GenAI-based synthetic noise generation (WGAN-GP, Beta-VAE, TimeGAN, DDPM)
  6. RNL-SBN mapping (hull transfer function + propagation loss)
  7. Vessel classification (baseline vs. GenAI-augmented)

See `project/outputs/reports/final_report.md` for the full, numbers-backed
write-up of every phase, and `project/outputs/reports/completion_report.md`
for a short, results-focused summary (problem statement, methodology,
per-class component contributions, and model results).

---

## Project Structure

  project/
  |
  +-- run_pipeline.py          Master runner — executes all 12 phases in order
  +-- post_run.py              Re-runs Phases 6-12b + report from cached Phase 1-3 data
  +-- requirements.txt         Python dependencies
  |
  +-- outputs/
  |   +-- waveforms/, fft/, psd/, spectrograms/   Per-file plots (Phases 2-3)
  |   +-- features/             inventory.csv, features.csv (Phases 1, 4)
  |   +-- decomposition/         Phase 7b: per-file decomposition CSVs, plots,
  |   |                          separated component/residual audio
  |   +-- genai/                 WGAN-GP, Beta-VAE, TimeGAN, DDPM outputs + synthetic corpus
  |   +-- mapping/                Phase 12a: hull transfer function, transfer loss
  |   +-- classification/         Phase 12b: classifier results, confusion matrices
  |   +-- ablation/                WGAN-GP / Beta-VAE ablation studies
  |   +-- reports/                 Summary plots, CSVs, final_report.md, completion_report.md
  |
  +-- src/
      +-- config.py            Central config (paths, parameters)
      +-- bootstrap.py / cache.py   Cached Phase 1-3 loading for post_run.py
      +-- preprocessing/
      |   +-- loader.py        Phase 1: WAV loading, metadata inventory
      |   +-- preprocess.py    Phase 2: DC removal, normalization, bandpass
      +-- spectral/
      |   +-- spectral_analysis.py   Phase 3: FFT, STFT, spectrogram, PSD
      +-- features/
      |   +-- feature_extraction.py  Phase 4: all features -> features.csv
      +-- signatures/
      |   +-- signature_discovery.py   Phase 5: avg FFT/PSD per class
      |   +-- machinery_signatures.py  Phase 6: harmonic peak detection + category label
      +-- decomposition/
      |   +-- source_estimation.py     Phase 7: global PSD-domain NMF (feeds Phase 12a)
      |   +-- machinery_decomposition.py  Phase 7b: per-file named source decomposition
      |   +-- residual_analysis.py     Phase 8: residual noise statistics
      +-- modeling/
      |   +-- noise_modeling.py        Phase 9: GMM + alpha-stable fit
      +-- genai/
      |   +-- wgan_gp.py                Phase 10A: WGAN-GP residual generator
      |   +-- vae_latent.py             Phase 10C: Beta-VAE latent representation
      |   +-- timegan.py                Phase 10D: TimeGAN sequence synthesis
      |   +-- ddpm_separator.py         Phase 10B: conditional DDPM spectrogram patches
      |   +-- synthetic_generator.py    Phase 11: synthetic corpus assembly + KL gate
      +-- mapping/
      |   +-- rnl_sbn_mapping.py        Phase 12a: hull transfer function + Thorp TL
      +-- classification/
      |   +-- dataset.py, train.py      Phase 12b: classifier datasets + training
      +-- ablation/
      |   +-- run_ablations.py          WGAN-GP / Beta-VAE ablation studies
      +-- reporting/
          +-- generate_report.py        Builds outputs/reports/final_report.md

---

## Pipeline Phases (summary)

| Phase | Description |
|---|---|
| 1 | Signal inventory — `outputs/features/inventory.csv` |
| 2 | Preprocessing — DC removal, normalization, 10-1000 Hz Butterworth bandpass |
| 3 | Spectral analysis — FFT, STFT/mel spectrogram, Welch PSD |
| 4 | Feature extraction — 65-column feature table |
| 5 | Signature discovery — per-class average FFT/PSD comparison |
| 6 | Machinery signature exploration — harmonic-series detection + Appendix B frequency-band label |
| 7 | Global PSD-domain NMF (3 components) — feeds Phase 12a |
| **7b** | **Per-file machinery source decomposition (core deliverable)** — STFT-NMF (R=6) + Wiener masking + Appendix B labelling -> named components + residual `eps[n]` |
| 8 | Residual noise analysis (KS, Shapiro-Wilk, kurtosis/skewness) on `eps[n]` |
| 9 | GMM (2/3/5 components, BIC-selected) + alpha-stable baseline fit on `eps[n]` |
| 10A | WGAN-GP — learns `eps[n]` distribution per class for data augmentation |
| 10B | Conditional DDPM — class-conditioned spectrogram-patch synthesis |
| 10C | Beta-VAE — latent representation of residual segments |
| 10D | TimeGAN — temporal waveform sequence synthesis |
| 11 | Synthetic corpus assembly + KL-divergence quality gate (WGAN-GP only) |
| 12a | RNL-SBN mapping — hull transfer function `H(f)` + Thorp transfer-loss model |
| 12b | Vessel classification — CNN / ResNet-lite / CRNN / Transformer + Ensemble, baseline vs. GenAI-augmented |

Full methodology, formulas, and per-phase results: `outputs/reports/final_report.md`.

---

## How to Run

1. Install dependencies:
     pip install -r project/requirements.txt

2. Place WAV files in the correct class folders:
     Cargo/        <- WAV files numbered e.g. 15.wav, 27.wav ...
     Passengership/
     Tanker/
     Tug/

3. Run the full pipeline from the repo root (all 12 phases, Phase 1-3 cached
   on first run):
     python project/run_pipeline.py

4. To re-run Phases 6-12b + rebuild the report from cached Phase 1-3 data
   (faster iteration once Phase 1-3 caches exist):
     python project/post_run.py

   Or run any individual phase module, e.g.:
     python -m src.decomposition.machinery_decomposition   (from inside project/)

5. Generate the final Markdown report on its own:
     python -m src.reporting.generate_report    (from inside project/)

---

## Configuration

All tunable parameters are in project/src/config.py:

  TARGET_SR              Sample rate (None = keep native)
  BANDPASS_LOW/HIGH      Bandpass filter cutoffs (Hz)
  N_FFT / HOP_LEN        STFT parameters
  N_MELS                 Number of mel bands
  WELCH_NPERSEG          Welch PSD segment length
  N_MFCC                 Number of MFCC coefficients
  MAX_SEGMENTS_PER_CLASS WGAN-GP/ablation segment cap per class
  VAE_BETA, VAE_EPOCHS   Beta-VAE training parameters
  WGAN_EPOCHS, WGAN_LR, WGAN_LAMBDA_GP   WGAN-GP training parameters
  CLASSIFIER_EPOCHS, CLASSIFIER_FOLDS    Phase 12b classifier training

---

## Dependencies

  numpy, scipy, librosa, soundfile, matplotlib, pandas,
  scikit-learn, torch

See project/requirements.txt for pinned versions.

---

## Original Dataset Reference

DeepShip dataset — Northwestern Polytechnical University
Contact: mirfan@mail.nwpu.edu.cn
Recording period: 02 May 2016 – 04 October 2018
Location: Strait of Georgia delta node

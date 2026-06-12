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

The objective is to progressively analyze underwater ship-radiated noise recordings
and build toward:

  1. Signal understanding
  2. Feature extraction
  3. Machinery signature analysis
  4. Source decomposition
  5. Residual noise modeling
  6. Synthetic noise generation

---

## Project Structure

  project/
  |
  +-- run_pipeline.py          Master runner — executes all 11 phases in order
  +-- requirements.txt         Python dependencies
  |
  +-- data/                    (reserved for any additional data files)
  +-- notebooks/               (reserved for Jupyter exploration)
  +-- docs/                    (reserved for extended documentation)
  |
  +-- outputs/
  |   +-- waveforms/           Before/after preprocessing plots
  |   +-- fft/                 FFT and STFT plots per file
  |   +-- psd/                 PSD (Welch) plots per file
  |   +-- spectrograms/        Mel spectrogram PNG files
  |   +-- features/            inventory.csv, features.csv
  |   +-- reports/             Summary plots and CSV reports
  |
  +-- src/
      +-- config.py            Central config (paths, parameters)
      +-- preprocessing/
      |   +-- loader.py        Phase 1: WAV loading, metadata inventory
      |   +-- preprocess.py    Phase 2: DC removal, normalization, bandpass
      +-- spectral/
      |   +-- spectral_analysis.py   Phase 3: FFT, STFT, spectrogram, PSD
      +-- features/
      |   +-- feature_extraction.py  Phase 4: all features -> features.csv
      +-- signatures/
      |   +-- signature_discovery.py   Phase 5: avg FFT/PSD per class
      |   +-- machinery_signatures.py  Phase 6: harmonic peak detection
      +-- decomposition/
      |   +-- source_estimation.py     Phase 7: NMF source decomposition
      |   +-- residual_analysis.py     Phase 8: residual noise statistics
      +-- modeling/
          +-- noise_modeling.py        Phase 9: GMM + alpha-stable fit
          +-- gan_model.py             Phase 10: GAN synthetic noise
          +-- validation.py            Phase 11: MSE, KL, JS divergence

---

## Pipeline Phases

### Phase 1 – Signal Loading
Scans all class folders for WAV files. Extracts filename, class, duration,
sample rate, and number of samples. Saves results to:
  outputs/features/inventory.csv

### Phase 2 – Preprocessing
For every signal:
  - DC offset removal (subtract mean)
  - Amplitude normalization (peak normalization)
  - Bandpass filter (default: 10 Hz – 1000 Hz, Butterworth order 4)
  - Before/after waveform comparison plots saved to outputs/waveforms/

### Phase 3 – Spectral Analysis
For every preprocessed signal:
  - FFT via numpy.fft.rfft() with Hanning window
    -> Dominant frequencies and peak magnitudes extracted
  - STFT via librosa.stft()
  - Mel Spectrogram via librosa.feature.melspectrogram()
  - PSD via scipy.signal.welch()
  All plots saved as PNG files under outputs/fft/, outputs/spectrograms/, outputs/psd/

### Phase 4 – Feature Extraction
Extracts a comprehensive feature vector per signal:
  - Spectral: centroid, bandwidth, roll-off, flatness, top peak frequencies
  - Harmonic: fundamental frequency (f0), voiced ratio, 2f/3f/4f estimates
  - Energy: RMS, band energy (4 bands), PSD peak frequencies and magnitudes
  - Statistical: mean, variance, std, kurtosis, skewness
  - MFCC: 13 coefficients (mean + std each)
Saved to: outputs/features/features.csv

### Phase 5 – Signature Discovery
Groups signals by vessel class and computes element-wise average FFT and PSD.
Generates:
  - Per-class overlay comparison (all 4 classes on one plot)
  - 2x2 grid of individual class PSD profiles
Saved to: outputs/reports/

### Phase 6 – Machinery Signature Exploration
Research-oriented. Does NOT hardcode any conclusions.
Sweeps candidate fundamental frequencies and scores how many harmonics (up to 6)
align with PSD peaks. Outputs a ranked list of candidate machinery frequencies
per file to: outputs/reports/machinery_signatures.csv

### Phase 7 – Source Contribution Estimation
Models each signal's PSD as:
  RNL = w1*M1 + w2*M2 + w3*M3 + e
where M are latent source components and e is residual noise.
Uses Non-Negative Matrix Factorization (NMF) to decompose the PSD matrix.
  - Component spectra saved as plots
  - Per-file source weights saved to: outputs/reports/nmf_weights.csv
  - Residual signals passed forward to Phase 8

### Phase 8 – Residual Noise Analysis
Analyzes each residual signal for noise character:
  - Kolmogorov-Smirnov test vs. normal distribution
  - Shapiro-Wilk test
  - Kurtosis and skewness
  - Classification: Gaussian / Near-Gaussian / Non-Gaussian / Heavy-tailed
Saved to: outputs/reports/residual_analysis.csv
Histogram plots saved to: outputs/reports/residuals/

### Phase 9 – Noise Modeling
Fits Gaussian Mixture Models (GMM) to each residual distribution using 2, 3,
and 5 components. Selects best fit by BIC. Also attempts alpha-stable distribution
fit via scipy.stats.levy_stable where available.
  - Model comparison scores saved to: outputs/reports/noise_model_scores.csv
  - Fitted distribution plots saved to: outputs/reports/noise_models/

### Phase 10 – GAN-Based Synthetic Noise Generation
A modular Dense GAN trained on fixed-length residual noise segments:
  - Generator: 4-layer dense network, Tanh output
  - Discriminator: 3-layer dense network, Sigmoid output
  - Input: random Gaussian noise (z_dim=64)
  - Output: synthetic noise segments of length 256
Training loss curves saved. Synthetic samples saved as .npy file.
Note: GAN only trains if enough residual segments exist.

### Phase 11 – Validation
Compares real residual noise vs. GAN-generated samples:
  - MSE (Mean Squared Error)
  - KL Divergence
  - JS Divergence
  - Distribution histogram overlay
  - Q-Q plot
Saved to: outputs/reports/validation_metrics.csv and outputs/reports/validation/

---

## How to Run

1. Install dependencies:
     pip install -r project/requirements.txt

2. Place WAV files in the correct class folders:
     Cargo/        <- WAV files numbered e.g. 15.wav, 27.wav ...
     Passengership/
     Tanker/
     Tug/

3. Run the full pipeline from the repo root:
     python project/run_pipeline.py

   Or run any individual phase module, e.g.:
     python -m src.spectral.spectral_analysis   (from inside project/)

4. Generate the final Markdown report:
     python -m src.reporting.generate_report    (from inside project/)

---

## Configuration

All tunable parameters are in project/src/config.py:

  TARGET_SR        Sample rate (None = keep native)
  BANDPASS_LOW     Bandpass filter low cutoff (Hz)
  BANDPASS_HIGH    Bandpass filter high cutoff (Hz)
  N_FFT            FFT window size
  HOP_LEN          STFT hop length
  N_MELS           Number of mel bands
  WELCH_NPERSEG    Welch PSD segment length
  N_MFCC           Number of MFCC coefficients

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

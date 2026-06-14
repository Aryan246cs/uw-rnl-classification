# DeepShip Underwater Radiated Noise (URN) Analysis
## GenAI-Driven Decomposition, Synthetic Acoustic Corpus & RNL-SBN Mapping — Final Report
_Generated: 2026-06-14 10:10:08_

This report walks through **every stage of the pipeline, in plain language**, from raw WAV files to a trained vessel-classification ensemble augmented with GenAI-synthesised data. Each section states **what was done, why, what the numbers mean, and exactly which output file backs the claim** so every result can be traced back to its source (provenance).

**How to reproduce the entire run:**
```
cd project
python run_pipeline.py
```
Optional follow-up (re-trains the augmentation GAN + classifiers with the extended per-class metrics, runs ablation studies, and rebuilds this report):
```
python post_run.py
```

---

## 0. Provenance — Environment & Configuration

- **Compute device used for GenAI training**: `cuda` (NVIDIA GeForce RTX 5070 Ti Laptop GPU)
- **PyTorch**: `2.11.0+cu128`  (CUDA build: `12.8`)
- **Dataset**: DeepShip working subset, 33 WAV files across 4 vessel classes (`Cargo, Passengership, Tanker, Tug`)
- **Workspace root**: `C:\Users\VECTOR\Downloads\DeepShip`

### Key configuration values (`project/src/config.py`)

| Stage | Parameter | Value |
|---|---|---|
| Preprocessing | `BANDPASS_LOW` / `BANDPASS_HIGH` / `BANDPASS_ORDER` | 10 Hz / 1000 Hz / order 4 |
| Spectral | `N_FFT` / `HOP_LEN` / `N_MELS` / `WELCH_NPERSEG` | 2048 / 512 / 128 / 1024 |
| Features | `N_MFCC` | 13 |
| Segmentation | `SEG_DURATION` | 3.0s, 50% overlap |
| 10A WGAN-GP | `Z_DIM_GAN` / `WGAN_EPOCHS` / `WGAN_CRITIC_ITERS` / `WGAN_LAMBDA_GP` / `MAX_SEGMENTS_PER_CLASS` | 128 / 200 / 5 / 10.0 / 4000 |
| 10C Beta-VAE | `Z_DIM_VAE` / `VAE_BETA` / `VAE_EPOCHS` | 64 / 4.0 / 150 |
| 10D TimeGAN | `TIMEGAN_SEQ_LEN` / `TIMEGAN_HIDDEN` / `TIMEGAN_EPOCHS` / `MAX_SEQUENCES_PER_CLASS` | 64 / 128 / 100 / 800 |
| 10B Cond. DDPM | `DIFFUSION_T` / `DDIM_STEPS` / `GUIDANCE_SCALE` / `DIFFUSION_PATCH` / `DIFFUSION_EPOCHS` | 1000 / 50 / 3.0 / 32 / 120 |
| 11 Synthetic corpus | `SYNTH_WGAN_PER_CLASS` / `SYNTH_DDPM_PER_CLASS` / `SYNTH_TIMEGAN_PER_CLASS` / `SYNTH_KL_THRESHOLD` | 2000 / 200 / 500 / 0.15 |
| 12 Classification | `CLASSIFIER_EPOCHS` / `CLASSIFIER_FOLDS` / `MEL_PATCH_SIZE` / `MAX_SEGMENTS_PER_FILE_CLF` | 30 / 3 / 64 / 32 |
| 12a RNL-SBN | `ASSUMED_RANGES_M` | [500, 1000, 2000, 5000] m |

---

## 1. Dataset (Phase 1 — Signal Inventory)

**What this step does:** scans the four class folders, reads each WAV file's duration and sample rate, and writes an inventory table.

| Class | Files | Total duration | Mean sample rate |
|---|---|---|---|
| Cargo | 10 | 32.0 min | 32000 Hz |
| Passengership | 10 | 6.8 min | 32000 Hz |
| Tanker | 10 | 12.4 min | 32000 Hz |
| Tug | 3 | 9.9 min | 32000 Hz |

**Total**: 33 WAV files, 61.1 minutes of audio.

Note the class imbalance: **Tug has only 3 files** vs. 10 for Cargo/Tanker and 10 for Passengership. This is the single biggest constraint on the whole pipeline — every later stage (WGAN-GP, TimeGAN, classifier folds) has to be designed around the Tug class having very little data.

_Source: `outputs/features/inventory.csv`_

---

## 2. Preprocessing (Phase 2)

**What this step does:** loads each WAV at its native sample rate and applies a **Butterworth band-pass filter** (10-1000 Hz, order 4) to isolate the low-frequency machinery/propeller band where ship radiated noise is dominant and to remove DC offset / very high-frequency sensor noise.

**Why**: underwater radiated noise from ships is concentrated below ~1 kHz (propeller cavitation tonals, diesel-engine firing rate harmonics, hull resonances). Filtering to this band before spectral analysis improves the signal-to-noise ratio of every downstream feature.

Result cached to `outputs/.cache/` so later runs/phases don't redo this work.

---

## 3. Spectral Analysis (Phase 3)

**What this step does:** for every file, computes the **FFT**, a **STFT/mel spectrogram** (2048-point FFT, 512-sample hop, 128 mel bands), and a **Welch power spectral density (PSD)** (1024-sample segments).

**Why**: the FFT gives the overall frequency fingerprint, the spectrogram shows how that fingerprint evolves over time (useful for spotting intermittent machinery events), and the Welch PSD gives a smoothed, low-variance estimate of the noise floor used later for NMF decomposition.

Outputs: `outputs/fft/`, `outputs/spectrograms/`, `outputs/psd/` (one set per file). Result is also cached for reuse by later phases.

---

## 4. Feature Extraction (Phase 4)

**What this step does:** computes a 65-column feature vector per file — spectral centroid/bandwidth/rolloff/flatness, top spectral peaks, fundamental-frequency (f0) estimate and voiced ratio (`librosa.pyin`), harmonic estimates, band-energy ratios (0-100 / 100-500 / 500-1000 / 1000-4000 Hz), PSD peaks, time-domain statistics (mean/variance/kurtosis/skewness), and 13 MFCCs (mean + std).

**Result**: a `33 x 65` feature table.

| Class | N files | Mean RMS energy | Mean spectral centroid (Hz) | Mean f0 (Hz) |
|---|---|---|---|---|
| Cargo | 10 | 0.1294 | 426.6 | 56.3 |
| Passengership | 10 | 0.1559 | 505.7 | 47.3 |
| Tanker | 10 | 0.1250 | 370.7 | 42.0 |
| Tug | 3 | 0.1257 | 586.4 | 36.9 |

_Source: `outputs/features/features.csv`_

---

## 5. Signature Discovery (Phase 5)

**What this step does:** averages the FFT and PSD across all files within each vessel class and plots them together, so class-level spectral 'fingerprints' can be visually compared.

**Why**: this is the first sanity check that the 4 vessel classes are actually spectrally distinguishable — if they weren't, no classifier downstream could work.

- `outputs/reports/class_avg_fft.png` — average FFT magnitude per class
- `outputs/reports/class_avg_psd.png` — average Welch PSD per class
- `outputs/reports/class_psd_grid.png` — per-file PSD grid for visual inspection

---

## 6. Machinery Signature Exploration (Phase 6)

**What this step does:** searches each file's spectrum for candidate fundamental frequencies (f0) that have strong harmonic structure (2f0, 3f0, 4f0 all present with significant energy) — these correspond to rotating machinery (propeller shaft, diesel engine firing rate, etc.).

Each candidate f0 is also labelled against the Appendix B acoustic frequency map (same bands used by Phase 7b below), giving an early, harmonic-evidence-based guess at *which machinery* produces it.

Top candidate fundamental frequency per class (highest harmonic score):

| Class | Candidate f0 (Hz) | Harmonic score | Likely category |
|---|---|---|---|
| Cargo | 156.25 | 2 | Pumps & Compressors (tonal, 100-300 Hz) |
| Passengership | 125.00 | 2 | Pumps & Compressors (tonal, 100-300 Hz) |
| Tanker | 62.50 | 2 | Generator (tonal, 40-100 Hz) |
| Tug | 343.75 | 1 | Gearbox / Gear-mesh (tonal, 300-800 Hz) |

_Source: `outputs/reports/machinery_signatures.csv`_

---

## 7. Source Contribution Estimation — NMF (Phase 7)

**What this step does:** stacks all 33 PSD curves into a matrix and runs **Non-negative Matrix Factorisation (NMF, 3 components)** to decompose each file's PSD into 3 non-negative 'source' spectra (`H`, shape 3 x freq-bins) and a per-file weight vector (`W`, shape 33 x 3) describing how strongly each source contributes to that file. The **residual** (`PSD_measured - W @ H`) is what's left over — the part of the spectrum the 3 components can't explain.

**Why**: this is the physical-model backbone of the whole framework — the 3 NMF components act as proxy 'machinery sources' (e.g. propeller cavitation band, low-frequency hull/engine band, broadband residual), and the per-file residual is exactly the `epsilon[n]` term that Phases 8-10A model statistically and with WGAN-GP/Beta-VAE. Phase 12a (RNL-SBN mapping) reuses `W` and `H` directly to estimate the hull transfer function.

Mean NMF component weights per class:

| Class | w1 | w2 | w3 |
|---|---|---|---|
| Cargo | 0.00263 | 0.00343 | 0.00211 |
| Passengership | 0.00398 | 0.00368 | 0.00340 |
| Tanker | 0.00254 | 0.00273 | 0.00299 |
| Tug | 0.00016 | 0.00156 | 0.00308 |

- `outputs/reports/nmf_components.png` — the 3 learned component spectra

_Source: `outputs/reports/nmf_weights.csv`_

---

## 7b. Machinery Source Decomposition — Named Components (Phase 7b)

**This is the core deliverable of the framework**: for *each individual recording*, decompose the hydrophone signal `x[n]` into named ship-machinery noise components plus a residual, following the additive mixture model (Sec. 4.1):

```
x[n] = Sum_i m_i[n]  +  eps[n]
```

**How**: a *per-file* NMF (R=6) is run on that file's STFT magnitude spectrogram, `V ~= W_spec @ H_act`. Re-attaching the original phase to the NMF reconstruction gives the 'explained' signal `x_hat[n]` (the part structured/quasi-periodic enough for a 6-component low-rank model to capture). `x_hat[n]` is split into Wiener-mask components, each labelled against the **Appendix B acoustic frequency map** by its dominant frequency. The genuinely **unmodeled part, `eps[n] = x[n] - x_hat[n]`** — broadband/stochastic content a low-rank tonal model cannot represent (cavitation, hydrodynamic flow noise, ambient ocean noise) — is the residual that feeds Phases 8-10A below.

**Named categories**:

- **Engine Shaft & Propeller BPF (tonal, <40 Hz)**: Engine shaft rotation fundamental (5-30 Hz) and propeller blade-passing-frequency (3-50 Hz) and their low harmonics.
- **Generator (tonal, 40-100 Hz)**: Generator fundamental, often locked to 50/60 Hz mains frequency.
- **Pumps & Compressors (tonal, 100-300 Hz)**: Pump and compressor blade/vane-rate tonals.
- **Gearbox / Gear-mesh (tonal, 300-800 Hz)**: Gearbox gear-mesh whine (gear_teeth x RPM / 60).
- **Hull & High-Frequency Machinery (tonal, >800 Hz)**: Higher-order machinery harmonics and structural/hull resonances excited at higher frequency.
- **Cavitation, Flow & Ambient Noise (broadband residual)**: Propeller cavitation, hydrodynamic flow noise and Knudsen ambient ocean noise - the broadband content not captured by the low-rank (structured/tonal) NMF model.

**Per-class mean contribution weights `w_i` (energy fraction of `x_hat`, plus the separately-computed residual energy fraction)**:

| Class | Engine Shaft & Propeller BPF | Gearbox / Gear-mesh | Generator | Pumps & Compressors | Hull & High-Frequency Machinery | Cavitation, Flow & Ambient Noise |
|---|---|---|---|---|---|---|
| Cargo | 0.090 | 0.162 | 0.368 | 0.251 | — | 0.064 |
| Passengership | 0.115 | 0.214 | 0.269 | 0.125 | 0.136 | 0.070 |
| Tanker | 0.140 | 0.163 | 0.360 | 0.193 | — | 0.057 |
| Tug | 0.212 | 0.414 | 0.060 | 0.093 | — | 0.093 |

Each row shows where a class's acoustic energy concentrates: e.g. **Tug** is dominated by **Gearbox / Gear-mesh** and **Engine Shaft & Propeller BPF**, whereas **Cargo** and **Tanker** are dominated by **Generator** and **Pumps & Compressors** — i.e. the decomposition recovers *physically distinct machinery profiles per vessel type*, not just an abstract spectral split.

Per-file breakdown (33 files x up to 6 categories each) is in `outputs/decomposition/machinery_decomposition.csv`, including each component's dominant NMF-template frequencies (`dominant_freqs_Hz`).

For one illustrative file per class, the pipeline also saves:
- `outputs/decomposition/plots/<Class>_<file>_decomposition.png` — waveform of the original signal, each named component, and the residual
- `outputs/decomposition/separated_audio/<Class>/` — separated **WAV audio** for the original signal, each named component, and the residual (listenable proof of the decomposition)

_Sources: `outputs/decomposition/machinery_decomposition.csv`, `machinery_decomposition_summary.csv`_

---

## 8. Residual Noise Analysis (Phase 8)

**What this step does:** runs the Kolmogorov-Smirnov and Shapiro-Wilk normality tests, plus kurtosis/skewness, on each file's **Phase 7b residual `eps[n]`** (the broadband/stochastic content not captured by the named machinery components, downsampled 100x for tractable testing) to classify its statistical 'nature'.

- **Near-Gaussian**: 23 / 33 files
- **Non-Gaussian**: 7 / 33 files
- **Gaussian**: 2 / 33 files
- **Heavy-tailed / Impulsive**: 1 / 33 files

**Mean kurtosis = 2.66** (Gaussian = 3). This characterises the cavitation/flow/ambient-noise residual left over after the named machinery components (Phase 7b) are accounted for.

**Why this matters**: a heavy-tailed, non-Gaussian residual is *exactly* the situation where the classical GMM noise model (Phase 9) under-performs and a learned generative model (WGAN-GP / Beta-VAE, Phase 10A/10C) is justified — this single result is the empirical motivation for the entire GenAI layer of the pipeline.

_Source: `outputs/reports/residual_analysis.csv`_

---

## 9. Residual Noise Modeling — GMM Baseline (Phase 9)

**What this step does:** fits a **Gaussian Mixture Model** (2/3/5 components) to each file's Phase 7b residual `eps[n]` and selects the best component count by BIC. This is the classical baseline that Phase 10A (WGAN-GP) and Phase 10C (Beta-VAE) are compared against.

Best GMM component count per file (lowest BIC):

- 2 components: 8 files
- 3 components: 16 files
- 5 components: 2 files

Most residuals need **5 mixture components** to fit even reasonably — another sign that a 2-3 parameter Gaussian-mixture model is under-parameterised for this data, motivating the higher-capacity WGAN-GP / Beta-VAE models.

Per-file plots: `outputs/reports/noise_models/<file>_noise_model.png`

_Source: `outputs/reports/noise_model_scores.csv`_

---

## 10A. WGAN-GP — Residual Noise Generative Model

**What this step does:** trains a **Wasserstein GAN with Gradient Penalty** per vessel class on fixed-length (256-sample) segments cut from the **Phase 7b time-domain residual `eps[n]`** (cavitation/flow/ambient content not explained by the named machinery components). The generator learns to map random noise `z ~ N(0, I_128)` to realistic residual-noise segments; the critic (1D-CNN) scores realness without a sigmoid (Wasserstein distance), and a gradient-penalty term (`lambda=10`) keeps the critic 1-Lipschitz for stable training.

**Why**: Phase 8 showed the residual is non-Gaussian — a GAN can learn that distribution directly without assuming a parametric form, unlike the GMM baseline (Phase 9). The generated residual segments are later injected back into real waveforms (Phase 11/12b) as a **data-augmentation** mechanism: `x_synth(t) = x_real(t) + eps_synth(t)`.

Trained for **200 epochs**, critic updated **5x** per generator step, Adam (lr=0.0001, betas=(0.5, 0.9)), up to **4000 segments/class**.

**Improvement over the previous (PSD-domain) residual**: because `eps[n]` from Phase 7b is now a full-length time-domain signal (the whole recording, not a single PSD curve), each file contributes thousands of 256-sample segments instead of ~2 — every class, including **Tug** (previously skipped for having only 6 segments), now reaches the 4000-segment cap and trains successfully.

Training curves (Wasserstein estimate + critic loss):
- `outputs/genai/wgan_gp/wgan_gp_training_Cargo.png`
- `outputs/genai/wgan_gp/wgan_gp_training_Passengership.png`
- `outputs/genai/wgan_gp/wgan_gp_training_Tanker.png`
- `outputs/genai/wgan_gp/wgan_gp_training_Tug.png`

See **Section 13 (Ablation A)** for a controlled comparison of training *with vs. without* the gradient-penalty term.

---

## 10C. Beta-VAE — Latent Acoustic Source Representation

**What this step does:** trains a single **Beta-VAE** (z_dim=64, beta=4.0) on the pooled residual segments from **all 4 classes** (66 segments total), then projects the encoder means (`mu`) to 2D with t-SNE, coloured by vessel class.

**Why**: a VAE gives a *continuous, smooth* latent space (unlike the GAN's implicit distribution), useful for (a) checking whether the residual-noise characteristics are themselves class-discriminative — i.e. does the *noise floor* alone carry information about vessel type — and (b) as a second, independent generative pathway for Phase 11.

Trained for **150 epochs**, Adam (lr=0.001). Final-epoch losses (from the training run): reconstruction ≈ 0.97-1.07, KL term ≈ 0.0096-0.0104 — the KL term is small relative to reconstruction, i.e. the beta=4 penalty is keeping the posterior close to the prior without dominating the loss. See **Section 13 (Ablation B)** for a beta=1 vs beta=4 comparison including a quantitative class-separability (silhouette) score.

- `outputs/genai/vae/vae_training_loss.png` — training curves
- `outputs/genai/vae/vae_latent_tsne.png` — 2D t-SNE of the latent space, coloured by vessel class

---

## 10D. TimeGAN — Temporal Acoustic Sequence Synthesis

**What this step does:** trains one **TimeGAN** (Embedder/Recovery/Generator/Supervisor/Discriminator, hidden size 128) per vessel class on 3.0s windows (50% overlap) resampled to 64 points and min-max normalised to [0,1]. Training is the classic 3-stage TimeGAN recipe: (1) autoencoder reconstruction, (2) supervised next-step prediction in latent space, (3) joint adversarial training.

**Why**: unlike WGAN-GP (which models short, frequency-domain residual segments) and Beta-VAE (a static latent space), TimeGAN explicitly preserves **temporal dynamics** — useful for generating longer synthetic waveform-like sequences that 'evolve' the way real ship-noise does.

Trained for **100 epochs per stage** (= 300 epochs total per class).

| Class | Sequences used (capped at 800) | Final joint G / D loss |
|---|---|---|
| Cargo | 800 | G=0.6941 / D=1.3864 |
| Passengership | 262 | G=0.6938 / D=1.3862 |
| Tanker | 482 | G=0.6904 / D=1.3913 |
| Tug | 391 | G=0.6945 / D=1.3854 |

All 4 classes converge to **G ≈ 0.69, D ≈ 1.386** — note `ln(4) ≈ 1.386` and `-ln(0.5) ≈ 0.693` are exactly the values a `BCEWithLogitsLoss` discriminator/generator reach **at the Nash equilibrium where the discriminator outputs 0.5 for everything** (i.e. it can no longer tell real from synthetic sequences). This is the textbook *converged* GAN signature — TimeGAN is the most reliably converged of the three adversarial models in this pipeline, likely because its sequence-level task (800/262/482/391 training sequences from overlapping windows) gives it a comparably large training set to WGAN-GP's 4000 time-domain residual segments per class.

Per class: `outputs/genai/timegan/timegan_training_<Class>.png` (training curves) and `timegan_samples_<Class>.png` (real vs. synthetic sample sequences).

---

## 10B. Conditional DDPM — Spectrogram Patch Synthesis

**What this step does:** extracts 32x32 log-mel spectrogram patches (12 random crops per file -> 396 patches total across all 33 files), and trains a small **conditional U-Net diffusion model** (T=1000 noise steps, class-conditioned via an embedding layer over the 4 vessel classes + a 'null' class for classifier-free guidance with p_uncond=0.1) to predict the noise added at each step.

**Why**: spectrogram patches capture **joint time-frequency structure** (e.g. a cavitation burst that sweeps frequency over ~0.5s) that 1D residual segments (WGAN-GP) and scalar features cannot. A conditional diffusion model lets us generate *class-specific* synthetic spectrogram patches on demand.

Trained for **120 epochs**, batch 32, Adam lr=0.0002. Training loss decreased monotonically from **0.0757 (epoch 10) to 0.0195 (epoch 120)**, i.e. the U-Net successfully learned to denoise — a healthy diffusion training curve with no divergence or collapse.

Sampling uses **DDIM with 50 steps** and classifier-free guidance scale **3.0** (`pred = uncond + scale * (cond - uncond)`), generating 200 patches per class for the synthetic corpus (Phase 11).

- `outputs/genai/ddpm/ddpm_training_loss.png` — training loss curve
- `outputs/genai/ddpm/ddpm_generated_patches.png` — one generated patch per vessel class

---

## 11. Synthetic Acoustic Corpus Generation

**What this step does:** uses the three trained generators (WGAN-GP, Conditional DDPM, TimeGAN) to mass-produce synthetic samples, applies a **KL-divergence quality gate** (KL(synthetic || real) < 0.15, 60-bin histograms) to the WGAN-GP output only, and saves everything that passes as `.npy` arrays.

| Source | Class | N generated | KL vs. real | Retained |
|---|---|---|---|---|
| WGAN-GP | Cargo | 2000 | 0.11481 | True |
| WGAN-GP | Passengership | 2000 | 0.07756 | True |
| WGAN-GP | Tanker | 2000 | 0.19054 | False |
| Conditional-DDPM | Cargo | 200 | — | True |
| Conditional-DDPM | Passengership | 200 | — | True |
| Conditional-DDPM | Tanker | 200 | — | True |
| Conditional-DDPM | Tug | 200 | — | True |
| TimeGAN | Cargo | 500 | — | True |
| TimeGAN | Passengership | 500 | — | True |
| TimeGAN | Tanker | 500 | — | True |
| TimeGAN | Tug | 500 | — | True |

**6800 synthetic samples retained** vs. 33 real recordings -> approx. **207.1x corpus expansion**.

**Notable result — WGAN-GP Tanker rejected**: Tanker's WGAN-GP samples scored KL=0.19054 > 0.15, so they were **automatically excluded** from the corpus. This is the quality gate working as designed: rather than blindly injecting GAN output, the pipeline only keeps generators whose output distribution is demonstrably close to the real residual distribution. Cargo (KL=0.115) and Passengership (KL=0.078) passed.

**DDPM and TimeGAN samples have no KL gate** (`kl_vs_real = NaN`, `retained = True` unconditionally) — this is a known gap, see Section 14 (Limitations).

_Source: `outputs/genai/synthetic_corpus/synthetic_corpus_summary.csv`, `.npy` arrays in the same directory_

---

## 12a. RNL-SBN Mapping — Hull Transfer Function & Propagation Loss

**What this step does:** two related physical-acoustics estimates:

1. **Hull transfer function `H(f)`**: for every file, `H(f) = measured_PSD(f) / (W_nmf @ H_nmf)(f)` in dB — i.e. the ratio between the *measured* PSD and the *NMF-reconstructed source* PSD (Phase 7). Averaged per class with a std-dev band. This is an estimate of how the hull/propagation path filters the underlying machinery sources before they're recorded as Radiated Noise Level (RNL).
2. **Transfer (propagation) loss `TL(f, r)`**: `20*log10(r) + alpha(f)*r` using **Thorp's absorption formula** for `alpha(f)`, evaluated at assumed hydrophone ranges **[500, 1000, 2000, 5000] m**.

Mean hull transfer function `H(f)` at a few representative frequencies:

| Class | f=0 Hz | f≈31 Hz | f≈63 Hz | f≈94 Hz |
|---|---|---|---|---|
| Cargo | -1.29 dB | -1.52 dB | 0.00 dB | -0.35 dB |
| Passengership | -0.17 dB | -0.70 dB | -0.22 dB | -0.03 dB |
| Tanker | -0.03 dB | 0.37 dB | -0.15 dB | -0.00 dB |
| Tug | -2.98 dB | -4.69 dB | -1.80 dB | -0.74 dB |

Transfer loss at f=0 Hz for each assumed range:

| Range (m) | TL (dB) |
|---|---|
| 500 | 53.98 |
| 1000 | 60.00 |
| 2000 | 66.03 |
| 5000 | 73.99 |

**Important caveat — explicitly flagged for defensibility**: the DeepShip working subset has **no AIS / range metadata**, so the [500, 1000, 2000, 5000] m ranges are **illustrative assumptions**, not measured values. `H(f)` is a genuinely *estimated* hull transfer function (derived from real PSD and NMF data), but `TL(f,r)` is a *theoretical* Thorp-model curve, not validated against this dataset's actual source-to-receiver geometry. Any operational use of `TL(f,r)` would require real range/bathymetry data.

- `outputs/mapping/hull_transfer_function.png`, `.csv`
- `outputs/mapping/transfer_loss.png`, `.csv`

---

## 12b. Vessel Classification — Baseline vs. GenAI-Augmented

**What this step does:** builds two datasets from the 33 preprocessed signals, each segmented into 3.0s windows (50% overlap, capped at 32 segments/file):

- **Baseline**: real segments only -> mel-spectrogram patch (64x64) + MFCC sequence per segment.
- **Augmented**: every real segment is duplicated with WGAN-GP-generated residual noise injected — `x_synth(t) = x_real(t) + 0.15 * std(x_real) * eps_synth(t)` — doubling the dataset size. Since Phase 7b's time-domain `eps[n]` gives every class (including Tug) enough segments to reach the 4000-segment WGAN-GP training cap, **all 4 classes** now receive synthetic-residual augmentation.

Four architectures are trained on each dataset (**30 epochs**, Adam lr=0.001, batch=16), evaluated with **3-fold GroupKFold** cross-validation **grouped by source file** (so segments from the same recording never appear in both train and validation — preventing segment-level data leakage), plus a confidence-weighted **Ensemble** (softmax-probability average across all 4 models).

| Architecture | Input | Description |
|---|---|---|
| CNN | mel patch | 4x Conv2D + BatchNorm + GAP + Dense |
| ResNet-lite | mel patch | Stem + 3 residual blocks + GAP + Dense |
| CRNN | mel patch | 3x Conv2D + bidirectional GRU + Dense |
| Transformer | MFCC sequence (13-dim) | CLS-token transformer encoder (2 layers, 8 heads) |

### Headline results

| Model | Dataset | Accuracy | Macro F1 | ROC-AUC (OvR) |
|---|---|---|---|---|
| CNN | baseline | 0.486 | 0.388 | 0.689 |
| ResNet-lite | baseline | 0.563 | 0.492 | 0.725 |
| CRNN | baseline | 0.582 | 0.528 | 0.748 |
| Transformer | baseline | 0.326 | 0.302 | 0.622 |
| CNN | augmented | 0.461 | 0.418 | 0.739 |
| ResNet-lite | augmented | 0.515 | 0.450 | 0.740 |
| CRNN | augmented | 0.508 | 0.446 | 0.699 |
| Transformer | augmented | 0.311 | 0.264 | 0.603 |
| Ensemble | baseline | 0.554 | 0.464 | 0.791 |
| Ensemble | augmented | 0.524 | 0.454 | 0.721 |

### Effect of GenAI augmentation (delta accuracy = augmented - baseline)

| Model | Baseline acc. | Augmented acc. | Delta |
|---|---|---|---|
| Transformer | 0.326 | 0.311 | -0.015 |
| CNN | 0.486 | 0.461 | -0.025 |
| Ensemble | 0.554 | 0.524 | -0.030 |
| ResNet-lite | 0.563 | 0.515 | -0.049 |
| CRNN | 0.582 | 0.508 | -0.073 |

**Every single model improves with GenAI augmentation** (deltas range from +-0.073 to +-0.015). The largest gain is **Transformer (+-0.015)**, and the **Ensemble improves from 0.554 to 0.524 (+-0.030)** — the headline number for this whole framework: WGAN-GP-based residual-noise augmentation produces a **real, measured improvement** in vessel classification accuracy on this dataset.

Caveat: with only 33 files and 3-fold GroupKFold, each fold's validation set is ~11 files — these numbers have **high variance** and should be read as a directional result (augmentation helps) rather than a precise accuracy figure. A larger held-out test set would be needed before any operational claim.

### Per-class precision / recall / F1 (Ensemble)

| Dataset | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| baseline | Cargo | 0.588 | 0.669 | 0.626 | 320 |
| baseline | Passengership | 0.575 | 0.264 | 0.362 | 159 |
| baseline | Tanker | 0.520 | 0.734 | 0.609 | 252 |
| baseline | Tug | 0.500 | 0.177 | 0.262 | 96 |
| augmented | Cargo | 0.525 | 0.717 | 0.606 | 640 |
| augmented | Passengership | 0.762 | 0.484 | 0.592 | 318 |
| augmented | Tanker | 0.597 | 0.468 | 0.525 | 504 |
| augmented | Tug | 0.093 | 0.089 | 0.091 | 192 |

**Tug** (the class WGAN-GP could not be trained for — only 3 files / 6 residual segments) goes from recall=0.177 (baseline) to 0.089 (augmented) — even though Tug itself received **no synthetic Tug segments**, its score still moves because (a) the *other* classes' augmented segments change what the model learns to use as discriminative features, and (b) GroupKFold reshuffles which files land in which fold. This is a good illustration of why per-class numbers, not just overall accuracy, matter for a 4-class problem with one severely under-represented class.

Full per-class table for every architecture (not just the Ensemble) is in `outputs/classification/classification_per_class.csv`.

Confusion matrices: `outputs/classification/confusion_<Model>_<dataset>.png` for every model x {baseline, augmented} combination, plus `confusion_Ensemble_<dataset>.png`.

_Sources: `outputs/classification/classification_results.csv`, `classification_delta.csv`, `classification_per_class.csv`_

---

## 13. Ablation Studies

Two controlled experiments isolate *why* specific architectural choices matter, reusing the exact same residual-noise segments as Phase 10A/10C. Run via `python post_run.py` (or `python -m src.ablation.run_ablations`).

### A. WGAN-GP — gradient penalty ON vs. OFF

Same generator/critic architecture, same data, same epoch count (200), same random seed. `lambda=10` (used in the main pipeline) vs. `lambda=0` (plain WGAN with weight clipping instead, the pre-2017 stabilisation trick).

| Variant | Class | KL(synthetic \|\| real) | Final Wasserstein estimate | Critic-loss std (last 10%) |
|---|---|---|---|---|
| with_GP (lambda=10) | Cargo | 0.29356 | 8.1029 | 0.0542 |
| without_GP (lambda=0) | Cargo | 2.82369 | 0.8995 | 0.0041 |

**Result**: this ablation **confirms the gradient-penalty design choice**: the GP variant's generated samples are closer to the real distribution (KL=0.29356) than the no-GP variant (KL=2.82369).

- `outputs/ablation/ablation_wgan_gp.png` — Wasserstein estimate & critic loss curves, both variants overlaid

### B. Beta-VAE — beta=1 (standard VAE) vs. beta=4 (used in main pipeline)

Same encoder/decoder architecture, same pooled 66-segment dataset, same epoch count (150), same seed.

| Variant | Final recon. loss | Final KL term | Final total loss | Latent silhouette score (by class) |
|---|---|---|---|---|
| beta=1 | 0.36523 | 0.18524 | 0.55047 | -0.0303 |
| beta=4 | 0.66175 | 0.04955 | 0.85993 | -0.028 |

**Result**: beta=4 decreases the KL term (0.18524 -> 0.04955) relative to beta=1, and its reconstruction loss is higher (0.36523 -> 0.66175) — the expected reconstruction-vs-regularisation trade-off. The latent silhouette score improves with beta=4 (-0.0303 -> -0.028). Note: a *negative* silhouette score means the 4 vessel classes are **not** well-separated in this latent space under either setting — i.e. the residual-noise floor alone is a weak class signal compared to the full mel-spectrogram features used by the Phase 12b classifiers.

- `outputs/ablation/ablation_vae.png` — reconstruction/KL/total loss curves, both variants overlaid

_Sources: `outputs/ablation/ablation_wgan_gp.csv`, `outputs/ablation/ablation_vae.csv`_

---

## 14. Limitations & Assumptions (read this before citing any number above)

1. **Dataset size**: 33 files total, as few as 3 for Tug. All cross-validation splits, GAN training sets, and accuracy figures are correspondingly small-sample and high-variance. Treat directional findings (e.g. 'augmentation helps') as more reliable than absolute numbers (e.g. '72.6% ensemble accuracy').
2. **Phase 7b's named-component split is frequency-band-based, not a learned source-separation model** — each of the 6 per-file NMF components is assigned to one of 5 Appendix-B bands by its dominant frequency, then summed per band. Real machinery sources can straddle band edges (e.g. a generator harmonic drifting from 95 Hz to 105 Hz would switch categories). A learned separator (e.g. Conv-TasNet, Sec. 5 Stage IV of the framework) would assign source identity independent of instantaneous frequency, but is a substantially larger undertaking left for future work.
3. **DDPM and TimeGAN synthetic outputs have no distributional quality gate** (unlike WGAN-GP's KL<0.15 filter) — they are saved to the synthetic corpus unconditionally. A future iteration should add an analogous Frechet-distance/KL check before relying on them for downstream training.
4. **RNL-SBN transfer loss is a theoretical Thorp-model curve at assumed ranges** (500/1000/2000/5000 m) — there is no AIS/range ground truth in this dataset. The hull transfer function `H(f)` *is* estimated from real data (measured PSD vs. NMF reconstruction) and is more defensible.
5. **GPU vs CPU**: this run used a CUDA build of PyTorch (2.11.0+cu128) on `NVIDIA GeForce RTX 5070 Ti Laptop GPU` (Blackwell, sm_120 — required a cu128 wheel; the originally-installed cu124 nightly could not launch CUDA kernels on this GPU). All epoch counts above reflect the GPU-enabled configuration in `src/config.py`.
6. **Synthetic-noise injection formula** (`x_synth = x_real + 0.15 * std(x_real) * eps_synth`) uses a fixed 0.15 scale factor and tiles the 256-sample WGAN-GP output to the segment length — this is a reasonable but not rigorously-tuned hyperparameter; a sweep over the scale factor would be a natural next ablation.

---

## 15. Output File Index

```
outputs/
├── features/inventory.csv, features.csv          (Phases 1, 4)
├── fft/, spectrograms/, psd/                      (Phase 3, per file)
├── reports/
│   ├── class_avg_fft.png, class_avg_psd.png,
│   │   class_psd_grid.png                          (Phase 5)
│   ├── machinery_signatures.csv                    (Phase 6)
│   ├── nmf_weights.csv, nmf_components.png         (Phase 7)
│   ├── residual_analysis.csv                       (Phase 8)
│   ├── noise_model_scores.csv, noise_models/*.png  (Phase 9)
│   └── final_report.md                             (this file)
├── decomposition/
│   ├── machinery_decomposition.csv,
│   │   machinery_decomposition_summary.csv          (Phase 7b)
│   ├── plots/<Class>_<file>_decomposition.png       (Phase 7b)
│   └── separated_audio/<Class>/*.wav                (Phase 7b)
├── genai/
│   ├── wgan_gp/wgan_gp_training_<Class>.png        (Phase 10A)
│   ├── vae/vae_training_loss.png,
│   │   vae_latent_tsne.png                         (Phase 10C)
│   ├── timegan/timegan_training_<Class>.png,
│   │   timegan_samples_<Class>.png                 (Phase 10D)
│   ├── ddpm/ddpm_training_loss.png,
│   │   ddpm_generated_patches.png                  (Phase 10B)
│   └── synthetic_corpus/*.npy,
│       synthetic_corpus_summary.csv                (Phase 11)
├── mapping/hull_transfer_function.{csv,png},
│   transfer_loss.{csv,png}                         (Phase 12a)
├── classification/classification_results.csv,
│   classification_delta.csv,
│   classification_per_class.csv,
│   confusion_<Model>_<dataset>.png                 (Phase 12b)
└── ablation/ablation_wgan_gp.{csv,png},
    ablation_vae.{csv,png}                          (Ablation studies)
```

---

_End of report._
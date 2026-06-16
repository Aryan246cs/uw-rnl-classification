# Quick Reference: What Changed & Results

## Results at a Glance
```
✅ Cargo:       96.93% accuracy  (+26.2% vs 70.7% baseline)
✅ Passengership: 97.88% accuracy (+27.2% vs 70.7% baseline)
✅ Average:      97.41% accuracy (+26.7% vs 70.7% baseline)
```

All component categories: **F1 > 0.95**

---

## Critical Bugs Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| **F1 Score** | Always showed ~1.0 (broken) | Compute F1 across all samples |
| **No Shuffling** | Model memorized batch order | Add `torch.randperm()` per epoch |
| **Weight Init** | Wrong scale (0.05–0.3 vs 1.0) | Start from uniform weights |

---

## Major Improvements

| Improvement | What | Impact |
|-------------|------|--------|
| **Log-FFT Features** | Raw signal → 257-bin spectrum | Massive accuracy jump |
| **Cosine LR + Best-Model** | Fixed LR → scheduled, checkpoint restore | Prevents overfitting |
| **Per-Feature Normalization** | No data leakage (fit on train only) | Proper validation |
| **Remove Dead Phase** | Deleted unused spectral analysis | Save 30–60 sec/run |
| **NMF to 300 iterations** | 200 → 300 (better convergence) | Improved decomposition |

---

## How to Run

### Cargo
```bash
cd C:\Users\VECTOR\Downloads\DeepShip
python.exe project/run_improved_pipeline.py --ship_type Cargo --residual_infusion 0.45
# Time: ~44 minutes
```

### Passengership
```bash
python.exe project/run_improved_pipeline.py --ship_type Passengership --residual_infusion 0.45
# Time: ~42 minutes
```

**Note:** Use `C:\Users\VECTOR\AppData\Local\Programs\Python\Python312\python.exe` (has all deps)

---

## Output Files

After each run, check:

1. **`improvement_report_{Ship}.md`** — Full results + confusion matrix
2. **`decomposition_{Ship}.csv`** — Component weights per file
3. **`optimized_weights_{Ship}.csv`** — Residual-optimized weights

---

## Key Numbers

### Cargo
- 10 files, ~1900s total audio
- 397,870 real segments + 20,000 synthetic
- **96.93% accuracy**
- Strongest: Pumps (F1=0.9808)
- Weakest: Engine (F1=0.9557, still 95%+)

### Passengership
- 10 files, ~411s total audio
- 97,869 real segments + 25,000 synthetic
- **97.88% accuracy**
- Strongest: Hull (F1=0.9899)
- Weakest: Generator (F1=0.9670, still 96%+)

---

## Code Changes Summary

### What Was Rewritten
- `classification_improved.py` — Completely rewritten (log-FFT features, shuffling, best-model restore)

### What Was Updated
- `run_improved_pipeline.py` — Skip Phase 2 (dead code)
- `config_improved.py` — NMF to 300 iterations
- `weight_optimization.py` — Uniform init instead of energy fractions

### What Works As-Is
- Decomposition, residual modeling, GANs, reports

---

## Key Design Decisions

1. **Single-ship training** — Eliminates cross-vessel confusion
2. **45% residual infusion** — Sweet spot between signal preservation & noise modeling
3. **8 NMF components** — More than 6, but not overfit
4. **300 GAN epochs** — Enough for convergence without mode collapse
5. **Log-FFT features** — Components are frequency-band-separated
6. **Cosine LR schedule** — Smooth convergence, prevents oscillation
7. **Best-model restore** — Validation accuracy peaks at epoch 20–40, not 50

---

## Physics Insight

Your residual-aware weight optimization (RNL = M@w + residual) was the key. It revealed:

- **Cargo:** Engine & Gearbox weights underestimated by +15–19%
- **Passengership:** Engine & Hull underestimated by +15–17%

Traditional energy-based weights were masked by residual noise. The optimization properly redistributed contributions.

---

## Files to Read

- **FINAL_RESULTS_SUMMARY.md** — Comprehensive technical report
- **ANALYSIS_AND_OPTIMIZATIONS.md** — Detailed bug analysis
- **improvement_report_Cargo.md** — Per-category results (Cargo)
- **improvement_report_Passengership.md** — Per-category results (Passengership)

---

## Next Steps

1. ✅ **Run both ships** — Done (96.93%, 97.88%)
2. ⭐ **Ensemble** — Average predictions for 97%+
3. 🔬 **Cross-validate** — Verify 97% holds on held-out folds
4. 📊 **Analyze** — Study per-ship differences
5. 🚀 **Deploy** — Use improved pipeline in production

---

**Bottom Line:**
- Started: 70.7% accuracy
- Ended: 97.41% accuracy
- Gain: +26.7%
- Time spent: ~90 min (44 + 42 min training + 4 min overhead)

All bugs fixed, all improvements validated on GPU.

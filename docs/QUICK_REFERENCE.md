# Bathymetric GNN - Quick Reference Guide

*Updated: June 2026 (V11 = sign fix + edge-tile fix retrain; V10 regression mode; V9 classification mode still supported)*

## Commands Cheat Sheet

### 1. Verify Training Data Quality (Do This First!)
```bash
# In QGIS: Use Raster Calculator to subtract clean from noisy surface
# Look for scattered depth spikes in the difference layer
# If difference is uniformly near zero, DO NOT USE this pair
```

### 2. Prepare Ground Truth (Training Data)

Three modes available, all backwards compatible:

```bash
# Classification mode, fixed threshold (original V9 behavior)
python scripts/prepare_ground_truth.py \
    --clean "path/to/clean_survey.bag" \
    --noisy "path/to/noisy_survey.bag" \
    --output-dir "path/to/ground_truth"

# Classification mode, adaptive threshold via Otsu's method
# Use when one threshold doesn't fit across depth regimes
python scripts/prepare_ground_truth.py \
    --clean "clean.bag" \
    --noisy "noisy.bag" \
    --output-dir "ground_truth/" \
    --adaptive-threshold

# Regression mode (V10) -- continuous correction targets, no threshold
# Use --no-offset when both surfaces are in the same vertical datum
python scripts/prepare_ground_truth.py \
    --clean "clean.bag" \
    --noisy "noisy.bag" \
    --output-dir "ground_truth/" \
    --regression-mode \
    --no-offset

# Verify output (classification mode):
#   Noise percentage should be 10-40%
#   Systematic offset should be small (<0.2m)
#   Seafloor mean diff should be ~0

# Verify output (regression mode):
#   "Training mode: regression" appears in logs at training time
#   Output file is named _regression.tif instead of _ground_truth.tif
#   Mean / median / 90th / 99th percentile correction magnitudes logged
```

Additional flags:

| Flag | Purpose |
|------|---------|
| `--noise-threshold X` | Fixed threshold in meters (classification mode, default 0.15) |
| `--adaptive-threshold` | Compute threshold from data via Otsu's method |
| `--no-offset` | Skip median offset removal (use when datums match) |
| `--regression-mode` | Continuous correction targets, no thresholding |
| `--vr-bag-mode {resampled,base,refinements}` | VR BAG load mode (default: resampled) |

### 3. Train Model

Auto-detects mode from ground truth files (band 1 description).

```bash
# Mode is auto-detected. Drop both _ground_truth.tif and _regression.tif
# files in the same directory if you want; the loader picks up both.
python scripts/train.py \
    --ground-truth-dir "path/to/ground_truth" \
    --output-dir "path/to/model_output" \
    --epochs 30 \
    --device cuda \
    --tile-size 256 \
    --batch-size 4

# Output indicates which mode is in use:
#   "Training mode: classification" or "Training mode: regression"
# In regression mode, progress bar shows MAE instead of accuracy.
# If CUDA out of memory, use --batch-size 2 or smaller --tile-size
# Add --amp for bf16 mixed precision (~5x faster on RTX cards).
#   Use bf16 for EXPERIMENTATION (architecture/feature/data iteration on MAE/loss).
#   Train the SHIPPED model in fp32 (omit --amp): V11 showed bf16 raises the
#   shoal-target TVU breach up to 13.5x. The safety go/no-go is the shoal-breach
#   number, not MAE, and bf16 fails it. See CHANGELOG 2026-06-15, LESSONS Lesson 21.
```

### 3b. Verify graph construction (required once before retraining)

```bash
python scripts/verify_graph_equivalence.py --ground-truth-dir "path/to/ground_truth"
# Must print PASS before starting a training run after graph-construction changes.
```

### 4. Run Inference
```bash
# Conservative (high confidence only)
python scripts/inference_native.py \
    --input "path/to/noisy.bag" \
    --model "path/to/best_model.pt" \
    --output "path/to/denoised.bag" \
    --auto-correct-threshold 0.7

# Balanced
python scripts/inference_native.py \
    --input "path/to/noisy.bag" \
    --model "path/to/best_model.pt" \
    --output "path/to/denoised.bag" \
    --auto-correct-threshold 0.6

# Aggressive (more corrections)
python scripts/inference_native.py \
    --input "path/to/noisy.bag" \
    --model "path/to/best_model.pt" \
    --output "path/to/denoised.bag" \
    --auto-correct-threshold 0.5
```

### 5. Visual Validation in QGIS
```
1. Load sidecar GeoTIFF (*_gnn_outputs.tif)
2. Check Band 1 (classification) - noise should follow spatial patterns, NOT tile boundaries
3. Check Band 3 (correction) - magnitudes should be reasonable for the survey area
4. Create diff layer: predicted corrections minus actual (clean - noisy) difference
5. Large residuals in diff = model under/over-correcting at those locations
```

---

## Common Issues and Fixes

### Issue: "Tile size must be larger than 2x overlap"
**Fix:** Edit config.yaml in model output folder:
```bash
powershell -Command "(Get-Content 'path/to/model_output/config.yaml') -replace 'overlap: 128', 'overlap: 64' | Set-Content 'path/to/model_output/config.yaml'"
```

### Issue: "mat1 and mat2 shapes cannot be multiplied"
**Cause:** Model trained with different number of features than inference
**Fix:** Ensure ground truth files have 5 bands (including uncertainty)

### Issue: CUDA out of memory
**Fix:** Reduce batch-size: `--batch-size 2`

### Issue: Model classifies everything as noise (~95%+)
**Cause:** Systematic offset between clean/noisy surveys
**Fix:** Use updated prepare_ground_truth.py (removes offset automatically)

### Issue: Model classifies almost nothing as noise (<1%)
**Cause:** Class imbalance without proper weighting
**Fix:** Use updated trainer.py with auto class weighting (V6+)

### Issue: Noise classifications follow tile/survey boundaries
**Cause:** Nodata values bleeding into local feature statistics
**Fix:** Use boundary-aware feature computation (V7+, already in current code)

### Issue: Model detects noise but corrections are too small
**Cause:** Huber loss gradient plateau for large corrections
**Fix:** Use local_std correction normalization (V9+, already in current code)

### Issue: Ground truth has near-zero noise (<1%)
**Cause:** Noise in point cloud doesn't propagate to gridded BAG surface
**Fix:** This pair is not suitable for training. Find pairs where noise is visible in the gridded surface.

### Issue: Model over-predicts noise (detection rate >> ground truth noise %)
**Cause:** Model learned location-specific seafloor patterns as noise-like from limited geographic training data
**Fix:** Add geographically diverse training data. Also consider tracking precision/recall separately and tuning the classification threshold to reduce false positives.

### Issue: Pervasive cell-to-cell differences (60-99% noise) in prepare_ground_truth output
**Cause:** Survey pairs produced by running CUBE twice on different point clouds (with/without outliers), as opposed to Seward where manual grid edits were applied after a single CUBE run
**Fix:** This is expected behavior, not a quality problem. Use `--adaptive-threshold` for classification mode or `--regression-mode` for continuous targets.

### Issue: Variable offsets across sub-files of the same survey
**Cause:** Datum mismatch with spatial variation (e.g., survey datum vs MLLW separation varies geographically across the survey extent)
**Fix:** Apply offset removal per sub-file (the default behavior). Each sub-file gets its own median offset computed independently. For better accuracy, apply a proper datum transformation upstream.

### Issue: 50/50 shoal/deep split in difference distribution
**Cause:** Normal for CUBE re-grid differences (removing outliers shifts the weighted estimate symmetrically in both directions)
**Fix:** Not actually a problem. This is only a quality signal for surfaces produced by post-grid manual editing (where the cleanup direction biases toward shoals or deeps based on which type of noise the hydrographer targeted).

### Issue: Adaptive threshold varies widely across surveys (0.12m to 6.73m)
**Cause:** CUBE run-to-run variability scales with depth and resolution
**Fix:** Either accept different thresholds per pair (works for individual surveys), or use regression mode for a unified treatment across depth regimes.

### Issue: Pipeline crashes when loading regression-mode and classification-mode files together
**Cause:** Should not crash; both file types should coexist in the same directory
**Fix:** Verify each file's mode in startup logs ("Loaded X in regression mode" / "Loaded X in classification mode"). Mode is per-file, not global.

### Issue: Huber delta is very large (>50) or training loss plateaus
**Cause:** Delta computed from raw correction magnitudes instead of normalized values, putting Huber loss in pure linear mode (effectively MAE)
**Fix:** Verify startup log shows "Sampling 50 tiles to compute Huber delta from normalized corrections..." followed by a delta value typically in the 1-10 range. If delta is much larger, check that `_compute_training_stats` is using the dataset-sample approach rather than raw correction magnitudes.

### Issue: Validation loss bounces wildly between epochs
**Cause:** Training and validation distributions don't match (e.g., training on mixed depth regimes but validating on only one)
**Fix:** Use `--val-surveys` flag pointing at a folder with multiple files covering the same regimes as training. Alternative is tile-based split (not currently supported by train.py).

### Issue: VR survey pair shows huge one-sided difference (e.g. 99% deep, tens-of-meters offset)
**Cause:** Historic bug where the warp step re-opened the raw BAG with GDAL's default VR interpretation instead of the resampled surface. Fixed in `warp_grid_to_reference`.
**Fix:** Ensure you are on the fixed `prepare_ground_truth.py`. Validate against a CARIS difference export: a healthy noise-removal difference has a near-zero median and roughly symmetric (~50/50) direction split. A wildly asymmetric split signals a processing problem, not real noise.

### Diagnostic: Verify a resampled surface matches CARIS
Use `scripts/check_resampled_surface.py` to compare a resampled BAG surface against a CARIS XYZ export at matching locations:
```
python scripts/check_resampled_surface.py --bag "survey.bag" --caris "survey_caris.txt"
```
A near-1.0 correlation with small mean |diff| means the surface is correct. A near -1.0 correlation means the surfaces match but use opposite sign conventions (GDAL negative-down vs CARIS positive-down), which is expected and harmless within the pipeline.

---

## Output Files Explained

### Denoised BAG File
- Same format as input (VR stays VR, SR stays SR)
- Noise cells have corrected depth values
- Ready for use in downstream products

### Sidecar GeoTIFF (_gnn_outputs.tif)
4 bands:
1. **Classification**: 0=seafloor, 2=noise
2. **Confidence**: 0.0-1.0 model certainty
3. **Correction**: Depth adjustment applied (meters)
4. **Valid mask**: 1=valid data, 0=nodata

---

## Ground Truth Quality Checklist

Before adding a survey pair to training data:

| Check | Expected | Red Flag |
|-------|----------|----------|
| QGIS difference layer | Scattered spikes visible | Uniformly near zero |
| Noise percentage | 10-40% | <1% or >50% |
| Systematic offset | <0.2m | >0.3m |
| Seafloor mean diff | ~0 (after offset removal) | >0.01m |
| BAG type | VR or SR (both supported) | -- |

---

## Recommended Workflow

1. **Verify data pair quality** in QGIS (subtract surfaces, look for noise spikes)
2. **Prepare ground truth** with prepare_ground_truth.py
3. **Check noise percentage** (10-40% is usable)
4. **Train model** with auto class weighting
5. **Validate visually** in QGIS (not just metrics)
6. **Test on held-out survey** not used in training
7. **Adjust threshold** based on visual QC
8. **Compare original vs denoised** in GIS software

---

## Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| tile-size | 256 | 128-512 | Larger = more context, more memory |
| batch-size | 2 | 1-8 | Larger = faster, more memory |
| epochs | 30 | 20-50 | Usually early-stops around 15-25 |
| noise-threshold | 0.15m | 0.1-0.3 | For ground truth labeling |
| auto-correct-threshold | 0.85 | 0.4-0.9 | For inference corrections |

---

## Performance Expectations (Current Model - V9)

- **Training time**: ~3-5 hours (CUDA, batch-size 2, 298 tiles, 30 epochs)
- **Inference time**: ~20 seconds per survey (CUDA)
- **Best validation accuracy**: ~72%
- **Noise detection rate**: ~35% (over-predicts vs 18.8% validation ground truth; expect improvement with diverse training data)
- **Mean confidence**: 0.825
- **Memory**: ~6-8 GB GPU RAM with batch-size 2

---

## Architecture Summary

- **Model**: Graph Attention Network (GAT), 4 layers, 64 hidden channels, 182K parameters
- **Node features (8)**: depth, local mean, local std, gradient magnitude, gradient direction, curvature, uncertainty, boundary distance
- **Edge features (3)**: distance, depth difference, slope angle
- **Output heads (3)**: classification (3-class), confidence (0-1), correction (meters, normalized by local_std)
- **Loss**: Weighted cross-entropy (classification) + shoal safety asymmetric (safety) + Huber (correction)

---

## Data Acquisition Tracker

22 surveys across 8 regions requested from NCEI archive (2026-03-04). E00269 available locally.

| Region | Surveys | Status |
|--------|---------|--------|
| Gulf Coast (TX/LA/MS/FL) | H13818, H13651, H13837 | Awaiting archive |
| SE Atlantic (SC/GA) | H13851, F00881 | Awaiting archive |
| Mid-Atlantic (MD/VA/NC) | H13762, H13804, H13750 | Awaiting archive |
| Northeast (NY/CT/RI) | H13927 | Awaiting archive |
| Great Lakes | H13940, H13943 | Awaiting archive |
| Pacific NW (WA/OR) | H14070, H13847 | Awaiting archive |
| Alaska | H13774, F00886, H14116, H13914, E01093, H13695 | Awaiting archive |
| Pacific Islands | H13739, H13735, E00269 | E00269 ready, others awaiting |

**Processing order:** One per region first (prioritize Gulf Coast, Great Lakes, Pacific Islands), then fill in remaining surveys.

---

*Quick Reference v3.1 | June 2026*


## TVU breach evaluation (evaluate_v10.py)

```bash
python scripts/evaluate_v10.py \
    --checkpoint outputs/best_model.pt \
    --ground-truth-dir ground-truth-val/ \
    --iho-order general2 \
    --output eval/results.json
```

`--iho-order` accepts IHO S-44 labels (exclusive, special, 1a, 1b, 2) or NOAA
HSSD OCS Quality Metric labels (exceptional, critical, general1, general2,
general3, general4). Explicit `--tvu-a` / `--tvu-b` override the label. The
applicable metric per survey comes from its Project Instructions.

### Issue: "median valid depth is negative ... predates the depth-convention fix"

The ground-truth file stores elevation (negative-down) and was generated before
2026-06-09. Regenerate it with the current `prepare_ground_truth.py` (the
loader now enforces positive-down depth). Pre-fix checkpoints likewise embody
the inverted objective; V11 is the first model retrained with the fix, so use a
V11 (or later) checkpoint for any direction-sensitive evaluation.

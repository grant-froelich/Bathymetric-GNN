# Bathymetric GNN - Quick Reference Guide

*Updated: March 2026 (V9)*

## Commands Cheat Sheet

### 1. Verify Training Data Quality (Do This First!)
```bash
# In QGIS: Use Raster Calculator to subtract clean from noisy surface
# Look for scattered depth spikes in the difference layer
# If difference is uniformly near zero, DO NOT USE this pair
```

### 2. Prepare Ground Truth (Training Data)
```bash
# Works with both VR and SR BAGs (auto-detected)
python scripts/prepare_ground_truth.py \
    --clean "path/to/clean_survey.bag" \
    --noisy "path/to/noisy_survey.bag" \
    --output-dir "path/to/ground_truth"

# Verify output:
#   Noise percentage should be 10-40%
#   Systematic offset should be small (<0.2m)
#   Seafloor mean diff should be ~0
```

### 3. Train Model
```bash
python scripts/train.py \
    --ground-truth-dir "path/to/ground_truth" \
    --output-dir "path/to/model_output" \
    --epochs 30 \
    --device cuda \
    --tile-size 256 \
    --batch-size 4

# If CUDA out of memory, use batch-size 2
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
- **Noise detection rate**: ~35% (matching ground truth distribution)
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

*Quick Reference v2.1 | March 2026*

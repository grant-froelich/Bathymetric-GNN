# Lessons Learned: Real-World Training with Clean/Noisy Survey Pairs

This document captures practical lessons from training the Bathymetric GNN on real survey data (Seward, Alaska multibeam surveys). It complements the theoretical documentation in HOW_IT_WORKS.md and TRAINING_PLAN.md.

*Document Version: 2.0*
*Updated: March 2026*
*Based on: Seward, Alaska training data, V1-V9 training runs*

---

## Key Discoveries

### 1. Systematic Offset Between Survey Pairs

**Problem Discovered:** Clean and noisy survey pairs may have systematic datum/processing offsets (we found -12 to -18 cm in the Seward data).

**Symptoms:**
- Model classifies 95%+ of cells as noise
- Uniform correction applied across entire surface
- Validation accuracy oscillates between ~42% and ~58% (near random)

**Root Cause:** When the "seafloor" class (supposedly unchanged areas) has a non-zero mean difference, the model learns "everything needs correction" instead of "identify specific noise."

**Solution Implemented:** `prepare_ground_truth.py` now automatically:
1. Calculates median difference between clean and noisy surveys
2. Subtracts this offset before applying noise threshold
3. Logs the detected offset for transparency

```
[INFO] Detected systematic offset: -0.127m (will be removed)
[INFO] Seafloor mean diff (should be ~0): -0.002m
```

**Verification:** After fix, seafloor mean difference should be ~0 (within 0.01m).

---

### 2. Class Imbalance Causes Majority-Class Prediction

**Problem Discovered (V5):** Without class weights, the model learned to predict "seafloor" for every cell. Validation accuracy of 67% matched the seafloor proportion exactly, giving the illusion of learning.

**Symptoms:**
- High validation accuracy (~67%) but 0% noise detection at inference
- Model predicts majority class for everything
- Confidence very high (0.967) because the model is "confidently wrong"

**Root Cause:** Standard cross-entropy loss treats all errors equally. The model minimizes loss by predicting the majority class.

**Solution Implemented (V6+):** `trainer.py` automatically computes inverse-frequency class weights by scanning training tiles. Noise (minority class) gets higher weight, seafloor (majority class) gets lower weight. The weights are computed fresh each training run based on actual data distribution.

```
[INFO] Class distribution: {0: 5277551, 1: 0, 2: 1773525}
[INFO] Class weights: [0.252, 2.139, 0.609]
```

---

### 3. Uncertainty Feature is Critical

**Problem Discovered:** Models trained without uncertainty (7 features) couldn't use BAG uncertainty data during inference, causing dimension mismatch errors.

**Error Message:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5x8 and 7x64)
```

**Solution:** Ground truth files now include 5 bands:
1. Labels
2. Difference
3. Noisy depth
4. Clean depth
5. **Uncertainty** (from noisy survey)

Model checkpoint saves `in_channels` so inference knows expected feature count.

---

### 4. Boundary Feature Contamination

**Problem Discovered (V6):** The model classified cells along survey tile boundaries as noise, not actual noise locations. Visual validation in QGIS showed classification maps that followed rectangular data boundaries rather than noise spatial patterns.

**Root Cause:** `scipy.ndimage.uniform_filter` with `mode='nearest'` was used to compute local statistics (mean, std). At data boundaries where nodata values (1e6) exist, 'nearest' mode pads with those extreme values, creating artificially high gradients and curvature. The model learned these boundary artifacts as the easiest "noise-like" signal.

**Solution (V7):** Replaced uniform_filter with masked local statistics:
- Local mean/std computed using only valid (non-nodata) neighbors
- Nodata cells filled with local mean before gradient and curvature computation
- This eliminates artificial feature spikes at boundaries

**Impact:** Noise detection jumped from 3.3% (boundary artifacts) to 34.8% (actual noise), matching ground truth distributions of 16-34%.

**Lesson:** Always validate model outputs spatially. A model can achieve reasonable accuracy metrics while learning the wrong signal entirely. If V6's classifications had not been visualized in QGIS, this bug could have persisted through many iterations.

---

### 5. Correction Magnitude Requires Normalization

**Problem Discovered (V7/V8):** The model correctly identified noise locations but predicted corrections of ~0.4m where 70m was needed. The QGIS diff layer confirmed large residual errors at known noise cells.

**Root Cause:** The correction head uses Huber loss with delta=1.0. For any error above 1m, Huber loss gives identical gradient magnitude. The model has no incentive to predict 70m over 2m because both produce the same training signal. Since 95% of correction targets are below 0.652m, the model converges on small corrections for everything.

**Failed Fix (V8):** Attempted to compute Huber delta from the data's 95th percentile, but the 95th percentile (0.652m) was below the minimum delta floor of 1.0, resulting in identical training dynamics to V7.

**Working Fix (V9):** Normalize correction targets by local surface variability (local_std). Instead of learning "correct by 70 meters," the model learns "correct by N local standard deviations." A floor of 0.01m prevents division by zero, and values are capped at +/-50 std devs to handle extreme cases in flat areas. Inference denormalizes by multiplying predictions by local_std.

**V9 Results:** The diff layer (V9 minus V7 corrections) showed the model now applies corrections up to 32m larger than V7, confirming the normalization is working. Corrections still don't fully recover the clean surface, but the improvement is substantial.

---

### 6. Training Data Quality: Not All Survey Pairs Are Useful

**Problem Discovered:** Three new survey pairs (Florida river SR, Coastal Alaska VR, Norfolk river SR) all produced ground truth with near-zero noise cells (4, 149, and 1 noise cells respectively out of millions of valid cells).

**Root Cause:** The noise differences between clean and dirty versions did not propagate to the gridded BAG surface. Possible explanations:
- CUBE's robust gridding algorithm already rejected outlier soundings during surface generation
- The "dirty" versions were already substantially processed
- Noise exists in the point cloud but not in the gridded surface

**Impact if used for training:** Adding 54M all-seafloor cells to a training set with 5.3M seafloor and 1.8M noise cells would shift the balance from 75/25 to 97/3, pushing the model toward majority-class prediction (the V5 failure mode).

**Lesson:** Before running `prepare_ground_truth.py`, verify the data pair is useful:
1. Subtract the two surfaces in QGIS with the raster calculator
2. Look for spatially scattered spikes in the difference layer
3. If the difference is uniformly near zero, the pair won't produce useful training data
4. Noise percentage should be 10-40%. Near 0% means the pair isn't suitable.
5. This tool operates on gridded surfaces. If noise only exists in the point cloud, it won't help.

---

### 7. Persistent Overfitting Across All Versions

**Observation:** Validation loss diverges from training loss after approximately 5 epochs in every ground truth training run (V5-V9). Early stopping consistently triggers around epochs 15-25.

**Root Cause:** All 4 ground truth pairs come from the same Seward, Alaska area. The model memorizes Seward-specific patterns rather than learning generalizable noise signatures.

**Mitigation:** Geographic diversity in training data is the highest-priority improvement. The local_std normalization (V9) was specifically designed to make corrections comparable across different depth regimes in preparation for multi-location training.

---

## Training Performance Summary

| Version | Features | Key Change | Val Accuracy | Noise Detection | Behavior |
|---------|----------|------------|--------------|-----------------|----------|
| V1 | 7 | No uncertainty | 48% oscillating | N/A | Random |
| V2 | 8 | Added uncertainty | 49% oscillating | N/A | Random |
| V3 | 8 | Removed offset | ~67% stable | 0.6% | Too conservative |
| V4 | 8 | Manual 2x noise weight | -- | 96.8% | All-noise |
| V5 | 8 | No class weights (bug) | ~67% | 0.0% | All-seafloor |
| V6 | 8 | Auto class weights | ~61% | 3.3% | Boundary artifacts |
| V7 | 8 (boundary-aware) | Masked local stats | ~72% | 34.8% | First real detection |
| V8 | 8 (boundary-aware) | Dynamic Huber delta | ~71% | = V7 | No change (delta at floor) |
| V9 | 8 (boundary-aware) | local_std correction norm | ~72% | 34.8% | Larger corrections |

### Training Curves Interpretation

**Healthy training (V7+):**
- Train loss steadily decreases
- Val accuracy (~72%) is slightly below the seafloor proportion (75%), but this is expected: the model trades some seafloor accuracy for noise detection (34.8%), confirming genuine classification rather than majority-class collapse
- Val loss oscillates but stays in a bounded range
- Early stopping triggers around epoch 15-25

**Unhealthy training (V1/V2 - Synthetic data):**
- Val accuracy oscillates between ~42% and ~58%
- This is the model flipping between "predict all seafloor" and "predict all noise"
- Indicates data quality issue, not model issue

**Majority-class collapse (V5):**
- Val accuracy ~67% matches seafloor proportion exactly
- Inference produces 0% noise detection
- High confidence (0.967) masking total failure

---

## Recommended Ground Truth Preparation Workflow

```bash
# 1. FIRST: Visual check in QGIS
#    Subtract noisy - clean surfaces with raster calculator
#    Look for scattered spikes in the difference
#    If difference is uniformly near zero, DO NOT USE this pair

# 2. Run prepare_ground_truth.py with offset detection
python scripts/prepare_ground_truth.py \
    --clean "clean_survey.bag" \
    --noisy "noisy_survey.bag" \
    --output-dir "ground_truth/"

# 3. Verify in output:
#    - "Detected systematic offset: X.XXXm" - should be small (<0.2m)
#    - "Seafloor mean diff (should be ~0): 0.00Xm" - should be <0.01m
#    - Noise percentage should be 10-40% (not 50%+ or <1%)
#    - If noise percentage is near 0%, the pair is not useful for training

# 4. If offset is large (>0.3m), investigate survey datum differences

# 5. Repeat for all survey pairs
```

---

## Recommended Training Workflow

```bash
# Use class weighting (automatic in updated trainer.py)
python scripts/train.py \
    --ground-truth-dir "ground_truth/" \
    --output-dir "model_output/" \
    --epochs 30 \
    --device cuda \
    --tile-size 256 \
    --batch-size 4

# Verify in output:
#    - "Class distribution: seafloor=X, feature=Y, noise=Z"
#    - "Using class weights: [w0, w1, w2]" - noise weight should be highest
#    - Noise percentage should be 10-40%
```

---

## Inference Threshold Selection

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.8-0.9 | Very conservative | Production, high safety requirements |
| 0.6-0.7 | Balanced | General QC, recommended starting point |
| 0.4-0.5 | Aggressive | Exploratory analysis, noisy data |

**Tip:** Run inference at 0.5 threshold, review sidecar GeoTIFF in QGIS, then adjust based on false positive rate.

---

## Visual Validation Checklist

After any training run, validate in QGIS before trusting metrics:

1. Load the sidecar GeoTIFF classification band
2. Does the noise pattern follow data boundaries? (Bad - see V6 lesson)
3. Does the noise pattern match where you'd expect noise? (Good)
4. Load the correction band and check magnitudes against known noise
5. Create a diff layer (predicted corrections minus actual difference) to quantify residual error

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Offset not removed | 95%+ noise classification | Use updated prepare_ground_truth.py |
| No class weights | <1% noise classification | Use updated trainer.py (auto-weights) |
| Wrong feature count | Dimension mismatch error | Regenerate ground truth with uncertainty |
| Config overlap mismatch | "Tile size must be larger than 2x overlap" | Edit config.yaml: overlap: 64 |
| CUDA OOM | Out of memory error | Reduce batch-size to 2 |
| Boundary contamination | Classifications follow tile edges | Use boundary-aware feature computation (V7+) |
| Low correction magnitudes | Model flags noise but corrections too small | Use local_std correction normalization (V9+) |
| Near-zero noise in GT | Noise doesn't propagate to grid surface | Verify noise visible in gridded BAG, not just point cloud |
| All-seafloor training data | Pushes model toward majority-class collapse | Only use pairs with 10-40% noise in gridded surface |

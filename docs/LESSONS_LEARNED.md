# Lessons Learned: Real-World Training with Clean/Noisy Survey Pairs

This document captures practical lessons from training the Bathymetric GNN on real survey data (Seward, Alaska multibeam surveys and Pacific Islands surveys). It complements the theoretical documentation in HOW_IT_WORKS.md and TRAINING_PLAN.md.

*Document Version: 3.1*
*Updated: June 2026*
*Based on: Seward, Alaska training data, V1-V9 training runs, Pacific Islands and Alaska ground truth processing, V10 regression mode, VR warp bug fix*

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

**Problem Discovered (V5):** Without class weights, the model learned to predict "seafloor" for every cell. Validation accuracy stabilized at ~67%, and inference produced 0% noise detection, confirming majority-class-like collapse.

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

**Impact:** Noise detection jumped from 3.3% (boundary artifacts) to 34.8% (actual noise spatial patterns). However, the validation set (Seward 4of4) has 18.8% noise, so 34.8% detection indicates significant over-prediction -- the model flags nearly twice as many cells as are actually noise, meaning a substantial false positive rate. This is likely due to the model learning Seward-specific patterns: normal seafloor variation that looks "noise-like" within that one geographic context gets incorrectly flagged.

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
- Val accuracy (~72%) is slightly below the seafloor proportion (75%), but this is expected: the model trades some seafloor accuracy for noise detection, confirming genuine classification rather than majority-class collapse
- Caveat: the noise detection rate (34.8%) is nearly double the validation ground truth (18.8%), indicating significant false positives that should improve with geographic diversity in training data
- Val loss oscillates but stays in a bounded range
- Early stopping triggers around epoch 15-25

**Unhealthy training (V1/V2 - Synthetic data):**
- Val accuracy oscillates between ~42% and ~58%
- This is the model flipping between "predict all seafloor" and "predict all noise"
- Indicates data quality issue, not model issue

**Majority-class collapse (V5):**
- Val accuracy stabilizes at ~67% with no learning progression
- Inference produces 0% noise detection, confirming the model never learned to identify noise
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

### 8. Different CUBE Runs Produce Pervasive Cell-to-Cell Differences

**Problem Discovered (May 2026):** When processing E00269 and H13739 with `prepare_ground_truth.py`, noise percentages came out at 60-99% across all sub-files, far higher than Seward's 16-34%. Initial assumption was a datum mismatch or processing error.

**Symptoms:**
- 50/50 shoal/deep split in difference distribution (Seward typically has directional bias)
- Pervasive cell-to-cell differences across the entire surface
- Differences scale with depth (deeper water = larger differences)
- Variable median offset across sub-files of the same survey

**Root Cause:** Seward training pairs were produced by running CUBE once, then applying manual grid edits to the surface after the fact. Cells that weren't edited remained identical between clean and noisy versions. Other survey pairs (E00269, H13739) are produced by running CUBE twice on different point clouds (with and without outliers). When the input point cloud differs, the weighted CUBE estimate changes at every cell that had any noisy sounding contribute to it. Most cells differ slightly; cells affected by significant outliers differ a lot.

**Implication:** The difference distribution between two CUBE runs is a continuous signal. The "noise threshold" approach assumes a clean separation between "unchanged seafloor" and "noise" cells, but with re-gridded data there's no such separation. The threshold becomes arbitrary.

**Solution:** Either use an adaptive threshold (Otsu's method on log-scaled absolute differences) for classification mode, or shift to regression mode entirely. The regression approach uses the continuous difference as the training target directly, without any threshold.

**The 50/50 shoal/deep split is NOT a quality warning** for CUBE re-grid differences. Symmetric error distributions are expected when removing outlier soundings affects estimates in both directions equally. A directional bias only appears when the outliers have a consistent sign (refraction, multipath, etc.) AND when the cleaning method preserves that bias in the surface.

---

### 9. Datum Mismatches Manifest as Variable Offsets Across Sub-Files

**Problem Discovered (May 2026):** E00269 sub-files showed offsets ranging from -0.51m to +5.91m across the six sub-files. A simple datum offset should be constant.

**Root Cause:** The separation between survey datum and MLLW is not constant across a large survey area. It varies spatially with the geoid model and tidal zoning. Different sub-files cover different geographic areas, so each gets a different median offset.

**Symptoms:**
- Different sub-files of the same survey have systematically different offsets
- Median and mean offsets diverge (mean is much larger when there are extreme noise outliers)
- The offset is real and removal works correctly (seafloor mean diff is exactly 0.000m after removal)

**Solution:** The `--no-offset` flag was added to skip offset removal when both surfaces are confirmed in the same datum. When datums differ, the existing median-subtraction approach works for each sub-file individually. For a survey with sub-files in different geographic locations, you may need to apply a proper datum transformation rather than a single offset removal.

---

### 10. Classification Thresholds Don't Generalize Across Depth Regimes

**Problem Discovered (May 2026):** A fixed 0.15m noise threshold works for Seward (shallow water, ~20-200m) but is meaningless for deep water surveys. At 2000m depth, 0.15m is well within normal CUBE run-to-run variability.

**Symptoms:**
- Shallow water (Seward, 4m E00269): adaptive threshold settles at 0.12-0.20m
- Deep water (H13739): adaptive threshold settles at 6.73m
- Very deep water (128m E00269): mean correction is 84m, with the 99th percentile at 685m
- Using a single fixed threshold across surveys produces inconsistent labels for similar physical situations

**Root Cause:** The magnitude of CUBE run-to-run variability scales with depth and resolution. Cells with larger uncertainty have larger possible variation when their input soundings change. A 1m difference in 20m water means something completely different than a 1m difference in 2000m water.

**Implication for Classification Mode:** Each pair needs its own threshold. The `--adaptive-threshold` flag handles this automatically via Otsu's method. But mixing different-threshold pairs in one training set creates inconsistent labels.

**Implication for Regression Mode:** The local_std normalization (V9) and adaptive threshold concept converge to the same insight. Regression with local_std normalization expresses every correction in units of local variability, which is naturally consistent across depth regimes.

---

### 11. Binary Classification Loses Information

**Problem Identified (May 2026):** Reviewing V7/V9 over-prediction rates (34.8% detection vs 18.8% ground truth) led to a deeper question: is classification the right framework at all?

**Structural Problems with Classification Approach:**
- Forces a binary decision on a continuous signal
- Cells with 0.14m and 0.16m corrections get different labels when threshold is 0.15m, but they're nearly identical physically
- Correction head only trains on cells labeled noise, never sees small or zero corrections
- Threshold value determines what the model learns
- False positives concentrate near threshold boundary because no real boundary exists there

**The Continuous Reality:** When a hydrographer cleans a survey, the difference between the dirty and clean surfaces at every cell is the total correction needed. Some cells need 0.001m, some need 0.5m, some need 30m. It's a spectrum, not two categories.

**The Regression Insight:** The model should learn to predict the correction at every cell, including near-zero corrections for cells where cleaning barely changed anything. This:
- Preserves the full continuous signal
- Eliminates threshold dependence
- Handles mixed depth regimes naturally (with local_std normalization)
- Maps directly to what hydrographers actually need to know
- Allows the inference threshold to move from training time to inference time (more flexible)

**The V10 Architecture Shift:** Use the raw difference (offset-corrected if needed) as the regression target. Apply asymmetric Huber loss with shoal-safety asymmetry baked into the loss direction. Skip the binary classification step entirely.

---

### 12. Shoal Safety in Regression Mode

**Sign Convention (depths positive down, correction = noisy - clean):**
- `corrected_depth = noisy_depth - predicted_correction`
- `error = predicted_correction - target_correction`
- `error > 0`: corrected depth is shallower than reality (SAFE for navigation - we say there is less water than there really is)
- `error < 0`: corrected depth is deeper than reality (DANGEROUS - we say there is more water than there really is)

**Loss Implementation:** Asymmetric Huber penalty weights `error < 0` cases by `dangerous_weight` (default 3.0) and `error >= 0` cases by `safe_weight` (default 1.0). The penalty is on the depth error direction, not the correction error direction. Both shoal-direction (target < 0) and deep-direction (target > 0) corrections are subject to this same asymmetry.

**Comparison to Classification Mode:** In classification, shoal safety was an asymmetric penalty on false positives that removed real shoals (`ShoalSafetyLoss`). In regression, shoal safety is built directly into the primary loss function via the sign of the error.

---

### 13. The Difference Layer Has No Single Right Threshold

**Problem Identified (May 2026):** Even within a single deep water survey, the appropriate "noise threshold" varies spatially. Cells in flat areas have small natural variability, so a 1m difference is suspicious. Cells in rugged terrain have large natural variability, so a 1m difference may be normal.

**Why Adaptive Thresholds Help (Partially):** Otsu's method finds a global break point in the difference distribution for the whole survey. This is better than a fixed threshold across surveys, but still imposes a single break point on data that may have different appropriate cutoffs in different regions.

**Why Regression is Better:** The model learns from features (local depth, gradient, curvature, uncertainty) which cells need large corrections and which need small ones. The decision is per-cell, contextual, and continuous. No global threshold is needed at all during training.

**At Inference Time:** A threshold reappears, but as an operational choice: "apply automatic corrections where the predicted magnitude exceeds X meters." This threshold can be tuned per survey, per region, or per use case without retraining.

---

### 14. Classification Metrics Don't Fit Regression Output

**Problem Identified (May 2026):** Tempting to apply accuracy, precision, recall, and F1 to V10 by post-hoc thresholding the predicted corrections and comparing against thresholded ground truth.

**Why This Doesn't Work:** It reintroduces exactly the threshold problem V10 was designed to avoid. A cell with a 0.14m predicted correction and one with 0.16m are nearly identical predictions, but a 0.15m threshold classifies them differently. The arbitrary threshold determines whether the model "looks good" on classification metrics, so those metrics don't actually measure model quality.

**Metrics V10 Uses Instead:**
- **MAE and RMSE** in meters: standard regression accuracy
- **MAE normalized by local_std**: comparable across depth regimes
- **Per-magnitude-bucket MAE**: separates performance on small vs large corrections
- **Hazardous error rate**: safety-critical (fraction of cells where predicted < target, meaning corrected depth ends up deeper than reality)
- **Recovery RMSE**: how close the corrected surface gets to the clean reference (single-number operational summary)

**What Stays Outside the Model's Metrics:**
- IHO order compliance is determined by the uncertainty layer in the BAG, not the model
- Charted feature preservation is enforced by operational thresholds and human QC, not by training metrics

Implementation: `training/metrics.py` defines `V10Metrics` dataclass and `compute_v10_metrics()` function. See HOW_IT_WORKS.md for the full rationale per metric.

---

### 15. Huber Delta Must Match the Scale of Predictions

**Problem Discovered (May 2026):** V10's first multi-file training run reported a Huber delta of 281.7m. Training proceeded but with erratic validation loss curves and weak convergence.

**Root Cause:** `_compute_training_stats` was computing the 95th percentile of *raw* correction magnitudes in meters, while the model was training on *normalized* corrections (raw divided by local_std, then clipped to ±50 std-devs). The delta of 281 was orders of magnitude larger than any prediction error the model could produce, putting the Huber loss in pure linear mode for the entire training run. Effectively, Huber became MAE, losing the precise gradient signal it was supposed to provide for small errors.

**Diagnostic Symptoms:**
- Delta value much larger than the correction normalization cap (±50)
- Training loss decreases but plateaus far from zero
- Validation MAE stays high without strong overfitting signal (train and val loss similar)
- Loss curve looks "stuck" rather than improving steadily

**Fix:** Compute delta from the actual normalized correction targets the model sees during training. The updated `_compute_training_stats` samples 50 random tiles from the dataset, builds their graphs, and collects the normalized correction targets, then computes the 95th percentile of those.

**After Fix:** Delta typically lands in the range of 3-10 std-devs for E00269 data. The Huber loss now operates in the quadratic regime for most prediction errors, with the linear regime reserved for the actual tail of outlier errors.

**General Principle:** Any loss parameter that depends on the scale of model outputs must be computed in the same units as those outputs. Normalization is invisible at this level of code, so it's easy to mix units. Sampling actual graphs (rather than computing from raw data) eliminates the unit-mismatch risk entirely.

See HOW_IT_WORKS.md for a fuller explanation of Huber loss, the delta parameter, and the loss shape.

---

### 16. Per-Cell Resolution as a Conditioning Feature

**Problem Addressed (May 2026):** The single model handled shallow water reasonably but failed in deep water, predicting a roughly constant correction magnitude regardless of what each cell needed (flat per-bucket MAE). One hypothesis was that shallow and deep water are different enough to need separate models.

**Why Not Separate Models:** VR BAGs vary resolution within a single surface, so per-survey model routing is impossible (a VR surface has no single resolution to route on). Any regime conditioning must be per-cell, not per-survey. This pushed toward a single model conditioned on each cell's resolution rather than multiple models.

**The Feature:** Added `log_footprint` as a node feature: log2 of the cell footprint in meters. For SR surveys it is constant per file; for VR it will vary per cell once native resolution is preserved through loading. log2 is used so that equal resolution ratios map to equal feature distances (4m to 8m is the same step as 128m to 256m), matching the multiplicative nature of resolution effects.

**Result (controlled comparison on E00269, same split):**
- Best validation loss improved from 1.64 to 1.31
- Shallow water overall MAE roughly halved (1.99m to 0.87m)
- The <0.1m bucket (clean seafloor that should barely move) improved from 1.39m to 0.52m
- Shoal hazard rate stayed at 0.00%
- Deep water improved only marginally and stayed flat per-bucket

**Interpretation:** Resolution conditioning helps a single model handle multiple regimes, and it helped most in the regime with enough training signal (shallow). Deep water staying flat confirms the deep failure is a data limitation (one 128m survey), not something the feature can fix alone. This supports a single conditioned model over separate per-regime models.

**Caveat to Watch:** Deep-direction hazard rate in shallow water rose (7.32% to 31.28%) even as overall accuracy improved. Shoal protection was unaffected (0.00%). Worth monitoring as more data is added.

**Forward Note:** For VR surfaces, the per-cell resolution is lost during GDAL resampled loading (`MODE=RESAMPLED_GRID` collapses to uniform resolution). To make this feature vary per cell on VR data, the native refinement resolution must be carried through the resampling step. Until then, VR surveys get a constant log_footprint equal to their resampled resolution.

---

### 17. Difference Two Surfaces Only After Loading Them the Same Way

**Problem Discovered (June 2026):** Ground truth for VR survey pairs was systematically wrong whenever the clean and noisy surveys had different resolutions. H14070 showed a 78m phantom offset with 99% one-sided direction; H14116 showed 43m and 99.7% one-sided. H13739 was mildly wrong (correct direction, 3x inflated magnitudes). The source data was fine in all cases.

**Root Cause:** When resolutions differed, `prepare_ground_truth.py` loaded the noisy surface twice. First correctly via the resampled-mode loader (`MODE=RESAMPLED_GRID`), then it discarded that and re-opened the raw BAG with `gdal.Warp(tmp, str(noisy_path), ...)` to align it to the clean grid. But `gdal.Warp` on a raw VR BAG path does not use `MODE=RESAMPLED_GRID`; it falls back to GDAL's default VR interpretation (the low-resolution base grid). The clean surface was the resampled refinements; the noisy surface was the base grid. These are different surfaces, so the difference was meaningless.

**Why Severity Varied:** The discrepancy between the base grid and the resampled refinements depends on the VR structure. When clean and noisy structures are similar (H13739), the error is small. When they differ a lot (H14116: clean 64m base vs noisy 32.73m base), the error is large. This is why one survey looked "mostly fine" and masked the bug for weeks.

**How It Was Caught:** CARIS-derived difference exports (computed at native resolution by purpose-built hydrographic software) showed normal symmetric noise (median ~0, ~50/50 direction split, sub-meter typical magnitudes) for all three surveys. The BAG pipeline showed offsets of tens of meters and 99% one-sided splits. The mismatch between the two was the signal that the pipeline, not the data, was broken.

**The Diagnostic That Isolated It:** A separate script compared the resampled noisy surface alone against a CARIS export of the same surface. They matched with correlation -0.9999 (differing only by sign convention). This proved the resampling was correct and the bug was specifically in the warp re-opening the raw BAG.

**The Fix:** Warp the already-loaded, correctly-resampled in-memory grid instead of re-opening the raw BAG. Write the loaded grid to a temporary GeoTIFF, warp that, so both surfaces stay in the same interpretation through the difference.

**General Principle:** When differencing two surfaces, both must be loaded through the exact same path with the exact same options. Any divergence in how the two are read (resampling mode, interpolation, datum handling, sign convention) shows up as fake signal in the difference. The safest pattern is: load both with one function, confirm they are in the same representation, and only then subtract. If alignment or resampling is needed, operate on the already-loaded arrays rather than re-reading the source files with different settings.

**Validation Practice Worth Keeping:** When an external tool (CARIS here) can produce the same quantity, use it as ground truth to validate the pipeline. The discrepancy between pipeline output and CARIS output is what made this bug visible. A single-source pipeline with no external check would have trained on corrupted targets indefinitely. The 50/50 vs 99/1 direction split was the most diagnostic single number; a healthy noise-removal difference is close to symmetric, and a wildly asymmetric split is a red flag for a systematic processing problem rather than real noise.

**Note on Sign Convention:** This investigation incidentally revealed that GDAL reads these BAG depths as negative-down while CARIS exports positive-down. This does not affect the pipeline because both surfaces are loaded through the same GDAL path and the sign cancels in the subtraction. But it is worth knowing when comparing pipeline values against CARIS values directly: a near-perfect negative correlation between two surfaces that should be identical means a sign convention difference, not a data problem.

---

## Recommended Training Workflow

### Classification Mode (V9, existing approach)

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

### Regression Mode (V10, new approach)

```bash
# Generate ground truth in regression mode
python scripts/prepare_ground_truth.py \
    --clean "clean_survey.bag" \
    --noisy "noisy_survey.bag" \
    --output-dir "ground_truth/" \
    --regression-mode \
    --no-offset   # only if both surfaces share the same vertical datum

# Train with regression-mode files
python scripts/train.py \
    --ground-truth-dir "ground_truth/" \
    --output-dir "model_output/" \
    --epochs 30 \
    --device cuda \
    --tile-size 256 \
    --batch-size 2

# Verify in output:
#    - "Loaded {survey}_regression.tif in regression mode"
#    - "Training mode: regression"
#    - Progress bar shows MAE instead of accuracy
#    - Training loss should decrease over epochs
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
| CUDA OOM | Out of memory error | Reduce batch-size to 2 or reduce tile-size |
| Boundary contamination | Classifications follow tile edges | Use boundary-aware feature computation (V7+) |
| Low correction magnitudes | Model flags noise but corrections too small | Use local_std correction normalization (V9+) |
| Near-zero noise in GT | Noise doesn't propagate to grid surface | Verify noise visible in gridded BAG, not just point cloud |
| All-seafloor training data | Pushes model toward majority-class collapse | Only use pairs with 10-40% noise in gridded surface (classification mode) |
| Noise over-prediction | Detection rate (~35%) far exceeds ground truth (~19%) | Add geographically diverse training data; track precision/recall separately; tune classification threshold |
| Pervasive cell-to-cell differences | 60-99% noise percentage from `prepare_ground_truth.py` | Expected behavior for CUBE re-grid pairs; use `--adaptive-threshold` or `--regression-mode` |
| Variable offsets across sub-files | Different median offsets in different sub-files of same survey | Datum mismatch with spatial variation; check whether sub-files are in same datum |
| 50/50 shoal/deep split | Not actually a problem | Expected for CUBE re-grid differences; only a quality signal for surfaces produced by post-grid editing |
| Fixed threshold doesn't fit | Adaptive threshold varies widely across surveys (0.12m to 6.73m) | Use `--adaptive-threshold` per pair, or shift to regression mode for consistency |
| Classification head over-predicts | False positives near threshold boundary | Consider regression mode; eliminates threshold-induced false positives |
| Huge one-sided difference in VR pair | 99% deep (or shoal) direction, tens-of-meters phantom offset | Warp re-opened raw BAG with wrong VR interpretation; fixed in `warp_grid_to_reference` (warp the loaded resampled grid, not the raw file) |
| VR difference magnitudes inflated | Mean correction several times larger than CARIS | Same root cause as above; severity is small when clean/noisy VR structures are similar, large when they differ |
| Pipeline difference disagrees with CARIS | Pipeline median offset tens of meters, CARIS near zero | Validate against CARIS export; near-perfect negative correlation between surfaces means a sign convention difference, not a data problem |

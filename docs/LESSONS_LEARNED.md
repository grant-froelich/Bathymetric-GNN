# Lessons Learned: Real-World Training with Clean/Noisy Survey Pairs

This document captures practical lessons from training the Bathymetric GNN on real survey data (Seward, Alaska multibeam surveys and Pacific Islands surveys). It complements the theoretical documentation in HOW_IT_WORKS.md and TRAINING_PLAN.md.

*Document Version: 4.0*
*Updated: June 2026*
*Based on: Seward training data, V1-V9 runs, Pacific Islands/Pacific NW/Alaska ground truth, V10 regression mode, VR warp fix, first cross-geography generalization result, depth-convention fix, V11 retrain and fp32/bf16 comparison*

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

**Note on Sign Convention (CORRECTED 2026-06-09):** This investigation incidentally revealed that GDAL reads these BAG depths as negative-down while CARIS exports positive-down. The original version of this note claimed the difference "does not affect the pipeline because the sign cancels in the subtraction." That was true for MAGNITUDES only, and the claim concealed a critical bug: the sign does NOT cancel for direction semantics. Every direction-sensitive component (the 3x shoal-safety weighting, the hazard metrics, the shoal/deep split) assumed positive-down and was therefore inverted by the negative-down data. See Lesson 20. The diagnostic observation stands: a near-perfect negative correlation between two surfaces that should be identical means a sign convention difference, not a data problem.

---

### 18. Aggregate Error Metrics Can Be Dominated by a Tiny Cell Population

**Observed (June 2026):** On the first cross-geography test (model trained on Pacific Islands + Pacific NW, evaluated on unseen Alaska), the aggregate looked alarming: 19m MAE, 72m recovery RMSE, -18m mean error. Read literally, that says the model's corrections would degrade an unseen survey and are not usable.

**What the spatial analysis revealed:** A per-cell error map showed the worst 1% of cells accounted for 87.6% of total squared error, and the worst 5% for 96.5%. The other 95% of cells contributed only 3.5%. The model was handling almost the entire survey well and failing on a small scattered population of large-magnitude correction cells (true correction >=10m, mostly deeper water). QGIS confirmed these were isolated cells and tiny clusters, not a contiguous failure region.

**Why This Matters:** MAE and especially RMSE are squared- or magnitude-weighted, so a small number of large errors dominate them. An aggregate metric that looks like wholesale failure can actually be "works well on 95% of cells, fails on a hard 5%." The two cases call for completely different responses: wholesale failure means the approach is wrong; localized failure means the approach works and a specific hard subset needs more data. You cannot tell which from the aggregate number alone.

**The Diagnostic Pattern:** When an aggregate metric looks bad, before concluding the model failed, check error concentration. Sort cells by squared error, compute the cumulative fraction of total error against the cumulative fraction of cells. If a small fraction of cells carries most of the error, the failure is localized; investigate that population specifically (what depth, what correction magnitude, what location) rather than treating the whole result as a failure. `scripts/spatial_error_map.py` does this and also writes a per-cell error GeoTIFF for visual inspection.

**Corollary on Recovery RMSE:** Recovery RMSE (corrected surface vs clean reference) is a useful single number but inherits the same sensitivity. A 72m recovery RMSE driven by the worst 1% of cells does not mean "applying corrections degrades the surface everywhere"; it means a few cells get large wrong corrections while the bulk is fine. Report the concentration alongside the RMSE so the number is not misread.

**General Principle:** Always look at the distribution of errors, not just their summary statistics, before drawing conclusions about model quality. A mean or RMS hides whether error is spread evenly or concentrated, and that distinction often changes the conclusion entirely.

---

### 19. Judge Navigation Safety on a TVU-Budget Breach Rate, Not a Sign-Count Hazard Rate

> **CORRECTION (2026-06-09), RESOLVED (2026-06-15):** the measurements in this
> lesson predate the sign-convention fix (Lesson 20). Every direction label in the
> table below is inverted: "hazardous" counted the safe direction, "shoal-target"
> rows are deep-spike cells, and vice versa. The lesson's core argument (budget-aware
> metrics over sign counts) is unaffected and the magnitudes are real, but its bf16
> verdict was wrong. The corrected V11 re-measurement (end of this lesson) reverses
> it: bf16 is NOT defensible on the deliverable path.

**Observed (June 2026):** bf16 mixed-precision training (a ~5x speedup, 5.51 -> 1.07 s/it) appeared to regress safety. On the raw `hazardous_error_rate`, the rate rose 2-3x on the deep and unseen surveys, including the shoal-target subset (Alaska shoal 5.08% -> 15.27%, deep E00269 shoal 7.82% -> 11.03%). Read literally, that blocks bf16 for a navigation-safety deliverable, even though bf16's MAE was better at all three locations.

**What the budget-aware metric revealed:** `hazardous_error_rate` counts any dangerous-direction error (corrected deeper than truth) at any magnitude, with no reference to the uncertainty the survey is actually permitted. Re-counting a cell as a breach only when its dangerous-direction error exceeds the allowable TVU = sqrt(a^2 + (b*depth)^2) at that cell's depth (NOAA HSSD General 1 shallow, General 2 deep/Alaska) collapsed the alarm:

| Survey | subset | raw hazard (fp32 -> bf16) | TVU breach (fp32 -> bf16) |
|--------|--------|---------------------------|---------------------------|
| Shallow E00269 | shoal-target | 0.00% -> 0.00% | 0.00% -> 0.00% |
| Shallow E00269 | deep-target | 5.07% -> 31.10% | 0.07% -> 0.62% |
| Deep E00269 | shoal-target | 7.82% -> 11.03% | 0.00% -> 0.01% |
| Deep E00269 | deep-target | 28.96% -> 44.53% | 0.03% -> 0.12% |
| Alaska H14116 | shoal-target | 5.08% -> 15.27% | 0.00% -> 0.00% |
| Alaska H14116 | deep-target | 17.38% -> 48.82% | 0.05% -> 0.04% |

The shoal-target dangerous-breach rate (the shoal-preservation number) stayed at ~0% for both precisions. The deep-target dangerous-breach rate rose with bf16 but stayed under 0.7% everywhere. The apparent 2-3x raw regression was almost entirely sub-budget noise.

**Why This Matters:** Most cells have a near-zero true correction, so a sign-count hazard rate is dominated by sub-meter (often sub-decimeter) directional flips that are far smaller than the survey's own allowable TVU. The raw metric conflates "wrong direction" with "dangerously wrong," and once TVU is applied the two differ by one to two orders of magnitude. For a navigation-safety model, the metric that gates a go/no-go decision must be the one the deliverable is actually certified against.

**Direction Semantics (stated precisely so this is not re-confused):** `hazardous` is `error < 0`, meaning the corrected surface is deeper than truth, i.e. less clearance than exists, i.e. dangerous. The `_shoal` and `_deep` breach fields partition by the *true correction's* direction, not the error's; both are dangerous-direction rates. The shoal-target subset is the one that bears on shoal preservation.

**The Diagnostic Pattern:** When a single-run comparison shows aggregate error (MAE) and a thresholded safety metric moving in *opposite* directions, suspect the threshold/metric definition before concluding a real regression. Here bf16 improved MAE everywhere while the raw hazard rate worsened; the budget-aware metric resolved the contradiction.

**Coefficients:** Use what the survey is certified to. NOAA HSSD rounds S-44's depth term (General 1 uses b=0.01 vs S-44 1a's 0.013; General 2/3 uses b=0.02 vs S-44 Order 2's 0.023), so using S-44 values for a NOAA survey makes the budget slightly too generous, materially so in deep water. The applicable OCS Quality Metric is set in the Project Instructions, not derived from depth.

**General Principle:** A sign-count safety metric over-reports because it ignores the allowed uncertainty. The budget-aware framing is the right one and it carries forward to V11 unchanged.

**Corrected re-measurement (V11, 2026-06-15):** repeating this comparison on the
fixed pipeline (positive-down ground truth, correct direction labels) overturns the bf16
verdict. The shoal-target dangerous breach is no longer ~0: it is 0 / 114 / 515 cells in
fp32 at shallow / deep / Alaska, and bf16 raises it to 0 / 1,539 / 924 (13.5x worse at
deep, 1.8x at Alaska). bf16 also no longer wins MAE; fp32 is lower at all three. The
earlier "bf16 improved MAE everywhere and shoal breach stayed at zero" reading was an
artifact of the inverted signs and the pre-fix ground truth. Verdict now: fp32 for the
qualified release (the shoal-breach number is the go/no-go and bf16 fails it), bf16 for
experimentation only. This is one paired run; the bar for ever moving bf16 onto the
deployed path is shoal-breach parity across a few paired runs. See Lesson 21 for why
reduced precision hits the safety tail specifically.

---

### 20. Normalize Data Conventions at the Boundary; a Documented-but-Unhandled Convention Is a Time Bomb

**Observed (June 2026, full repo scrub):** Ground-truth bands stored GDAL
elevation (negative-down: -10604 to -116 m, verified during the Lesson 17
investigation) while every loss, metric, and document assumed positive-down
depth. Under the actual data, `error > 0` is the dangerous direction
(corrected surface deeper than truth), so the entire direction-sensitive stack
was inverted since regression mode began:

| Component | Intended | Actually did (pre-fix) |
|---|---|---|
| RegressionLoss 3x weight | penalize dangerous 3x | penalized SAFE 3x |
| ShoalSafetyLoss (V9) | weight shoal spikes 3x | weighted DEEP spikes 3x |
| metrics hazardous rate | count dangerous errors | counted SAFE errors |
| metrics shoal/deep split | shoal-spike cells | deep-spike cells |

**The observational fingerprint:** `recovery_mean_error` was negative in every
evaluation ever run (-0.96 to -23.8 m). Since recovery error equals -error,
every model had settled on the lightly-penalized side of the asymmetric loss,
which under the real convention is the deeper-than-truth side: the safety loss
was actively creating the dangerous bias it existed to prevent. A consistently
one-sided signed bias metric is worth interrogating against the loss's
direction semantics.

**The trap that hid it:** the convention difference WAS discovered (Lesson 17)
and documented as harmless because "the sign cancels in the subtraction." It
cancels in magnitudes; it inverts directions. A known convention mismatch that
is documented but not normalized in code is worse than an unknown one, because
the documentation note inoculates future readers against suspicion.

**The fix (2026-06-09):** one convention, enforced at the boundary.
`BathymetricLoader` normalizes every source to positive-down on load
(auto-detected by median sign, overridable), ground truth records the
convention in metadata, and `GroundTruthDataset` refuses files whose median
valid depth is negative so stale pre-fix data cannot enter a run. All
checkpoints trained pre-fix embody the inverted objective; V11 is the first
version trained with the safety asymmetry pointing the right way.

**Confirmed (V11, 2026-06-15):** the retrain validates the fix. V11
`recovery_mean_error` is negative at all three validation surfaces (-0.18 to
-6.61 m) and the hazardous rate is under 50% everywhere, meaning the corrected
surface now sits shallower than truth on average: the model errs to the safe
side. Note the sign subtlety that the pre-fix dashboard prediction got wrong. It
expected `recovery_mean_error` to flip *positive* as the success signal, but the
load-boundary negation flipped the sign-to-meaning mapping too: pre-fix (negative-
down) a negative value meant dangerous, post-fix (positive-down) a negative value
means safe. So success shows as the value staying negative, not flipping. A
still-broken model would show positive `recovery_mean_error` and a hazardous rate
above 50%.

**General Principles:**
- Normalize external-data conventions at the load boundary, once, and make
  every downstream component entitled to assume the normalized form. Scattered
  per-component assumptions about conventions cannot be audited.
- Guard the assumption in code, not in prose: a cheap runtime check (median
  sign) converts a silent inversion into an immediate, explained failure.
- When a signed bias metric is consistently one-sided across all runs and
  regimes, check it against the loss's direction semantics before accepting it
  as a model property.
- Magnitude metrics cannot detect direction inversions. Any pipeline whose
  purpose is directional (safety asymmetry) needs at least one end-to-end
  direction test: construct a tiny synthetic case with a known dangerous error
  and assert the loss penalizes it more, not less.

---

### 21. Reduced Precision Erodes an Asymmetric Safety Loss at the Tail, Not the Mean

**Observed (V11, 2026-06-15):** bf16 and fp32 were trained as a paired run on
identical data and split, then evaluated with correct direction semantics. bf16
is the more conservative model *on average* (its `recovery_mean_error` is more
negative at all three surfaces, so it sits further to the safe side in the mean),
yet it produces far more dangerous-direction shoal-target breaches: 0 / 1,539 /
924 cells vs fp32's 0 / 114 / 515 at shallow / deep / Alaska (13.5x worse at deep).
A model can be safer in the mean and more dangerous in the tail at the same time.

**Why this happens:** the safety mechanism here is the 3x asymmetric penalty on the
dangerous direction, and it does its work at near-zero corrections, where most cells
live and where the safe-vs-dangerous decision is a small signed quantity. bf16's
8-bit mantissa (~2-3 significant decimal digits) cannot represent that fine signed
gradient precisely, so it rounds it away. The penalty still shifts the bulk
distribution to the safe side (hence the more-conservative mean), but the rounding
widens both tails, and the dangerous tail is the one that breaches the TVU budget.
Precision loss does not bias the model dangerous; it blunts the instrument that
keeps the dangerous tail thin.

**Why the mean misleads here:** `recovery_mean_error` answers "which side is the
model on, on average," which is the wrong question for navigation safety. A shoal
hazard is a tail event: one cell left dangerously shallow-removed is a charted
depth that is wrong in the direction that grounds vessels, regardless of how
conservative the surrounding 10,000 cells are. Always gate the safety decision on
the breach tail (shoal-target TVU breach), and treat a conservative mean as
necessary but not sufficient.

**General Principles:**
- For any loss whose value comes from an asymmetry applied to small signed
  quantities, suspect that low-precision arithmetic will degrade the asymmetry
  before it degrades the aggregate fit. Verify the safety tail under the deployment
  precision, not just the loss curve or MAE.
- Separate the mean-bias metric from the tail metric in evaluation and decide on the
  tail. A single conservative-looking summary statistic can hide a worse tail.
- Decouple training precision from the qualification metric. Use the fast precision
  (bf16) for experimentation where aggregate metrics drive the decision; train the
  shipped model in the precision that wins the tail metric, since inference precision
  is a separate choice and training the release in fp32 costs nothing per survey.

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
| Raw hazard rate inflated by a precision change | `hazardous_error_rate` up 2-3x while MAE improves | Sign count over-reports; use the TVU-breach rate (dangerous error exceeding TVU at depth). Sub-budget directional flips do not count |
| Direction semantics inverted by data convention | recovery_mean_error consistently negative across all runs; "safety" bias worsens | Normalize convention at load (positive-down), guard with a median-sign check, add an end-to-end direction test (Lesson 20) |
| S-44 coefficients used for a NOAA survey | TVU budget slightly too generous, esp. in deep water | Use HSSD OCS Quality Metric coefficients (General 1 b=0.01, General 2/3 b=0.02); the metric comes from Project Instructions, not depth |

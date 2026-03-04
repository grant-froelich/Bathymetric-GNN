# Bathymetric GNN -- Training Performance Tracker

**V1-V9 | Seward, Alaska | VR BAG Noise Detection | Updated 2026-03-02**

---

## Version Overview

| Version | Key Change | Val Accuracy | Noise Det. | Confidence | Auto-Corr | Status |
|---------|-----------|-------------|-----------|-----------|----------|--------|
| V1 | Baseline (7 features) | 47.7% | -- | -- | -- | Synthetic |
| V2 | +uncertainty (8 features) | 48.6% | -- | -- | -- | Synthetic |
| V3 | +shoal safety loss | -- | 0.6% | 0.674 | 2,203 | Too conservative |
| V4 | Manual 2x noise weight | -- | 96.8% | 0.455 | 4,612 | :x: All-noise |
| V5 | No class weights (bug) | ~67% | 0.0% | 0.967 | 0 | :x: All-seafloor |
| V6 | Auto class weights | ~61% | 3.3% | 0.713 | 0 | :warning: Boundary bug |
| **V7** | **Boundary-aware features** | **~72%** | **34.8%** | **0.825** | **11,790** | :star: Best detection |
| V8 | Dynamic Huber delta | = V7 | = V7 | = V7 | = V7 | No change |
| **V9** | **local_std correction norm** | **~72%** | **34.8%** | **0.825** | **12,823** | :star: Best corrections |

---

## Phase 1: Synthetic Noise Training (V1, V2)

Both models oscillated around 48% validation accuracy (near random). Adding uncertainty as the 8th feature gave negligible improvement. Synthetic noise did not capture real acoustic artifact characteristics.

![V1 vs V2 Loss](images/01_v1v2_loss.png)

![V1 vs V2 Accuracy](images/02_v1v2_acc.png)

---

## Phase 2: Ground Truth Training (V5-V9)

Switched from synthetic noise to real clean/noisy survey pairs from Seward, Alaska. Four VR BAG pairs with noise percentages ranging from 16% to 34%.

### Loss Curves by Version

![V5-V9 Loss Grid](images/03_v5v9_loss_grid.png)

Key observations per version:

- **V5 (purple):** Train loss decreases smoothly but val loss diverges wildly. Model collapsed to all-seafloor prediction (67% accuracy = seafloor proportion). Root cause: missing class weights.
- **V6 (green):** Val loss diverges early. Model learned boundary artifacts instead of real noise. Root cause: `uniform_filter` bleeding nodata into features.
- **V7 (yellow):** Most stable training. Val loss stays bounded. First version to genuinely classify noise spatially. Boundary-aware masked statistics fixed the V6 bug.
- **V9 (teal):** Higher absolute loss values (expected, since correction targets are now in std dev units). Training dynamics similar to V7 because classification head is unchanged.

### Validation Accuracy Comparison

![Validation Accuracy V5-V9](images/04_val_acc_compare.png)

The red dashed line marks the seafloor proportion (75%). V5 sits right at the seafloor proportion, confirming majority-class collapse. V7 and V9 land slightly below it (~72%), which is expected: they trade some seafloor accuracy for genuine noise detection (34.8%). The accuracy dip below 75% reflects false positives (seafloor misclassified as noise), but the model is doing useful work because it's actually finding noise cells. The V6 dip to ~35% (epoch 10) shows the boundary artifact model periodically "losing" its signal.

### Validation Loss Comparison

![Validation Loss V5-V9](images/05_val_loss_compare.png)

Validation loss diverges from training loss after approximately 5 epochs in every version. This is the persistent overfitting signature from training on a single geographic area (Seward only). Early stopping triggers around epochs 15-25.

---

## Inference Results (Seward 1of4 Unclean)

### Noise Detection Rate

![Noise Detection by Version](images/06_noise_detection.png)

V4 (96.8%) flagged everything as noise -- a failed overcorrection from V3's conservatism. V5 (0.0%) swung to the opposite extreme. V7 and V9 both land at 34.8%, matching the ground truth distribution.

### Mean Confidence

![Confidence by Version](images/07_confidence.png)

V5's 0.967 confidence is misleading -- the model was "confidently wrong" about everything being seafloor. V7/V9's 0.825 reflects genuine uncertainty where appropriate.

### Auto-Corrections Applied

![Auto-Corrections by Version](images/08_auto_corrections.png)

V9 applies 12,823 auto-corrections vs V7's 11,790 (+8.7%). The increase comes from larger correction magnitudes shifting cells near the confidence threshold.

### Best Validation Loss

![Best Validation Loss](images/11_best_val_loss.png)

Note the gap between synthetic (V1/V2, ~0.84) and ground truth (V5-V9, 1.4-1.9). Ground truth loss is higher because the real noise classification problem is harder than synthetic noise. The V5 anomaly (lowest GT loss at 1.483) reflects the model "solving" the problem by predicting one class.

---

## Correction Normalization (V9)

V7 correctly identified noise locations but predicted 0.4m corrections where 70m was needed. Huber loss with delta=1.0 gives identical gradient magnitude for errors above 1m, so the model had no incentive to predict large corrections.

V9 normalizes correction targets by per-node local surface variability (local_std). The model learns in relative units (standard deviations), and predictions are denormalized at inference by multiplying by local_std.

![Correction Magnitude V7 vs V9](images/10_correction_comparison.png)

| Stat | Raw | Normalized |
|------|-----|-----------|
| Mean |correction| | 0.302m | 0.676 sigma |
| Max |correction| | 33.10m | 50.0 sigma (capped) |
| 95th percentile | -- | 0.997 sigma |
| Noise cells | 537,940 | -- |
| Normalization floor | 0.01m | -- |
| Cap | -- | +/-50 sigma |

---

## Training Data

### Noise Percentage by Survey Pair

![Data Pair Noise Percentages](images/09_data_pairs.png)

The orange dashed line marks the minimum useful noise threshold (~10%). The four Seward pairs (green) all fall well above it. The three new pairs (red) have effectively zero grid-level noise and were rejected.

### Active Training Pairs (Seward, Alaska)

| Survey | Type | Valid Cells | Noise Cells | Noise % | Role |
|--------|------|------------|------------|---------|------|
| Seward 1of4 | VR | 1,803K | 551K | 30.5% | Training |
| Seward 2of4 | VR | 1,740K | 281K | 16.2% | Training |
| Seward 3of4 | VR | 1,882K | 636K | 33.8% | Training |
| Seward 4of4 | VR | 1,626K | 306K | 18.8% | Validation |

**Combined:** ~5.3M seafloor cells, ~1.8M noise cells (~75/25 split)

### Rejected Pairs (2026-03-02)

| Survey | Type | Location | Valid Cells | Noise Cells | Noise % | Reason |
|--------|------|----------|------------|------------|---------|--------|
| H13532 | SR 1m | Florida (river) | 456K | 4 | 0.00% | No grid-level noise |
| H14190 | VR | Alaska (coastal) | 30,250K | 149 | 0.00% | No grid-level noise |
| F00889 | SR 0.5m | Norfolk (river) | 23,573K | 1 | 0.00% | No grid-level noise |

The gridded BAG surfaces are nearly identical between clean and dirty versions. CUBE's robust gridding algorithm likely already rejected the outlier soundings during surface generation. This model operates on gridded surfaces, so noise that only exists in the point cloud cannot be detected or learned from.

---

## Architecture

| Component | Detail |
|-----------|--------|
| Model | Graph Attention Network (GAT) |
| Layers | 4 |
| Hidden channels | 64 |
| Parameters | 182K |
| Node features (8) | depth, local mean, local std, gradient magnitude, gradient direction, curvature, uncertainty, boundary distance |
| Edge features (3) | distance, depth difference, slope angle |
| Output heads | classification (3-class), confidence (0-1), correction (meters, normalized by local_std) |
| Loss | Weighted cross-entropy + shoal safety asymmetric + Huber (on normalized corrections) |

---

## Data Acquisition Plan (2026-03-04)

22 surveys identified across 8 regions to address the geographic diversity gap. Clean BAGs will be downloaded from NCEI. Processed (pre-cleaning) data requested from the NCEI archive to produce noisy BAGs for training pairs. One survey (E00269) is available locally and can be processed immediately.

![Acquisition Plan](images/12_acquisition_plan.png)

### Target Surveys by Region

| Region | Count | Registry Numbers | Environment |
|--------|-------|-----------------|-------------|
| Gulf Coast Shallow (TX/LA/MS/FL) | 3 | H13818, H13651, H13837 | Shallow saltwater, warm water refraction, post-storm dynamics |
| Southeast Atlantic (SC/GA) | 2 | H13851, F00881 | Sandy, tidal, dynamic bottom |
| Mid-Atlantic (MD/VA/NC) | 3 | H13762, H13804, H13750 | Estuarine, mixed bottom, current-affected |
| Northeast (NY/CT/RI) | 1 | H13927 | Rocky/sand mixed, different acoustic environment |
| Great Lakes | 2 | H13940, H13943 | Freshwater, glacial sediment, shallow, thermocline refraction |
| Pacific NW (WA/OR) | 2 | H14070, H13847 | Rocky, deep, cold saltwater |
| Alaska | 6 | H13774, F00886, H14116, H13914, E01093, H13695 | Mixed: shallow flat (set line spacing), Bering Sea/North Slope (trackline), plus standard multibeam |
| Pacific Islands | 3 | H13739, H13735, E00269 | Coral/volcanic substrate, warm shallow water, Northern Mariana Islands |

### Acquisition Status

| Source | Surveys | Status |
|--------|---------|--------|
| NCEI archive (clean BAGs) | 21 | Download directly |
| NCEI archive (processed data) | 21 | Requested, awaiting delivery (days to weeks) |
| Local data (E00269, N. Mariana Islands) | 1 | Available now, ready to process |

### Processing Plan

**Immediate:** Process E00269 locally. Run QGIS difference-layer check first.

**As data arrives:** Prioritize one survey per region before processing all surveys from any single region. This reveals early if any region consistently produces unusable pairs. Suggested first batch: one Gulf Coast, one Great Lakes, one Pacific Islands.

**Quality gate:** For each pair, subtract surfaces in QGIS raster calculator before running `prepare_ground_truth.py`. If the difference layer is uniformly near zero, reject the pair. Target 10-40% noise in the gridded surface.

**Expected yield:** At ~30-50% attrition (near-zero grid noise), expect 11-15 usable pairs. Combined with 4 existing Seward pairs, that would give 15-19 total training pairs from 8+ distinct environments.

### Training Value by Region

| Region | Key Diversity Contribution |
|--------|---------------------------|
| Gulf Coast | Shallow water noise regime, warm water refraction artifacts |
| SE / Mid-Atlantic | Sandy/estuarine bottom, tidal current effects |
| Northeast | Rocky New England coast, different from Alaska rocky |
| Great Lakes | Freshwater acoustics, glacial sediment, no tidal corrections |
| Pacific NW | Deep cold saltwater, volcanic/rocky substrate |
| Alaska (non-Seward) | Validates generalization within Alaska; trackline and set-line-spacing survey types add acquisition geometry diversity |
| Pacific Islands | Coral/volcanic, warm shallow, most acoustically distinct from Seward |

---

## Persistent Issues

**Overfitting:** Validation loss diverges from training loss after ~5 epochs in every ground truth run (V5-V9). All training data is from Seward, Alaska. The model memorizes location-specific patterns rather than learning generalizable noise signatures. The data acquisition plan above is the primary mitigation.

**Classification plateau:** ~72% overall accuracy is slightly below the 75% seafloor proportion. The model trades some seafloor accuracy for noise detection (34.8%), which is useful but indicates room for improvement on both fronts. Both false positives (seafloor flagged as noise) and false negatives (noise missed) are present.

**Correction magnitude gap:** V9 corrections are up to 32m larger than V7, but still don't fully recover the clean reference surface.

---

## Next Steps

1. :star: **Process E00269 (N. Mariana Islands)** -- Available locally. Run QGIS difference check, then `prepare_ground_truth.py` if viable. First non-Seward training pair.
2. **Process archive data as it arrives** -- One per region first, QGIS pre-check on each, reject pairs with <1% noise.
3. **Train V10 with multi-region data** -- Incorporate ShoalSafetyLoss, local_std correction normalization, and geographic diversity.
4. **Sounding density feature** -- Per-cell sounding count from point cloud data would be the strongest noise discriminator. Single-hit cells in the surface are almost certainly noise.
5. **Validate V9 corrections** -- Quantify how close V9 corrections come to recovering the clean surface at known noise locations.
6. **Adaptive Huber delta** -- Options documented in `training/losses.py` for future implementation once diverse data is available.

---

*Dashboard v2.1 | March 2026*

# Changelog

## 2026-05-28 - Per-Cell Resolution Feature (log_footprint)

### Added Resolution Conditioning Feature
- Added `log_footprint` node feature to `data/graph_construction.py`
- Encodes each cell's resolution (footprint) as log2 of the footprint in meters
- For SR surveys: constant per file (log2(4)=2, log2(8)=3, log2(128)=7, log2(256)=8)
- For VR surveys: will vary per cell once native resolution is preserved through loading (future work)
- Rationale: lets a single model condition its correction behavior on scale rather than needing separate models per resolution regime
- log2 chosen so equal resolution ratios map to equal feature distances (4->8 is the same step as 128->256)
- `FOOTPRINT_FLOOR = 0.1` guards against log of zero on malformed resolution values
- Input channels increased from 8 to 9; existing checkpoints not loadable (retrain required)

### Controlled Comparison Result (E00269, same train/val split)
Same data and split as the prior baseline; only the resolution feature was added.

Best validation loss improved from 1.64 to 1.31.

Shallow water (2of6, 8m) showed large improvement:
| Metric | Baseline | With feature |
|--------|----------|--------------|
| Overall MAE | 1.99m | 0.87m |
| MAE <0.1m bucket | 1.39m | 0.52m |
| MAE 0.1-1m bucket | 1.86m | 0.79m |
| MAE 1-10m bucket | 3.53m | 1.74m |
| Recovery RMSE | 3.40m | 1.56m |
| Recovery mean error | -2.16m | -0.94m |
| Hazardous (shoal) | 0.00% | 0.00% |

Deep water (5of6, 128m) showed marginal improvement only:
| Metric | Baseline | With feature |
|--------|----------|--------------|
| Overall MAE | 28.09m | 26.59m |
| Recovery RMSE | 46.81m | 40.05m |
| Recovery mean error | -23.76m | -23.53m |

Deep water per-bucket MAE remained flat (~24m across all magnitude buckets), indicating the model still outputs a default magnitude rather than discriminating. This points to deep water being a data limitation (one 128m survey) rather than something the feature alone can fix.

### Interpretation
- The feature helped where the hypothesis predicted: the shallow regime with enough signal to learn from
- Shallow MAE roughly halved with no cost to shoal safety (still 0.00% shoal hazard)
- Deep water still needs more training examples, not architecture changes
- Supports staying with a single conditioned model rather than splitting by regime
- One side effect to watch: deep-direction hazard rate in shallow water rose (7.32% to 31.28%); shoal protection unaffected

---

## 2026-05-21 - Huber Delta Computation Fixed for Normalized Corrections

### Bug Identified in V10 Multi-File Training
- First V10 training run on 5 E00269 sub-files reported Huber delta of 281.7m
- Training proceeded but with erratic validation loss (range 5.98 to 18.71 across 9 epochs)
- Root cause: `_compute_training_stats` was computing delta from raw correction magnitudes in meters
- Model trains on normalized corrections (divided by local_std, clipped to +/-50 std-devs)
- A delta of 281 put the Huber loss in pure linear mode for the entire run, effectively becoming MAE
- The quadratic gradient signal that Huber provides for small errors was completely absent

### Fix
- Updated `_compute_training_stats` to sample 50 random tiles from the dataset
- Builds graphs for sampled tiles and collects normalized correction targets
- Computes 95th percentile from the same normalized values the model sees during training
- Adds 99-second startup cost but eliminates unit-mismatch risk
- Typical post-fix delta on E00269 data: 3-10 std-devs

### Validation Strategy Improvement
- Previous runs used single-file validation (sub-file 3of6 alone), which only validated shallow water performance
- New approach uses `--val-surveys` flag to point at a separate folder with multiple files
- Current recommended split: train on 1of6 + 3of6 + 4of6 + 6of6; validate on 2of6 (shallow) + 5of6 (deep)
- Validation now spans both shallow and deep regimes, giving a more honest signal about generalization

### Documentation Updates
- New "Huber Loss and the Delta Parameter" section in HOW_IT_WORKS.md
- Lesson 15 added to LESSONS_LEARNED.md
- Troubleshooting entries added to QUICK_REFERENCE.md

---

## 2026-05-20 - V10 First Training Run Operational

### First V10 Training Run on Regression-Mode Data
- Trained V10 on E00269 sub-file 1of6 (regression-mode ground truth)
- Parameters: 5 epochs, batch size 2, tile size 256, 91 tiles from 1 survey
- Training loss decreased steadily: 0.79 -> 0.74 -> 0.71 -> 0.70 -> 0.69
- MAE (normalized std-dev units) stabilized at ~0.83
- No NaN values, no divergence, gradient flow healthy
- Pipeline confirmed working end-to-end in regression mode

### E00269 Sub-Files Processed (Regression Mode)
| Sub-file | Resolution | Valid Cells | Mean Correction | Max Correction | Offset Removed |
|----------|------------|-------------|-----------------|----------------|----------------|
| 1of6 | 4m SR | 918,831 | 0.30m | 16.49m | -0.51m |
| 2of6 | 8m SR | 356,601 | 0.53m | 88.00m | -0.20m |
| 3of6 | 8m SR | 3,214,339 | 0.50m | 605.64m | -0.41m |
| 4of6 | 128m SR | 11,390,629 | 83.84m | 4934.40m | +5.91m |
| 5of6 | 128m SR | 1,118,639 | 6.77m | 775.70m | +0.42m |
| 6of6 | 256m SR | 59,793 | 17.97m | 552.44m | -0.15m |

- Correction magnitudes scale with resolution as expected; local_std normalization handles cross-regime training
- Variable offsets across sub-files were initially flagged as datum issues; confirmed all are MLLW; offsets are real signal in survey datum vs MLLW separation
- 50/50 shoal/deep split is normal for CUBE re-grid differences (not a quality warning, contrary to earlier interpretation)

### V10 Architecture Shift: Regression Mode
- Added `--regression-mode` flag to `prepare_ground_truth.py`
- Output bands changed in regression mode: band 1 = valid_mask (1=valid, 0=invalid), band 2 = correction target (meters, continuous, no threshold applied)
- Output file naming: `{survey}_regression.tif` (vs `{survey}_ground_truth.tif`) so both modes can coexist
- Added `RegressionLoss` class to `losses.py` with asymmetric Huber penalty
  - Shoal safety baked into loss direction: predictions leaving the corrected surface deeper than reality penalized 3x
  - Sign math: `corrected = noisy - predicted_correction`; error < 0 means corrected depth > true depth = navigation hazard
  - Applies to every valid cell, not just cells flagged as noise
- `BathymetricGNNLoss.forward()` dispatches on `targets['mode']`; existing classification path unchanged
- `GroundTruthDataset` auto-detects mode from band 1 description; tiles carry mode flag
- Training loop tracks MAE in regression mode, accuracy in classification mode
- Backwards compatible: existing classification workflow works exactly as before

### Code Changes
- `scripts/prepare_ground_truth.py`: +312/-77 lines
  - `--adaptive-threshold` flag (Otsu's method on log-scaled absolute differences)
  - `--no-offset` flag (skip median offset removal when both surfaces share a datum)
  - `--regression-mode` flag (skip thresholding, emit continuous correction targets)
  - GDAL warp for resolution mismatch between VR BAGs with different refinement structures
  - Nodata fix: read actual `grid.nodata_value` plus `abs(depth) < 1e5` safety check (catches both +/-1e6 sentinels)
- `training/losses.py`: Added `RegressionLoss` class, dispatched forward by mode, restored `compute_correction_delta` and `correction_delta` parameter from V8
- `training/trainer.py`:
  - Mode-aware dataset (auto-detect from band 1 description)
  - V9 local_std normalization preserved for both modes
  - `_compute_training_stats` now handles both modes
  - MAE metric for regression mode; history tracking handles both
  - `CORRECTION_NORM_FLOOR` and `CORRECTION_NORM_CAP` constants made explicit
- `scripts/train.py`: Picks up both `_ground_truth.tif` and `_regression.tif` files

---

## 2026-05-19 - V10 Regression Mode Design

### Architectural Rationale
- Identified structural problem with classification approach: forces binary decision on continuous signal
- Cells near threshold boundary get inconsistent labels for nearly identical real-world cases
- Correction head only trains on cells labeled noise, never learns to predict near-zero corrections
- Threshold value determines what model learns, but threshold is arbitrary even when adaptive
- Mixed depth regimes (shallow vs deep water) need different threshold values, creating inconsistent labeling across the training set

### Key Insight
- The difference between the clean and dirty CUBE surfaces IS the training signal at every cell
- A cell with a 0.01m difference and a cell with a 30m difference are both informative
- Regression preserves the full continuous signal; classification discards it
- Navigation safety use case is better served by predicting magnitudes than by binary classification
- Hydrographers want to know "how wrong is this cell" not "is this cell noise"

### Decision
- Pursue regression as parallel implementation (V10) while keeping V9 classification working
- Build on existing model architecture (still outputs correction head); change only what's needed in dataset, loss, and training loop
- Validate on E00269 first since data is available locally
- Defer model architecture simplification (removing unused classification head) until V10 proven

---

## 2026-05-18 - Ground Truth Preparation for New Survey Pairs

### H13739 Processed (Pacific Islands VR)
- Clean and noisy BAGs both at 16m resolution after VR resampling (different refinement structures: 16.078m vs 16.001m finest)
- Required GDAL warp to align grids before differencing
- Mean correction 32.7m, max 1304m (deep water Pacific)
- Both surfaces in MLLW; `--no-offset` flag used
- 61% noise cells / 39% seafloor at 6.73m adaptive threshold
- Noise concentrated along sparse trackline coverage, as expected for deep water surveys

### E00269 1of6 Processed (Pacific Islands SR, Northern Mariana Islands)
- 4m resolution single-resolution BAG
- Initially produced high noise percentages (67-99%) with fixed 0.15m threshold
- Investigation revealed variable systematic offsets (-0.51m to +5.91m) across sub-files
- Clean BAGs explicitly labeled MLLW; dirty BAGs in survey datum
- 4m sub-file with offset removal enabled: 75% noise at 0.117m adaptive threshold, seafloor mean diff = 0.000m (offset removal worked correctly)

### Diagnostic Script Developed (check_pair.py)
- Quick pre-check for clean/noisy BAG pair quality
- Loads both surfaces, computes difference, runs noise statistics
- Initially used too-rigid pass/fail logic from Seward assumptions; lessons led to insight that classification thresholds don't generalize across depth regimes
- Script abandoned in favor of running `prepare_ground_truth.py` directly with appropriate flags

### Adaptive Threshold Development
- Otsu's method applied to log-scaled absolute differences
- Adapts naturally across depth regimes (0.12m in shallow Pacific, 6.73m in deep Pacific)
- Replaces the requirement to pick a fixed threshold per survey
- Implementation in `compute_adaptive_threshold()` in `prepare_ground_truth.py`

### Datum Diagnostic Insights
- Variable median offsets across sub-files of the same survey indicates datum mismatch, not simple processing offset
- Survey datum to MLLW separation varies spatially with geoid model and tidal zoning
- Different sub-files cover different areas, so each gets a different offset
- This isn't a bug in the data; it's why the offset removal flag is needed

### CUBE Re-Grid Behavior Documented
- When the clean and noisy point clouds differ (noise cleaning removes outliers), CUBE re-gridding produces pervasive cell-to-cell differences, not just isolated spikes
- Every cell that had any outlier sounding in its weighted contribution changes
- Cell-to-cell differences are the training signal, not an artifact to filter out
- The Seward training data worked differently because manual edits were applied directly to the grid surface after CUBE ran, leaving unchanged cells identical
- 50/50 shoal/deep split is normal for re-grid differences; not a quality warning

---

## 2026-03-04 - Data Acquisition Plan for Geographic Diversity

### Survey Identification
- Identified 22 surveys across 8 regions for training data expansion
- Regions: Gulf Coast (3), SE Atlantic (2), Mid-Atlantic (3), Northeast (1), Great Lakes (2), Pacific NW (2), Alaska (6), Pacific Islands (3)
- Alaska surveys include diverse acquisition types: set line spacing in shallow flat water, Bering Sea/North Slope trackline, standard multibeam
- E00269 (Northern Mariana Islands) available locally for immediate processing

### Archive Request
- Clean BAGs downloadable directly from NCEI for 21 of 22 surveys
- Processed (pre-cleaning) data requested from NCEI archive to produce noisy BAGs
- Expected delivery: days to weeks
- Processing plan: one survey per region first to detect unusable pairs early

### Expected Outcome
- At 30-50% attrition, expect 11-15 usable pairs
- Combined with 4 existing Seward pairs: 15-19 total from 8+ environments
- Primary mitigation for persistent overfitting observed in V5-V9

---

## 2026-03-02 - Training Data Diversity Investigation

### Data Pair Evaluation
- Tested 3 new survey pairs for training data diversity:
  - H13532: Florida river survey (SR BAG, 1m) - 4 noise cells / 456K valid (0.00%)
  - H14190: Coastal Alaska (VR BAG, 1m) - 149 noise cells / 30M valid (0.00%)
  - F00889: Norfolk river survey (SR BAG, 0.5m) - 1 noise cell / 23M valid (0.00%)
- All three pairs produced near-zero noise because differences don't propagate to gridded surfaces
- `prepare_ground_truth.py` confirmed working for both SR and VR BAGs (auto-detection via BathymetricLoader)
- Decision: Do not include these pairs in training; find pairs with grid-visible noise instead

### Lesson Documented
- Added data quality verification step to workflow: visually inspect difference layer in QGIS before running ground truth preparation
- Training data with near-zero noise would shift class balance from 75/25 to 97/3, risking majority-class collapse

---

## 2026-02-27 - V7/V8/V9 Training Runs & Correction Normalization

### V7: Boundary-Aware Feature Computation
- **Root cause identified (V6 failure):** `scipy.ndimage.uniform_filter` with `mode='nearest'` bled nodata values (1e6) into local statistics at survey boundaries, creating artificial feature spikes
- **Fix:** Replaced with masked local statistics using only valid neighbors; nodata filled with local mean before gradient/curvature computation
- **Results:** Noise detection jumped from 3.3% to 34.8%; peak val accuracy ~72% with meaningful noise detection (confirming genuine classification, not majority-class collapse); 11,790 auto-corrections applied; mean confidence 0.825. Note: 34.8% detection against the validation set's 18.8% actual noise rate indicates significant over-prediction (false positives), likely from Seward-specific pattern memorization.
- **Visual validation:** QGIS confirmed noise classifications follow actual noise spatial patterns, not survey boundaries

### V8: Dynamic Huber Delta (No Effect)
- Added data-derived Huber delta computation from correction target distribution
- 95th percentile of raw corrections (0.652m) was below min_delta floor (1.0)
- Training dynamics identical to V7; no inference run needed
- Documented adaptive delta Options 2/3 in `losses.py` for future implementation

### V9: Local Standard Deviation Correction Normalization
- **Problem:** V7 predicted 0.4m corrections where 70m was needed; Huber loss gradient plateau above delta means model cannot distinguish large from small corrections
- **Solution:** Normalize correction targets by per-node local_std from graph construction
  - `graph_construction.py`: `_compute_node_features` returns `(features, local_std)` tuple; `build_graph` stores `data.local_std`
  - `trainer.py`: Both dataset classes divide correction targets by `max(local_std, 0.01m)`, clamp to +/-50 std devs
  - `inference_native.py`: Denormalize by multiplying predicted corrections by local_std
- Constants: `CORRECTION_NORM_FLOOR = 0.01`, `CORRECTION_NORM_CAP = 50.0`
- **Initial extreme value problem:** Max normalized correction of 2,692 from noise spikes in flat areas; solved by capping at +/-50 std devs
- **V9 training results:** Best val loss 1.813 (epoch 5), early stopping at epoch 19, classification identical to V7 (34.8% noise, 0.825 confidence), but corrections up to 32m larger than V7

### Code Changes
- `data/graph_construction.py`: Added local_std output from node feature computation
- `training/trainer.py`: Added correction normalization in both dataset classes and stats computation
- `scripts/inference_native.py`: Added denormalization step in `NativeVRProcessor.process_grid`
- `training/losses.py`: Added documentation for adaptive Huber delta options (unchanged functionally)

---

## 2026-02-27 - V5/V6 Training Iterations

### V5: Class Weight Bug Discovery
- Training without class weights resulted in model predicting seafloor for all cells
- Val accuracy stabilized at ~67% with no learning progression (majority-class-like collapse)
- Inference produced 0% noise detection with 0.967 confidence

### V6: Auto Class Weight Implementation
- `trainer.py` now scans training tiles to count class distributions automatically
- Computes inverse-frequency weights with smoothing: `weight = total / (n_classes * count + smooth)`
- Broke the all-seafloor pattern, noise detection at 3.3%
- However, visual validation revealed boundary artifact problem (see V7 above)

---

## 2026-02-12 - Unified Native BAG Processing & Documentation Updates

### Major Changes

#### Unified Native BAG Inference
- `inference_native.py` now handles BOTH VR and SR BAGs automatically
- Added `detect_bag_type()` function to auto-detect BAG type
- Added `SRBagHandler` and `SRBagWriter` classes for SR BAG native processing
- Single script for all native BAG processing simplifies user workflow

#### Auto-Detection in BathymetricLoader
- `BathymetricLoader` now auto-detects SR vs VR BAGs before loading
- SR BAGs are loaded directly, ignoring `--vr-bag-mode` setting
- Eliminates "No supergrids available" errors when loading SR BAGs with default settings

### Bug Fixes

#### Critical: Correction Sign Error in Native Inference Scripts
- Fixed correction application direction in `scripts/inference_native.py`
- **Bug**: Script was using `+= correction` which would double noise instead of removing it
- **Fix**: Changed to `-= correction` to match `models/pipeline.py` behavior
- The model predicts `correction = noisy_depth - clean_depth`, so recovery requires `clean = noisy - correction`

#### SR BAG Metadata Parsing
- Fixed `SRBagHandler` failing on SR BAGs with array-based metadata
- Metadata stored as numpy array of bytes now correctly converted via `tobytes()`

### Documentation Updates
- Simplified "Architecture" section in README to plain language "How It Works"
- Clarified that ALL classification classes (0, 1, 2) contribute to training
- Updated training strategy to emphasize real data pairs over synthetic noise
- Clarified `inference_native.py` handles both VR and SR BAGs
- Added "Practical Usage" section to README with workflow position
- Added "Documentation" section with links to HOW_IT_WORKS.md and TRAINING_PLAN.md
- Expanded "Current State" in TRAINING_PLAN.md with milestone table
- Added "Training Terminology" section clarifying epochs vs iterations vs surveys
- Added "Ground Truth Acquisition Options" and "Why Train on Diverse Data" sections
- Added "Protecting Uncharted Features" section to HOW_IT_WORKS.md
- Added "Processing Time" and "Operational Confidence Thresholds" sections
- Fixed node feature description (was "roughness", now "local statistics, gradients, curvature")
- Fixed edge feature description (was "slope direction", now "slope angle")
- Clarified uncertainty scaling only applies to native processing scripts
- Clarified feature class training is Phase 3 (planned), not currently active

## 2025-12-10 - Native VR BAG Support & Pipeline Fixes

### Major Features

#### Native VR BAG Processing
- Added `data/vr_bag.py` module for handling Variable Resolution BAGs without resampling
- New `scripts/inference_native.py` for native VR inference that preserves multi-resolution structure
- `VRBagHandler` class for reading VR BAG refinement grids
- `VRBagWriter` class for copy-and-modify workflow
- `SidecarBuilder` class for generating GeoTIFF outputs from native VR processing
- Processes each refinement grid (3x3 to 50x50 cells) individually through the GNN
- Much higher confidence scores (73.5% vs 4%) compared to resampled approach

#### Training Plan & Scripts
- Added `docs/TRAINING_PLAN.md` with comprehensive training methodology
- New `scripts/prepare_ground_truth.py` for creating labels from clean/noisy pairs
- New `scripts/evaluate_model.py` for measuring model performance against ground truth
- New `scripts/analyze_noise_patterns.py` for understanding real noise characteristics

#### VR BAG Loading Improvements
- Added `--vr-bag-mode` argument to control VR BAG loading:
  - `resampled` (default): Uses GDAL's MODE=RESAMPLED_GRID for uniform output
  - `refinements`: Direct refinement subdataset access
  - `base`: Base grid only (not recommended)
- Fixed VR BAGs loading as tiny grids (was 512x512, now 25369x25369 at 1m)

### Bug Fixes

#### Tile Merging Fix
- Fixed `TileMerger` NaN initialization bug in `data/tiling.py`
- Weighted averaging was failing because NaN + anything = NaN
- Now properly initializes first writes to 0.0 before accumulating

#### Unprocessed Valid Data Preservation
- Valid cells in sparse tiles (below min_valid_ratio) now preserved in output
- Previously these cells were lost, causing coverage gaps
- Now marked as class 0 (seafloor) with confidence 0 (not analyzed)

#### PyTorch 2.6+ Compatibility
- Fixed `torch.load()` requiring `weights_only=False` for checkpoint loading
- Updated `config.py` to handle tuple/list conversion for YAML serialization

#### RTX 50 Series (Blackwell) GPU Support
- Auto-detect unsupported GPU and fall back to CPU gracefully
- Added to all scripts: inference.py, train.py, test_pipeline.py, pipeline.py, trainer.py

#### DLL Conflict Resolution (Windows)
- Added `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` to handle OpenMP conflicts
- Fixed import order: torch must be imported before numpy on Windows

### New Files
- `data/vr_bag.py` - Native VR BAG handler
- `scripts/inference_native.py` - Native VR inference script
- `scripts/diagnose_tiles.py` - Diagnostic tool for tile validity issues
- `scripts/explore_vr_bag.py` - VR BAG structure explorer

### Modified Files
- `data/__init__.py` - Added VR BAG exports
- `data/loaders.py` - VR BAG mode support, BAG writer improvements
- `data/tiling.py` - Fixed tile merging bug
- `config/config.py` - YAML serialization fixes
- `models/pipeline.py` - VR mode support, valid_mask band, correction band output
- `training/trainer.py` - VR mode support, GPU compatibility
- `scripts/inference.py` - Added --vr-bag-mode, --min-valid-ratio arguments
- `scripts/train.py` - Added --vr-bag-mode argument
- `scripts/test_pipeline.py` - GPU compatibility fixes

### Output Format Changes
- GeoTIFF output now includes 6 bands:
  1. Depth (cleaned)
  2. Uncertainty (from original)
  3. Classification (0=seafloor, 1=feature, 2=noise)
  4. Confidence (0-1, where 0=not analyzed)
  5. Correction (suggested depth adjustment)
  6. Valid_mask (1=valid, 0=nodata)

### Uncertainty Scaling
- Corrected cells have uncertainty scaled by model confidence:
  - High confidence (0.9) -> uncertainty x 1.1
  - Low confidence (0.5) -> uncertainty x 1.5
  - Formula: `scale_factor = 2.0 - confidence`

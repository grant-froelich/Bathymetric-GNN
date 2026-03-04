# Changelog

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
- **Results:** Noise detection jumped from 3.3% to 34.8%, matching ground truth distributions; peak val accuracy ~72% with meaningful noise detection (confirming genuine classification, not majority-class collapse); 11,790 auto-corrections applied; mean confidence 0.825
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
- Val accuracy of 67% matched seafloor proportion exactly (appeared healthy but was total failure)
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

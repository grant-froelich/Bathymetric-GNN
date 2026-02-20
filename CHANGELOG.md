# Changelog

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
  - High confidence (0.9) → uncertainty × 1.1
  - Low confidence (0.5) → uncertainty × 1.5
  - Formula: `scale_factor = 2.0 - confidence`

### Command Line Examples

```bash
# Native VR BAG processing (preserves VR structure)
python scripts/inference_native.py \
    --input survey.bag \
    --model outputs/final_model.pt \
    --output survey_clean.bag \
    --min-valid-ratio 0.01

# Resampled VR BAG processing (uniform grid output)
python scripts/inference.py \
    --input survey.bag \
    --model outputs/final_model.pt \
    --output survey_clean.tif \
    --vr-bag-mode resampled \
    --min-valid-ratio 0.01

# Diagnose tile validity issues
python scripts/diagnose_tiles.py --survey survey.bag --vr-bag-mode resampled

# Explore VR BAG structure
python scripts/explore_vr_bag.py --survey survey.bag
```

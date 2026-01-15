# Changelog

## 2026-01-15 - Documentation Updates

### New Documentation
- Added `docs/HOW_IT_WORKS.md` - Comprehensive guide explaining GNN theory and practical application
  - Graph construction from bathymetric grids
  - Message passing and attention mechanisms
  - Why spatial context matters for noise detection
  - Practical workflow for end users

### Training Plan Updates
- Added realistic ground truth data requirements (minimum vs target counts)
- Added diversity guidance for training data selection
- Added `train.py` extension roadmap (native BAG iteration, code consolidation)
- Added future architecture evaluation section (CNN+GNN hybrid consideration)
- Updated timeline to reflect iterative data collection

---

## 2026-01-14 - SR BAG Support

### New Features

#### Single Resolution (SR) BAG Support
- `inference_native.py` now handles both VR and SR BAGs automatically
- Added `detect_bag_type()` function to identify BAG type
- Added `SRBagHandler` and `SRBagWriter` classes for SR BAG processing
- SR BAGs output corrected SR BAG + sidecar GeoTIFF (same as VR)

#### Usage
```cmd
:: Works with both VR and SR BAGs - type detected automatically
python scripts/inference_native.py ^
    --input survey.bag ^
    --model outputs/final_model.pt ^
    --output cleaned_survey.bag
```

### Technical Details
- SR BAGs are processed as a single grid (no refinement iteration)
- Output preserves original BAG format (VR stays VR, SR stays SR)
- Sidecar GeoTIFF generated at native resolution for both types

---

## 2026-01-13 - S-57 ENC Feature Extraction via REST API

### New Features

#### NOAA ENC Direct Integration
- Added `scripts/extract_s57_features.py` for querying NOAA ENC data directly
- **No download required** - queries ENC Direct REST API automatically
- Extracts wrecks, obstructions, and underwater rocks within survey bounds
- Supports both REST API mode (recommended) and local S-57 ENC files

#### Data Source
- **ENC Direct** (`encdirect.noaa.gov`): Primary source for all features, updated weekly
- AWOIS historical data available optionally via `--awois` flag

#### Usage
```cmd
:: Query by survey bounds (automatic)
python scripts/extract_s57_features.py --survey survey.bag --labels features.tif

:: Export to GeoJSON for QGIS
python scripts/extract_s57_features.py --survey survey.bag --output features.geojson

:: Include historical AWOIS data
python scripts/extract_s57_features.py --survey survey.bag --labels features.tif --awois
```

### Documentation
- Updated `docs/TRAINING_PLAN.md` Phase 3 with ENC Direct workflow
- ENC Direct is now the sole source for wrecks, obstructions, and rocks

---

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

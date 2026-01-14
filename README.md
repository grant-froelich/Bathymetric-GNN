# Bathymetric GNN

Context-aware bathymetric data denoising using Graph Neural Networks.

## Overview

This system addresses the challenge of distinguishing acoustic noise from real seafloor features in bathymetric data. Unlike traditional approaches that treat each point independently, this GNN-based approach leverages spatial context to make better decisions - particularly in dynamic seafloor environments where noise and features are difficult to distinguish locally.

## Core Concept

```
Local ambiguity + Spatial context = Better classification

- A spike on flat seafloor → almost certainly noise
- A spike in a rocky area connected to other spikes → probably a real feature
- An isolated spike breaking ridge continuity → probably noise
```

## Current State

| Metric | Value |
|--------|-------|
| Training data | Synthetic noise only |
| Training surveys | 5 clean VR BAGs |
| Epochs completed | 50 |
| Synthetic accuracy | 63.2% |
| Real-world accuracy | Unknown |
| Mean confidence | 73.5% (native processing) |

**Next milestone:** Establish baseline metrics with real ground truth data. See `docs/TRAINING_PLAN.md` for the complete training roadmap.

## Key Features

### Native BAG Processing (VR and SR)

The `inference_native.py` script automatically detects and handles both BAG types:

| BAG Type | Processing | Output |
|----------|------------|--------|
| VR (Variable Resolution) | Iterates refinement grids | Corrected VR BAG + sidecar GeoTIFF |
| SR (Single Resolution) | Processes full grid | Corrected SR BAG + sidecar GeoTIFF |

Native processing preserves original structure and achieves higher confidence than resampled approaches.

### ENC Feature Extraction

Extract known seafloor features from NOAA's ENC Direct REST API for training:
- Shipwrecks, obstructions, underwater rocks
- No download required - queries API directly
- Creates labeled GeoTIFFs for training

### Training Phases

| Phase | Goal | Data Required | Scripts |
|-------|------|---------------|---------|
| 1 | Ground truth from clean/noisy pairs | 6-10 BAGs (3-5 pairs) | `prepare_ground_truth.py`, `evaluate_model.py` |
| 2 | Feature labels from ENC Direct | 5-10 feature-area BAGs | `extract_s57_features.py` |
| 3 | Train on real labels | (uses Phase 1-2 data) | `train.py` (needs extension) |
| 4 | Deploy and refine | Unlimited | `inference_native.py` |

**Total minimum:** 11-20 BAG files to begin real-data training.

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate bathymetric-gnn

# Verify installation
python -c "import torch; import torch_geometric; print('Ready')"
```

### Windows with GDAL HDF5 Support

```bash
conda activate bathymetric-gnn
conda install -c conda-forge gdal libgdal-hdf5
```

## Project Structure

```
bathymetric-gnn/
├── config/
│   └── config.py                   # Configuration dataclass
├── data/
│   ├── loaders.py                  # BAG/GeoTIFF loading via GDAL
│   ├── graph_construction.py       # Build graphs from grids
│   ├── synthetic_noise.py          # Generate training data
│   ├── tiling.py                   # Tile management for large grids
│   └── vr_bag.py                   # Native BAG handler (VR and SR)
├── models/
│   ├── gnn.py                      # Graph neural network
│   └── pipeline.py                 # Full inference pipeline
├── training/
│   ├── trainer.py                  # Training loop
│   └── losses.py                   # Loss functions
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── inference.py                # Inference (resampled mode)
│   ├── inference_native.py         # Native BAG inference (VR and SR)
│   ├── prepare_ground_truth.py     # Generate labels from clean/noisy pairs
│   ├── evaluate_model.py           # Evaluate predictions against ground truth
│   ├── extract_s57_features.py     # Extract features from NOAA ENC Direct
│   ├── analyze_noise_patterns.py   # Characterize noise (optional)
│   ├── diagnose_tiles.py           # Tile validity diagnostics
│   └── explore_vr_bag.py           # VR BAG structure explorer
└── docs/
    └── TRAINING_PLAN.md            # Detailed training roadmap
```

## Usage

### Native BAG Inference (Recommended)

Works with both VR and SR BAGs - type detected automatically:

```bash
python scripts/inference_native.py \
    --input survey.bag \
    --model outputs/final_model.pt \
    --output cleaned_survey.bag \
    --min-valid-ratio 0.01
```

**Outputs:**
- `cleaned_survey.bag` - Corrected BAG (same format as input)
- `cleaned_survey_gnn_outputs.tif` - Sidecar with classification/confidence

### Training

Current training uses synthetic noise only:

```bash
python scripts/train.py \
    --clean-surveys data/raw/clean \
    --output-dir outputs/models/v2 \
    --epochs 100
```

> **Note:** To train on real labels from Phases 1-2, `train.py` needs to be extended to accept a `--labels` argument. This is a pending development task.

### Ground Truth Generation

Create labeled data from clean/noisy survey pairs:

```bash
python scripts/prepare_ground_truth.py \
    --clean data/raw/clean/survey_clean.bag \
    --noisy data/raw/noisy/survey_noisy.bag \
    --output-dir data/processed/labels \
    --noise-threshold 0.15
```

### Feature Extraction from ENC Direct

Extract known features (wrecks, rocks, obstructions) for training:

```bash
# Query by survey bounds (no download required)
python scripts/extract_s57_features.py \
    --survey data/raw/features/harbor.bag \
    --labels data/processed/labels/harbor_features.tif \
    --output data/processed/labels/harbor_features.geojson
```

### Model Evaluation

Compare predictions against ground truth:

```bash
python scripts/evaluate_model.py \
    --ground-truth data/processed/labels/survey_ground_truth.tif \
    --predictions outputs/predictions/survey_gnn_outputs.tif \
    --output outputs/metrics/survey_eval.json
```

### Diagnostic Tools

```bash
# Check tile validity and coverage
python scripts/diagnose_tiles.py \
    --survey survey.bag

# Explore VR BAG HDF5 structure
python scripts/explore_vr_bag.py \
    --survey vr_survey.bag
```

## Output Products

- **Cleaned depth grid** - Same format as input (VR or SR BAG)
- **Classification map** - Per-cell: 0=seafloor, 1=feature, 2=noise
- **Confidence map** - 0-1 per cell
- **Correction map** - Suggested depth adjustments

### Uncertainty Scaling

Corrected cells have uncertainty scaled by model confidence:
- High confidence (0.9) → uncertainty × 1.1
- Low confidence (0.5) → uncertainty × 1.5

## Data Format Support

| Format | Input | Output |
|--------|-------|--------|
| VR BAG | Yes | Yes (preserves structure) |
| SR BAG | Yes | Yes |
| GeoTIFF | Yes | Yes |

## Hardware Requirements

- **Minimum**: 16 GB RAM, NVIDIA GPU with 8 GB VRAM
- **Recommended**: 32 GB RAM, NVIDIA GPU with 16+ GB VRAM
- **CPU-only**: Supported but significantly slower
- **RTX 50 Series**: Auto-detected, falls back to CPU until PyTorch adds support

## Known Issues

- RTX 50 series GPUs (sm_120) not yet supported by PyTorch; auto-falls back to CPU
- Full BAG XML metadata not preserved when creating new BAGs
- Current model trained on synthetic noise only - real-data training pending

## License

[TBD]

## Citation

[TBD]

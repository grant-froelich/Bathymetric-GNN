# Bathymetric GNN

Context-aware bathymetric data denoising using Graph Neural Networks.

## Overview

This system addresses the challenge of distinguishing acoustic noise from real seafloor features in bathymetric data. Unlike traditional approaches that treat each point independently, this GNN-based approach leverages spatial context to make better decisions - particularly in dynamic seafloor environments where noise and features are difficult to distinguish locally.

## Practical Usage

This tool fits into the hydrographic workflow as a QC aid, similar to Flier Finder:

| Use Case | When | Who |
|----------|------|-----|
| Field QC | After surface generation, before submission | Field hydrographers |
| Office QC | Reviewing submitted ESD | Branch processors |
| Batch processing | Screening multiple surveys | QC leads |

**Workflow position:**
```
Point Cloud → Surface Generation → GNN Noise Detection → Human Review → Final Product
```

The tool does NOT replace human review - it focuses attention on areas that need it.

## Documentation

- [Training Plan](docs/TRAINING_PLAN.md) - Detailed training phases, ground truth acquisition, and timeline
- [How It Works](docs/HOW_IT_WORKS.md) - Technical deep-dive on GNN architecture and attention mechanisms

## Core Concept

```
Local ambiguity + Spatial context = Better classification

- A spike on flat seafloor → almost certainly noise
- A spike in a rocky area connected to other spikes → probably a real feature
- An isolated spike breaking ridge continuity → probably noise
```

## How It Works

The tool examines each depth point and its neighbors to decide if it's real or noise:

1. **Look at each point**: Gather local info (depth, slope, roughness, uncertainty)
2. **Look at neighbors**: Build connections to surrounding points
3. **Share information**: Each point "asks" its neighbors what they look like
4. **Make a decision**: Classify as seafloor, feature, or noise with a confidence score
5. **Apply selectively**: Only correct high-confidence noise; flag uncertain areas for human review

The key advantage over simple filters: the model considers spatial context, not just local statistics.

### Classification Categories

The model classifies each depth point into one of three categories. The model learns from ALL classes during training:

| Class | Value | Meaning | Training Role |
|-------|-------|---------|---------------|
| Seafloor | 0 | Consistent with neighbors, true bottom | Learn what "normal" looks like |
| Feature | 1 | Different from neighbors BUT real (wreck, rock, etc.) | Learn to preserve real objects |
| Noise | 2 | Inconsistent with context, likely artifact | Learn what noise looks like |

**Important**: All three classes contribute to training. The model needs to see examples of clean seafloor (0), real features (1), and actual noise (2) to learn the differences between them.

### How ENC Features Help

ENC feature extraction (`scripts/extract_s57_features.py`) serves two purposes:

1. **Preservation**: Known charted features (wrecks, rocks, obstructions) are labeled as class 1, protecting them from being flagged as noise even if they appear as isolated spikes.

2. **Training**: The model learns what real features look like vs. noise by seeing examples where we KNOW the spike is real. This helps it recognize similar patterns in areas without ENC coverage.

**Current status**: The ENC feature extraction pipeline is implemented but not yet integrated into training. See Phase 3 in TRAINING_PLAN.md.

## Key Features

### Variable Resolution (VR) BAG Support

Native VR BAG processing preserves multi-resolution structure:
- Processes each refinement grid (3x3 to 50x50 cells) individually
- Maintains original VR BAG format in output
- Generates sidecar GeoTIFF with classification/confidence layers
- Much higher confidence scores than resampled approach

### Tile-Based Processing

Full surveys (e.g., 60,000 x 60,000 at 0.5m resolution) are too large for memory. 
The system processes overlapping tiles and stitches results.

### Training Strategy

Training requires clean/noisy survey pairs where we know what the correct answer is:

1. **Primary**: Real noisy/clean survey pairs from field units or archives
2. **Supplemental**: Human reviewer corrections fed back as new training data

**Real data pairs are preferred.** Synthetic noise injection was used for initial script testing but is not recommended for production training since it doesn't capture the full complexity of real acoustic artifacts.

See TRAINING_PLAN.md for detailed instructions on preparing ground truth data.

### Output Products

- Cleaned depth grid (same format as input)
- Classification map (noise/feature/seafloor per point)
- Confidence map (0-1 per point)
- Correction map (suggested depth adjustments)
- Valid data mask

### Uncertainty Scaling

When using native processing (`inference_native.py` or `inference_native.py`), corrected depths have their uncertainty scaled to reflect AI intervention:

```
scale = 2.0 - confidence
new_uncertainty = original_uncertainty × scale
```

| Confidence | Scale Factor | Effect |
|------------|--------------|--------|
| 0.9 (high) | 1.1 | Small uncertainty increase |
| 0.7 | 1.3 | Moderate increase |
| 0.5 (low) | 1.5 | Larger increase |

**Rationale**: This provides traceability - downstream users can identify where AI modified the data and how confident the model was in those changes. The scaling factors can be adjusted based on operational risk tolerance.

**Note**: Resampled processing (`inference.py`) currently preserves original uncertainty without scaling.

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
│   └── config.py                  # Configuration dataclass
├── data/
│   ├── loaders.py                 # BAG/GeoTIFF loading via GDAL
│   ├── graph_construction.py      # Build graphs from grids
│   ├── synthetic_noise.py         # Generate training data
│   ├── tiling.py                  # Tile management for large grids
│   └── vr_bag.py                  # Native VR BAG handler
├── models/
│   ├── gnn.py                     # Graph neural network
│   └── pipeline.py                # Full inference pipeline
├── training/
│   ├── trainer.py                 # Training loop
│   └── losses.py                  # Loss functions
└── scripts/
    ├── train.py                   # Training entry point
    ├── inference.py               # Inference (resampled mode)
    ├── inference_native.py     # Native BAG inference (VR and SR)
    ├── diagnose_tiles.py          # Tile validity diagnostics
    └── explore_vr_bag.py          # VR BAG structure explorer
```

## Usage

### Training

```bash
python scripts/train.py \
    --clean-surveys /path/to/clean/surveys \
    --output-dir /path/to/model/output \
    --epochs 100 \
    --vr-bag-mode resampled
```

### Inference (Single Resolution or Resampled VR)

```bash
python scripts/inference.py \
    --input /path/to/survey.bag \
    --model /path/to/model.pt \
    --output /path/to/output.tif \
    --vr-bag-mode resampled \
    --min-valid-ratio 0.01
```

### Native BAG Inference (Recommended)

Handles both VR and SR BAGs automatically, preserving original structure:

```bash
python scripts/inference_native.py \
    --input /path/to/survey.bag \
    --model /path/to/model.pt \
    --output /path/to/output.bag \
    --min-valid-ratio 0.01
```

The script auto-detects whether the input is VR or SR and processes appropriately:
- **VR BAGs**: Iterates through refinement grids, preserves multi-resolution structure
- **SR BAGs**: Processes the full elevation grid directly

Output:
- `output.bag` - BAG with corrections applied (same type as input)
- `output_gnn_outputs.tif` - Sidecar GeoTIFF with classification/confidence

### Diagnostic Tools

```bash
# Check tile validity and coverage
python scripts/diagnose_tiles.py \
    --survey /path/to/survey.bag \
    --vr-bag-mode resampled

# Explore VR BAG HDF5 structure
python scripts/explore_vr_bag.py \
    --survey /path/to/vr_survey.bag
```

## Data Format Support

- **Input**: ONSWG BAG (SR and VR), GeoTIFF, ASC (via GDAL)
- **Output**: 
  - BAG: Copy-and-modify for SR, new SR for resampled VR
  - GeoTIFF: Multi-band with depth, uncertainty, classification, confidence, correction, valid_mask
  - Native VR: Preserved VR structure + sidecar GeoTIFF

### Why BAG Files?

**Current implementation uses BAG files because:**
- Open format with well-documented HDF5 structure
- Easy to read/write with standard GDAL/Python tools
- Available for both field units and archives

**Future expansion to CSAR working formats is planned to enable:**
- Pushing edits back to point cloud
- Integration with standard CARIS workflows
- Better traceability of changes

## Hardware Requirements

- **Minimum**: 16 GB RAM, NVIDIA GPU with 8 GB VRAM
- **Recommended**: 32 GB RAM, NVIDIA GPU with 16+ GB VRAM
- **CPU-only**: Supported but significantly slower
- **RTX 50 Series (Blackwell)**: Auto-detected and falls back to CPU until PyTorch adds support

## Known Issues

- RTX 50 series GPUs (sm_120) not yet supported by PyTorch; auto-falls back to CPU
- VR BAG output from resampled input creates SR BAG (cannot recreate VR structure)
- Full BAG XML metadata not preserved when creating new BAGs

## License

[TBD]

## Citation

[TBD]

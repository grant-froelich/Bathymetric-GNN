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

## Architecture

```
Stage 1: Local Feature Extraction
    Extract per-point features (roughness, gradients, local statistics)
    
Stage 2: Graph Construction  
    Build spatial graph connecting neighboring points
    
Stage 3: Graph Neural Network
    Message passing to incorporate spatial context
    Classify each point: noise / feature / smooth seafloor
    Estimate confidence
    
Stage 4: Selective Correction
    Auto-correct high-confidence noise
    Preserve high-confidence features
    Flag low-confidence regions for human review
```

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

1. **Primary**: Synthetic noise added to clean reference surveys
2. **Validation**: Small set of real noisy/clean pairs
3. **Refinement**: Human reviewer corrections fed back as training data

### Output Products

- Cleaned depth grid (same format as input)
- Classification map (noise/feature/seafloor per point)
- Confidence map (0-1 per point)
- Correction map (suggested depth adjustments)
- Valid data mask

### Uncertainty Scaling

Corrected cells have uncertainty scaled by model confidence:
- High confidence (0.9) → uncertainty × 1.1
- Low confidence (0.5) → uncertainty × 1.5

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
    ├── inference_vr_native.py     # Native VR BAG inference
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

### Native VR BAG Inference (Preserves VR Structure)

```bash
python scripts/inference_vr_native.py \
    --input /path/to/vr_survey.bag \
    --model /path/to/model.pt \
    --output /path/to/output.bag \
    --min-valid-ratio 0.01
```

This creates:
- `output.bag` - VR BAG with corrections applied
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

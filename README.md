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

## Key Design Decisions

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
- Review priority map (regions needing human attention)

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate bathymetric-gnn

# Verify installation
python -c "import torch; import torch_geometric; print('Ready')"
```

## Project Structure

```
bathymetric-gnn/
├── config/
│   └── config.py              # Configuration dataclass
├── data/
│   ├── loaders.py             # BAG/GeoTIFF loading via GDAL
│   ├── graph_construction.py  # Build graphs from grids
│   ├── synthetic_noise.py     # Generate training data
│   └── tiling.py              # Tile management for large grids
├── models/
│   ├── feature_extractor.py   # Local feature extraction
│   ├── gnn.py                 # Graph neural network
│   └── pipeline.py            # Full inference pipeline
├── training/
│   ├── trainer.py             # Training loop
│   └── losses.py              # Loss functions
├── evaluation/
│   ├── metrics.py             # Quality metrics
│   └── visualization.py       # Result visualization
├── review/
│   └── human_feedback.py      # Human-in-the-loop integration
└── scripts/
    ├── train.py               # Training entry point
    ├── inference.py           # Inference entry point
    └── export_review.py       # Export for human review
```

## Usage

### Training

```bash
python scripts/train.py \
    --clean-surveys /path/to/clean/surveys \
    --output-dir /path/to/model/output \
    --epochs 100
```

### Inference

```bash
python scripts/inference.py \
    --input /path/to/noisy/survey.bag \
    --model /path/to/trained/model.pt \
    --output /path/to/output/
```

### Export for Review

```bash
python scripts/export_review.py \
    --results /path/to/inference/output \
    --format geotiff \
    --confidence-threshold 0.7
```

## Data Format Support

- **Input**: ONSWG BAG, GeoTIFF, ASC (via GDAL)
- **Output**: Same as input format, plus confidence/classification layers

## Hardware Requirements

- **Minimum**: 16 GB RAM, NVIDIA GPU with 8 GB VRAM
- **Recommended**: 32 GB RAM, NVIDIA GPU with 16+ GB VRAM
- **CPU-only**: Supported but significantly slower

## License

[TBD]

## Citation

[TBD]

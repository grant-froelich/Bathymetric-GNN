#!/bin/bash
# Installation script for bathymetric-gnn
# 
# This uses the recommended PyTorch Geometric installation method
# which pulls pre-built wheels matching your PyTorch/CUDA versions.

set -e

echo "=========================================="
echo "Bathymetric GNN Installation"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create environment with base packages
echo ""
echo "Step 1: Creating conda environment..."
conda create -n bathymetric-gnn python=3.11 -y

# Activate environment
echo ""
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bathymetric-gnn

# Install PyTorch (adjust cuda version if needed)
echo ""
echo "Step 3: Installing PyTorch..."
echo "  Detecting CUDA version..."

# Try to detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "  Found CUDA $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "12"* ]]; then
        echo "  Installing PyTorch with CUDA 12.1..."
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    elif [[ "$CUDA_VERSION" == "11"* ]]; then
        echo "  Installing PyTorch with CUDA 11.8..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "  Unknown CUDA version, installing PyTorch with CUDA 12.1..."
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    fi
else
    echo "  No NVIDIA GPU detected, installing CPU-only PyTorch..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install PyTorch Geometric via pip (more reliable than conda)
echo ""
echo "Step 4: Installing PyTorch Geometric..."
pip install torch-geometric

# The optional dependencies (torch-scatter, etc.) are now included in torch-geometric
# or installed automatically. If you need them explicitly:
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu121.html

# Install geospatial packages
echo ""
echo "Step 5: Installing geospatial packages..."
conda install -c conda-forge gdal rasterio shapely pyproj h5py -y

# Install remaining dependencies
echo ""
echo "Step 6: Installing remaining packages..."
conda install -c conda-forge numpy scipy scikit-learn pandas matplotlib seaborn tqdm pyyaml -y
pip install wandb tensorboard

# Verify installation
echo ""
echo "Step 7: Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

import torch_geometric
print(f'PyTorch Geometric: {torch_geometric.__version__}')

from osgeo import gdal
print(f'GDAL: {gdal.__version__}')

print('')
print('All packages installed successfully!')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate bathymetric-gnn"
echo ""
echo "To test the pipeline:"
echo "  python scripts/test_pipeline.py --survey /path/to/survey.bag"
echo "=========================================="

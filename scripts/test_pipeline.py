#!/usr/bin/env python3
"""
Diagnostic script to test pipeline components individually.

Run this before full training to verify:
1. Data loading works
2. Tiling works  
3. Graph construction works
4. Synthetic noise generation works
5. Model forward pass works

Usage:
    python scripts/test_pipeline.py --survey /path/to/survey.bag
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Check imports
def check_imports():
    """Verify all required packages are available."""
    print("=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)
    
    errors = []
    
    # Core
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  └─ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  └─ CUDA not available, will use CPU")
    except ImportError as e:
        errors.append(f"✗ PyTorch: {e}")
    
    # PyTorch Geometric
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        errors.append(f"✗ PyTorch Geometric: {e}")
    
    # GDAL
    try:
        from osgeo import gdal
        print(f"✓ GDAL {gdal.__version__}")
    except ImportError as e:
        errors.append(f"✗ GDAL: {e}")
    
    # NumPy/SciPy
    try:
        import numpy as np
        import scipy
        print(f"✓ NumPy {np.__version__}, SciPy {scipy.__version__}")
    except ImportError as e:
        errors.append(f"✗ NumPy/SciPy: {e}")
    
    if errors:
        print("\nMissing dependencies:")
        for err in errors:
            print(f"  {err}")
        print("\nInstall with: conda env create -f environment.yml")
        return False
    
    print("\nAll imports OK!")
    return True


def test_data_loading(survey_path: Path):
    """Test loading a BAG/GeoTIFF file."""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    from data import BathymetricLoader
    
    loader = BathymetricLoader()
    
    print(f"Loading: {survey_path}")
    start = time.time()
    
    try:
        grid = loader.load(survey_path)
        elapsed = time.time() - start
        
        print(f"✓ Loaded in {elapsed:.1f}s")
        print(f"  Shape: {grid.shape}")
        print(f"  Resolution: {grid.resolution[0]:.2f} x {grid.resolution[1]:.2f}")
        print(f"  Bounds: {grid.bounds}")
        print(f"  CRS: {grid.crs[:50] if grid.crs else 'None'}...")
        print(f"  NoData: {grid.nodata_value}")
        print(f"  Valid ratio: {grid.valid_ratio:.1%}")
        
        stats = grid.get_statistics()
        print(f"  Depth range: {stats['min']:.1f} to {stats['max']:.1f}")
        print(f"  Depth mean: {stats['mean']:.1f} ± {stats['std']:.1f}")
        
        if grid.uncertainty is not None:
            print(f"  ✓ Uncertainty layer present")
        else:
            print(f"  ○ No uncertainty layer")
        
        return grid
        
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tiling(grid):
    """Test tile extraction."""
    print("\n" + "=" * 60)
    print("TESTING TILING")
    print("=" * 60)
    
    from data import TileManager
    
    tile_manager = TileManager(
        tile_size=1024,
        overlap=128,
        min_valid_ratio=0.1,
    )
    
    num_rows, num_cols, specs = tile_manager.compute_tile_grid(grid.shape)
    print(f"Grid {grid.shape} -> {num_rows} x {num_cols} = {len(specs)} tiles")
    
    # Extract a few tiles
    print("\nExtracting sample tiles...")
    tiles_extracted = 0
    tiles_skipped = 0
    
    for i, tile in enumerate(tile_manager.iterate_tiles(grid, skip_empty=True)):
        tiles_extracted += 1
        if tiles_extracted <= 3:
            print(f"  Tile ({tile.tile_row}, {tile.tile_col}): "
                  f"{tile.shape}, valid={tile.valid_ratio:.1%}")
        if tiles_extracted >= 10:
            break
    
    # Count skipped
    for tile in tile_manager.iterate_tiles(grid, skip_empty=False):
        if tile.valid_ratio < 0.1:
            tiles_skipped += 1
    
    print(f"\n✓ Tiling works")
    print(f"  Total tiles: {len(specs)}")
    print(f"  Would skip (low valid): ~{tiles_skipped}")
    print(f"  Usable tiles: ~{len(specs) - tiles_skipped}")
    
    # Return a sample tile for further testing
    for tile in tile_manager.iterate_tiles(grid, skip_empty=True):
        return tile
    
    return None


def test_graph_construction(tile, resolution):
    """Test building a graph from a tile."""
    print("\n" + "=" * 60)
    print("TESTING GRAPH CONSTRUCTION")
    print("=" * 60)
    
    from data import GraphBuilder
    
    graph_builder = GraphBuilder(
        connectivity="8-connected",
        edge_features=["distance", "depth_difference", "slope"],
    )
    
    print(f"Building graph from tile {tile.shape}...")
    start = time.time()
    
    try:
        graph = graph_builder.build_graph(
            depth=tile.data,
            valid_mask=tile.valid_mask,
            uncertainty=tile.uncertainty,
            resolution=resolution,
        )
        elapsed = time.time() - start
        
        print(f"✓ Graph built in {elapsed:.2f}s")
        print(f"  Nodes: {graph.num_nodes:,}")
        print(f"  Edges: {graph.num_edges:,}")
        print(f"  Node features: {graph.x.shape}")
        print(f"  Edge features: {graph.edge_attr.shape}")
        print(f"  Avg degree: {graph.num_edges / graph.num_nodes:.1f}")
        
        # Check for NaN/Inf
        if torch.isnan(graph.x).any():
            print(f"  ⚠ Warning: NaN in node features")
        if torch.isinf(graph.x).any():
            print(f"  ⚠ Warning: Inf in node features")
        
        return graph
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_synthetic_noise(tile):
    """Test synthetic noise generation."""
    print("\n" + "=" * 60)
    print("TESTING SYNTHETIC NOISE GENERATION")
    print("=" * 60)
    
    from data import SyntheticNoiseGenerator
    
    generator = SyntheticNoiseGenerator(
        enable_gaussian=True,
        enable_spikes=True,
        enable_blobs=True,
        enable_systematic=True,
        seed=42,
    )
    
    print("Generating synthetic noise...")
    
    try:
        result = generator.generate(
            clean_depth=tile.data,
            valid_mask=tile.valid_mask,
            intensity=1.0,
        )
        
        # Statistics
        noise_added = np.sum(result.noise_mask & tile.valid_mask)
        total_valid = np.sum(tile.valid_mask)
        
        print(f"✓ Noise generated")
        print(f"  Noisy cells: {noise_added:,} ({100*noise_added/total_valid:.1f}%)")
        print(f"  Max noise magnitude: {np.max(result.noise_magnitude):.2f}")
        print(f"  Mean noise magnitude: {np.mean(result.noise_magnitude[result.noise_mask]):.2f}")
        
        # Show depth change statistics
        depth_diff = result.noisy_depth - result.clean_depth
        valid_diff = depth_diff[tile.valid_mask]
        print(f"  Depth change range: {valid_diff.min():.2f} to {valid_diff.max():.2f}")
        
        return result
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_forward(graph):
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("TESTING MODEL FORWARD PASS")
    print("=" * 60)
    
    import torch
    from models import BathymetricGNN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    in_channels = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1] if graph.edge_attr is not None else None
    
    print(f"Creating model (in_channels={in_channels}, edge_dim={edge_dim})...")
    
    try:
        model = BathymetricGNN(
            in_channels=in_channels,
            hidden_channels=64,
            num_gnn_layers=4,
            gnn_type="GAT",
            heads=4,
            num_classes=3,
            predict_correction=True,
            dropout=0.1,
            edge_dim=edge_dim,
        )
        model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {num_params:,}")
        
        # Forward pass
        graph = graph.to(device)
        model.eval()
        
        print("Running forward pass...")
        start = time.time()
        
        with torch.no_grad():
            outputs = model(graph)
        
        elapsed = time.time() - start
        
        print(f"✓ Forward pass completed in {elapsed:.3f}s")
        print(f"  Class logits: {outputs['class_logits'].shape}")
        print(f"  Predicted classes: {outputs['predicted_class'].shape}")
        print(f"  Confidence: {outputs['confidence'].shape}")
        print(f"  Correction: {outputs['correction'].shape}")
        
        # Class distribution
        classes, counts = torch.unique(outputs['predicted_class'], return_counts=True)
        print(f"  Class distribution (untrained):")
        for c, n in zip(classes.tolist(), counts.tolist()):
            print(f"    Class {c}: {n} ({100*n/graph.num_nodes:.1f}%)")
        
        print(f"  Mean confidence: {outputs['confidence'].mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_estimate(grid):
    """Estimate memory requirements for full processing."""
    print("\n" + "=" * 60)
    print("MEMORY ESTIMATE")
    print("=" * 60)
    
    # Tile-based estimate
    tile_size = 1024
    nodes_per_tile = tile_size * tile_size  # Worst case
    edges_per_tile = nodes_per_tile * 8  # 8-connected
    
    # Bytes per element
    node_features = 7 * 4  # 7 features, float32
    edge_features = 3 * 4  # 3 features, float32
    edge_index = 2 * 8     # 2 indices, int64
    
    tile_memory_mb = (
        nodes_per_tile * node_features +
        edges_per_tile * (edge_features + edge_index)
    ) / (1024 * 1024)
    
    # Model memory (rough estimate)
    model_memory_mb = 50  # ~50MB for typical model
    
    # Batch memory
    batch_size = 4
    batch_memory_mb = tile_memory_mb * batch_size + model_memory_mb
    
    print(f"Estimated memory per tile: {tile_memory_mb:.0f} MB")
    print(f"Model memory: ~{model_memory_mb} MB")
    print(f"Batch of {batch_size} tiles: ~{batch_memory_mb:.0f} MB")
    
    if batch_memory_mb > 8000:
        print(f"⚠ May need to reduce batch size or tile size")
    else:
        print(f"✓ Should fit in 8GB GPU memory")


def main():
    parser = argparse.ArgumentParser(description="Test pipeline components")
    parser.add_argument(
        "--survey",
        type=Path,
        required=True,
        help="Path to a test survey file (BAG or GeoTIFF)",
    )
    args = parser.parse_args()
    
    print("BATHYMETRIC GNN PIPELINE TEST")
    print("=" * 60)
    
    # Check imports first
    if not check_imports():
        sys.exit(1)
    
    # Need torch for later tests
    import torch
    
    # Test each component
    grid = test_data_loading(args.survey)
    if grid is None:
        print("\n✗ Cannot continue without data loading")
        sys.exit(1)
    
    tile = test_tiling(grid)
    if tile is None:
        print("\n✗ Cannot continue without tiling")
        sys.exit(1)
    
    graph = test_graph_construction(tile, grid.resolution)
    if graph is None:
        print("\n✗ Cannot continue without graph construction")
        sys.exit(1)
    
    noise_result = test_synthetic_noise(tile)
    if noise_result is None:
        print("\n⚠ Noise generation failed, but can continue")
    
    model_ok = test_model_forward(graph)
    if not model_ok:
        print("\n✗ Model forward pass failed")
        sys.exit(1)
    
    test_memory_estimate(grid)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ All pipeline components working!")
    print(f"\nYour survey yields ~{len(list(range(0, grid.shape[0], 896))) * len(list(range(0, grid.shape[1], 896)))} tiles")
    print("\nYou're ready to run training:")
    print(f"  python scripts/train.py --clean-surveys /path/to/surveys --output-dir ./outputs")


if __name__ == "__main__":
    main()

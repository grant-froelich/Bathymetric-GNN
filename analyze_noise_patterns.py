#!/usr/bin/env python3
"""Diagnose tile validity issues."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from data import BathymetricLoader, TileManager


def main():
    parser = argparse.ArgumentParser(description="Diagnose tile validity")
    parser.add_argument("--survey", type=Path, required=True, help="Path to BAG file")
    parser.add_argument("--vr-bag-mode", default="resampled", help="VR BAG mode")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size")
    args = parser.parse_args()

    print("=" * 60)
    print("TILE VALIDITY DIAGNOSTIC")
    print("=" * 60)

    # Load grid
    print(f"\nLoading: {args.survey}")
    loader = BathymetricLoader(vr_bag_mode=args.vr_bag_mode)
    grid = loader.load(args.survey)

    print(f"\n--- Grid Statistics ---")
    print(f"Shape: {grid.shape}")
    print(f"NoData value: {grid.nodata_value}")
    print(f"Depth dtype: {grid.depth.dtype}")
    
    # Count different cell types
    total_cells = grid.depth.size
    nan_cells = np.sum(np.isnan(grid.depth))
    inf_cells = np.sum(np.isinf(grid.depth))
    nodata_cells = np.sum(grid.depth == grid.nodata_value) if grid.nodata_value is not None else 0
    finite_cells = np.sum(np.isfinite(grid.depth))
    
    # Valid mask as computed by the code
    valid_mask = np.isfinite(grid.depth)
    if grid.nodata_value is not None and not np.isnan(grid.nodata_value):
        valid_mask &= (grid.depth != grid.nodata_value)
    valid_cells = np.sum(valid_mask)
    
    print(f"\n--- Cell Breakdown ---")
    print(f"Total cells:     {total_cells:>12,}")
    print(f"NaN cells:       {nan_cells:>12,} ({100*nan_cells/total_cells:.2f}%)")
    print(f"Inf cells:       {inf_cells:>12,} ({100*inf_cells/total_cells:.2f}%)")
    print(f"NoData cells:    {nodata_cells:>12,} ({100*nodata_cells/total_cells:.2f}%)")
    print(f"Finite cells:    {finite_cells:>12,} ({100*finite_cells/total_cells:.2f}%)")
    print(f"Valid cells:     {valid_cells:>12,} ({100*valid_cells/total_cells:.2f}%)")
    
    if valid_cells > 0:
        valid_depths = grid.depth[valid_mask]
        print(f"\n--- Valid Depth Statistics ---")
        print(f"Min depth:  {np.min(valid_depths):.2f}")
        print(f"Max depth:  {np.max(valid_depths):.2f}")
        print(f"Mean depth: {np.mean(valid_depths):.2f}")
    
    # Check unique values near nodata
    print(f"\n--- Sample of large values (potential NoData) ---")
    large_values = grid.depth[grid.depth > 10000]
    if len(large_values) > 0:
        unique_large = np.unique(large_values)[:10]
        print(f"Unique large values: {unique_large}")
    else:
        print("No values > 10000 found")
    
    # Tile analysis
    print(f"\n--- Tile Analysis (tile_size={args.tile_size}) ---")
    tm = TileManager(tile_size=args.tile_size, overlap=64, min_valid_ratio=0.0)
    
    tile_stats = []
    for tile in tm.iterate_tiles(grid, skip_empty=False):
        tile_stats.append({
            'row': tile.tile_row,
            'col': tile.tile_col,
            'valid_ratio': tile.valid_ratio,
            'valid_cells': np.sum(tile.valid_mask),
        })
    
    # Sort by valid_ratio
    tile_stats.sort(key=lambda x: x['valid_ratio'], reverse=True)
    
    print(f"\nTotal tiles: {len(tile_stats)}")
    
    # Count tiles at different thresholds
    thresholds = [0.10, 0.05, 0.01, 0.001, 0.0001]
    for thresh in thresholds:
        count = sum(1 for t in tile_stats if t['valid_ratio'] >= thresh)
        print(f"Tiles >= {thresh*100:5.2f}% valid: {count:>6}")
    
    # Show top 20 tiles
    print(f"\n--- Top 20 Tiles by Valid Ratio ---")
    print(f"{'Row':>4} {'Col':>4} {'Valid%':>8} {'Valid Cells':>12}")
    for t in tile_stats[:20]:
        print(f"{t['row']:>4} {t['col']:>4} {t['valid_ratio']*100:>7.3f}% {t['valid_cells']:>12,}")
    
    # Show some middle tiles
    if len(tile_stats) > 40:
        mid = len(tile_stats) // 2
        print(f"\n--- Middle Tiles (around position {mid}) ---")
        for t in tile_stats[mid-5:mid+5]:
            print(f"{t['row']:>4} {t['col']:>4} {t['valid_ratio']*100:>7.3f}% {t['valid_cells']:>12,}")
    
    # Check if there might be a different nodata value
    print(f"\n--- Checking for potential alternate NoData values ---")
    # Look at most common values
    flat = grid.depth.flatten()
    # Sample if too large
    if len(flat) > 1000000:
        flat = np.random.choice(flat, 1000000, replace=False)
    
    unique, counts = np.unique(flat[np.isfinite(flat)], return_counts=True)
    top_indices = np.argsort(counts)[-10:][::-1]
    print("Most common values:")
    for idx in top_indices:
        print(f"  {unique[idx]:>15.2f}: {counts[idx]:>10,} occurrences")


if __name__ == "__main__":
    main()

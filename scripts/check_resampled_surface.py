#!/usr/bin/env python3
"""
scripts/check_resampled_surface.py

Verify whether _load_vr_bag_resampled produces depths that match a CARIS
export of the same surface. This isolates whether the VR resampling bug
affects a single surface (bad for using resampled surfaces as model input)
or only appears when differencing two surfaces (in which case resampled
surfaces are fine for input and only the target needs to come from CARIS).

Workflow:
  1. In CARIS, export the noisy surface as XYZ points:
     columns Easting, Northing, Depth (comma-separated, one header row)
  2. Run this script pointing at the noisy BAG and that CARIS export.

Usage:
  python scripts/check_resampled_surface.py \
      --bag "path/to/H13739_VR_Dirty_1of1.bag" \
      --caris "path/to/H13739_noisy_caris.txt" \
      --target-resolution 16.0
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import torch FIRST, before numpy or any GDAL-linked module. On Windows,
# importing torch after those libraries can trigger a DLL load failure
# (WinError 127). This script does not use torch directly, but the `data`
# package import chain pulls it in via graph_construction, so it must load
# in the right order here.
import torch  # noqa: F401

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data import BathymetricLoader


def load_caris_xyz(path: Path):
    """Load a CARIS XYZ export (Easting, Northing, Depth)."""
    eastings, northings, depths = [], [], []
    with open(path) as f:
        header = f.readline()  # skip header
        logger.info(f"CARIS file header: {header.strip()}")
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                eastings.append(float(parts[0]))
                northings.append(float(parts[1]))
                depths.append(float(parts[2]))
            except ValueError:
                continue
    return (np.array(eastings), np.array(northings), np.array(depths))


def main():
    parser = argparse.ArgumentParser(description="Compare resampled BAG surface to CARIS export")
    parser.add_argument('--bag', type=Path, required=True, help="Path to the VR BAG")
    parser.add_argument('--caris', type=Path, required=True, help="CARIS XYZ export of the same surface")
    parser.add_argument('--target-resolution', type=float, default=None,
                        help="Optional: force resampling resolution in meters. "
                             "Leave unset to match the pipeline (GDAL native finest).")
    args = parser.parse_args()

    # Load the BAG the same way the pipeline does:
    # prepare_ground_truth.py uses BathymetricLoader(vr_bag_mode='resampled')
    # and calls load() without forcing a target resolution (GDAL picks the
    # native finest). Pass --target-resolution only to override for testing.
    loader = BathymetricLoader(vr_bag_mode='resampled')
    logger.info(f"Loading BAG (resampled): {args.bag}")
    grid = loader.load(str(args.bag), vr_target_resolution=args.target_resolution)
    logger.info(f"Resampled grid: {grid.depth.shape} at {grid.resolution[0]:.3f}m")
    logger.info(f"Grid bounds: {grid.bounds}")
    logger.info(f"Nodata value: {grid.nodata_value}")

    # Load CARIS reference points
    logger.info(f"Loading CARIS export: {args.caris}")
    east, north, caris_depth = load_caris_xyz(args.caris)
    logger.info(f"CARIS points: {len(caris_depth):,}")

    # Map each CARIS point to a grid cell using the geotransform.
    # transform = (min_x, px_w, 0, max_y, 0, px_h) with px_h negative
    gt = grid.transform
    min_x, px_w, _, max_y, _, px_h = gt

    col = ((east - min_x) / px_w).astype(int)
    row = ((north - max_y) / px_h).astype(int)  # px_h negative, so this works

    H, W = grid.depth.shape
    in_bounds = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    logger.info(f"Points falling inside grid bounds: {in_bounds.sum():,} / {len(caris_depth):,}")

    col_b = col[in_bounds]
    row_b = row[in_bounds]
    caris_b = caris_depth[in_bounds]

    # Look up the resampled depth at each point
    grid_depth = grid.depth[row_b, col_b]

    # Filter to cells where the grid has valid data
    nodata = grid.nodata_value
    if nodata is not None and not np.isnan(nodata):
        valid = (grid_depth != nodata) & np.isfinite(grid_depth) & (np.abs(grid_depth) < 1e5)
    else:
        valid = np.isfinite(grid_depth) & (np.abs(grid_depth) < 1e5)

    gd = grid_depth[valid]
    cd = caris_b[valid]
    logger.info(f"Points with valid resampled depth: {valid.sum():,}")

    if len(gd) == 0:
        logger.error("No overlapping valid cells. Check resolution and that the CARIS export matches this BAG.")
        sys.exit(1)

    # Compare
    diff = gd - cd
    abs_diff = np.abs(diff)

    print()
    print("=" * 60)
    print("Resampled BAG depth  vs  CARIS depth")
    print("=" * 60)
    print(f"Points compared:        {len(gd):,}")
    print(f"GDAL depth range:       {gd.min():.3f} to {gd.max():.3f}m")
    print(f"CARIS depth range:      {cd.min():.3f} to {cd.max():.3f}m")
    print(f"GDAL mean depth:        {gd.mean():.3f}m")
    print(f"CARIS mean depth:       {cd.mean():.3f}m")
    print()
    print(f"Depth difference (GDAL - CARIS):")
    print(f"  Mean (signed):        {diff.mean():+.4f}m")
    print(f"  Median (signed):      {np.median(diff):+.4f}m")
    print(f"  Mean |diff|:          {abs_diff.mean():.4f}m")
    print(f"  Median |diff|:        {np.median(abs_diff):.4f}m")
    print(f"  90th percentile:      {np.percentile(abs_diff, 90):.4f}m")
    print(f"  99th percentile:      {np.percentile(abs_diff, 99):.4f}m")
    print(f"  Max |diff|:           {abs_diff.max():.4f}m")
    print()
    # Correlation as a quick agreement check
    if len(gd) > 1:
        corr = np.corrcoef(gd, cd)[0, 1]
        print(f"  Correlation:          {corr:.6f}")
    print()
    print("Interpretation:")
    print("  If mean |diff| is small (sub-decimeter to ~resolution-scale) and")
    print("  correlation is ~1.0, the resampled surface matches CARIS and is")
    print("  fine to use for model INPUT features. The bug is only in")
    print("  differencing two surfaces, so CARIS-difference-as-target solves it.")
    print()
    print("  If mean |diff| is large or correlation is poor, the resampling")
    print("  itself is corrupting depths and resampled surfaces cannot be")
    print("  trusted for input either.")
    print("=" * 60)


if __name__ == '__main__':
    main()

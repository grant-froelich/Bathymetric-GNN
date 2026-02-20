#!/usr/bin/env python3
"""
scripts/prepare_ground_truth.py

Generate ground truth labels from clean/noisy survey pairs.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch  # Must import before data module on Windows

import argparse
import logging
from pathlib import Path
import numpy as np
from osgeo import gdal

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import BathymetricLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Class definitions
CLASS_SEAFLOOR = 0
CLASS_FEATURE = 1
CLASS_NOISE = 2


def load_survey(path: Path, vr_bag_mode: str = 'resampled'):
    """Load survey and return full BathymetricGrid object."""
    loader = BathymetricLoader(vr_bag_mode=vr_bag_mode)
    return loader.load(path)


def find_intersection(bounds1, bounds2):
    """Find intersection of two bounding boxes.
    
    Bounds format: (min_x, min_y, max_x, max_y)
    Returns: (min_x, min_y, max_x, max_y) or None if no intersection
    """
    min_x = max(bounds1[0], bounds2[0])
    min_y = max(bounds1[1], bounds2[1])
    max_x = min(bounds1[2], bounds2[2])
    max_y = min(bounds1[3], bounds2[3])
    
    if min_x < max_x and min_y < max_y:
        return (min_x, min_y, max_x, max_y)
    return None


def extract_region(grid, intersection):
    """Extract a region from a grid based on geographic bounds.
    
    Args:
        grid: BathymetricGrid object
        intersection: (min_x, min_y, max_x, max_y)
        
    Returns:
        Tuple of (depth_array, uncertainty_array, new_transform)
    """
    min_x, min_y, max_x, max_y = intersection
    transform = grid.transform
    res_x = abs(transform[1])
    res_y = abs(transform[5])
    origin_x = transform[0]
    origin_y = transform[3]
    
    # Calculate pixel coordinates
    col_start = int(round((min_x - origin_x) / res_x))
    col_end = int(round((max_x - origin_x) / res_x))
    row_start = int(round((origin_y - max_y) / res_y))
    row_end = int(round((origin_y - min_y) / res_y))
    
    # Clamp to valid range
    col_start = max(0, col_start)
    col_end = min(grid.depth.shape[1], col_end)
    row_start = max(0, row_start)
    row_end = min(grid.depth.shape[0], row_end)
    
    # Extract arrays
    depth = grid.depth[row_start:row_end, col_start:col_end]
    uncertainty = None
    if grid.uncertainty is not None:
        uncertainty = grid.uncertainty[row_start:row_end, col_start:col_end]
    
    # New transform for the extracted region
    new_origin_x = origin_x + col_start * res_x
    new_origin_y = origin_y - row_start * res_y
    new_transform = (new_origin_x, transform[1], transform[2], 
                     new_origin_y, transform[4], transform[5])
    
    return depth, uncertainty, new_transform


def compute_ground_truth(
    clean_path: Path,
    noisy_path: Path,
    output_dir: Path,
    noise_threshold: float = 0.15,
    vr_bag_mode: str = 'resampled',
):
    """
    Compute ground truth labels from clean/noisy pair.
    
    Handles surveys with different extents by finding the intersection.
    
    Args:
        clean_path: Path to clean survey
        noisy_path: Path to noisy survey
        output_dir: Directory for output files
        noise_threshold: Minimum depth difference to classify as noise (meters)
        vr_bag_mode: How to load VR BAGs
    """
    logger.info(f"Loading clean survey: {clean_path}")
    clean_grid = load_survey(clean_path, vr_bag_mode)
    
    logger.info(f"Loading noisy survey: {noisy_path}")
    noisy_grid = load_survey(noisy_path, vr_bag_mode)
    
    # Check for geographic intersection
    intersection = find_intersection(clean_grid.bounds, noisy_grid.bounds)
    if intersection is None:
        raise ValueError("Surveys do not overlap geographically")
    
    logger.info(f"Clean bounds: {clean_grid.bounds}")
    logger.info(f"Noisy bounds: {noisy_grid.bounds}")
    logger.info(f"Intersection: {intersection}")
    
    # Check resolution compatibility
    clean_res = clean_grid.resolution
    noisy_res = noisy_grid.resolution
    if abs(clean_res[0] - noisy_res[0]) > 0.01 or abs(clean_res[1] - noisy_res[1]) > 0.01:
        raise ValueError(
            f"Resolution mismatch: clean {clean_res} vs noisy {noisy_res}. "
            "Surveys must have the same resolution."
        )
    
    # Extract overlapping regions
    clean_depth, clean_uncert, transform = extract_region(clean_grid, intersection)
    noisy_depth, noisy_uncert, _ = extract_region(noisy_grid, intersection)
    crs = clean_grid.crs
    
    # Handle potential size mismatch due to rounding
    min_rows = min(clean_depth.shape[0], noisy_depth.shape[0])
    min_cols = min(clean_depth.shape[1], noisy_depth.shape[1])
    clean_depth = clean_depth[:min_rows, :min_cols]
    noisy_depth = noisy_depth[:min_rows, :min_cols]
    if clean_uncert is not None:
        clean_uncert = clean_uncert[:min_rows, :min_cols]
    if noisy_uncert is not None:
        noisy_uncert = noisy_uncert[:min_rows, :min_cols]
    
    logger.info(f"Aligned grid shape: {clean_depth.shape}")
    
    # Compute difference
    difference = noisy_depth - clean_depth
    
    # Valid mask (both surveys have data)
    nodata = 1.0e6
    valid_clean = (clean_depth != nodata) & np.isfinite(clean_depth)
    valid_noisy = (noisy_depth != nodata) & np.isfinite(noisy_depth)
    valid_mask = valid_clean & valid_noisy
    
    # Classify based on difference
    labels = np.full(clean_depth.shape, CLASS_SEAFLOOR, dtype=np.int32)
    
    # Mark noise where difference exceeds threshold
    noise_mask = np.abs(difference) > noise_threshold
    labels[noise_mask & valid_mask] = CLASS_NOISE
    
    # Mark invalid areas
    labels[~valid_mask] = -1  # NoData
    
    # Statistics
    valid_count = np.sum(valid_mask)
    noise_count = np.sum(labels == CLASS_NOISE)
    seafloor_count = np.sum(labels == CLASS_SEAFLOOR)
    
    logger.info(f"Ground truth statistics:")
    logger.info(f"  Valid cells: {valid_count:,}")
    logger.info(f"  Noise cells: {noise_count:,} ({100*noise_count/valid_count:.2f}%)")
    logger.info(f"  Seafloor cells: {seafloor_count:,} ({100*seafloor_count/valid_count:.2f}%)")
    
    if np.any(noise_mask & valid_mask):
        logger.info(f"  Mean noise magnitude: {np.mean(np.abs(difference[noise_mask & valid_mask])):.3f}m")
        logger.info(f"  Max noise magnitude: {np.max(np.abs(difference[noise_mask & valid_mask])):.3f}m")
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    survey_name = clean_path.stem.replace('_clean', '').replace('_Clean', '')
    output_path = output_dir / f"{survey_name}_ground_truth.tif"
    
    driver = gdal.GetDriverByName('GTiff')
    height, width = labels.shape
    
    ds = driver.Create(
        str(output_path),
        width, height, 4,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    
    ds.SetGeoTransform(transform)
    if crs:
        ds.SetProjection(crs)
    
    # Band 1: Labels
    band = ds.GetRasterBand(1)
    band.WriteArray(labels.astype(np.float32))
    band.SetDescription('labels')
    band.SetNoDataValue(-1)
    
    # Band 2: Difference
    band = ds.GetRasterBand(2)
    diff_out = difference.copy()
    diff_out[~valid_mask] = np.nan
    band.WriteArray(diff_out)
    band.SetDescription('difference')
    
    # Band 3: Noisy depth
    band = ds.GetRasterBand(3)
    band.WriteArray(noisy_depth)
    band.SetDescription('noisy_depth')
    
    # Band 4: Clean depth
    band = ds.GetRasterBand(4)
    band.WriteArray(clean_depth)
    band.SetDescription('clean_depth')
    
    ds.FlushCache()
    ds = None
    
    logger.info(f"Saved ground truth: {output_path}")
    
    # Save statistics as JSON
    import json
    stats = {
        'clean_survey': str(clean_path),
        'noisy_survey': str(noisy_path),
        'noise_threshold': noise_threshold,
        'grid_shape': list(clean_depth.shape),
        'valid_cells': int(valid_count),
        'noise_cells': int(noise_count),
        'noise_percentage': float(100 * noise_count / valid_count) if valid_count > 0 else 0,
        'seafloor_cells': int(seafloor_count),
    }
    
    if np.any(noise_mask & valid_mask):
        stats['mean_noise_magnitude'] = float(np.mean(np.abs(difference[noise_mask & valid_mask])))
        stats['max_noise_magnitude'] = float(np.max(np.abs(difference[noise_mask & valid_mask])))
    
    stats_path = output_dir / f"{survey_name}_ground_truth_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved statistics: {stats_path}")
    
    return labels, difference


def main():
    parser = argparse.ArgumentParser(description='Generate ground truth from survey pairs')
    parser.add_argument('--clean', type=Path, required=True, help='Clean survey path')
    parser.add_argument('--noisy', type=Path, required=True, help='Noisy survey path')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/labels'))
    parser.add_argument('--noise-threshold', type=float, default=0.15,
                        help='Minimum depth difference for noise (meters)')
    parser.add_argument('--vr-bag-mode', default='resampled',
                        choices=['resampled', 'base', 'refinements'])
    
    args = parser.parse_args()
    
    compute_ground_truth(
        args.clean,
        args.noisy,
        args.output_dir,
        args.noise_threshold,
        args.vr_bag_mode,
    )


if __name__ == '__main__':
    main()

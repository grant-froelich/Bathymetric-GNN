#!/usr/bin/env python3
"""
scripts/prepare_ground_truth.py

Generate ground truth labels from clean/noisy survey pairs.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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


def load_survey(path: Path, vr_bag_mode: str = 'resampled') -> tuple:
    """Load survey and return depth, uncertainty, geotransform, crs."""
    loader = BathymetricLoader(vr_bag_mode=vr_bag_mode)
    grid = loader.load(path)
    return grid.depth, grid.uncertainty, grid.geotransform, grid.crs


def compute_ground_truth(
    clean_path: Path,
    noisy_path: Path,
    output_dir: Path,
    noise_threshold: float = 0.15,
    vr_bag_mode: str = 'resampled',
):
    """
    Compute ground truth labels from clean/noisy pair.
    
    Args:
        clean_path: Path to clean survey
        noisy_path: Path to noisy survey
        output_dir: Directory for output files
        noise_threshold: Minimum depth difference to classify as noise (meters)
        vr_bag_mode: How to load VR BAGs
    """
    logger.info(f"Loading clean survey: {clean_path}")
    clean_depth, clean_uncert, geotransform, crs = load_survey(clean_path, vr_bag_mode)
    
    logger.info(f"Loading noisy survey: {noisy_path}")
    noisy_depth, noisy_uncert, _, _ = load_survey(noisy_path, vr_bag_mode)
    
    # Validate shapes match
    if clean_depth.shape != noisy_depth.shape:
        raise ValueError(
            f"Shape mismatch: clean {clean_depth.shape} vs noisy {noisy_depth.shape}"
        )
    
    logger.info(f"Grid shape: {clean_depth.shape}")
    
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
    
    ds.SetGeoTransform(geotransform)
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

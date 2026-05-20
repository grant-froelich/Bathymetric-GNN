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


def compute_adaptive_threshold(abs_differences):
    """
    Find the natural break point between CUBE variability and noise
    using Otsu's method on the absolute differences.

    The difference distribution between two CUBE runs has two populations:
    small values (cells where removing noisy soundings barely changed the
    depth estimate) and large values (cells where outlier removal
    significantly changed the depth). Otsu's method finds the threshold
    that best separates these two populations by maximizing the
    between-class variance.

    Args:
        abs_differences: 1D array of absolute depth differences (valid cells only)

    Returns:
        threshold: The computed threshold in meters
    """
    # Use log-scale for better separation of heavy-tailed distributions
    # (deep water noise can have very large outliers)
    log_diffs = np.log10(np.maximum(abs_differences, 1e-6))

    # Histogram
    n_bins = 512
    hist, bin_edges = np.histogram(log_diffs, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize
    hist_norm = hist.astype(np.float64) / hist.sum()

    # Otsu: find threshold maximizing between-class variance
    w0 = np.cumsum(hist_norm)
    w1 = 1.0 - w0
    mu0_num = np.cumsum(hist_norm * bin_centers)
    mu_total = mu0_num[-1]

    # Avoid division by zero
    mu0 = np.where(w0 > 1e-10, mu0_num / w0, 0.0)
    mu1 = np.where(w1 > 1e-10, (mu_total - mu0_num) / w1, 0.0)

    sigma_between = w0 * w1 * (mu0 - mu1) ** 2

    idx = np.argmax(sigma_between)
    log_threshold = bin_centers[idx]
    threshold = 10.0 ** log_threshold

    return threshold


def warp_noisy_to_clean(noisy_path, clean_grid):
    """
    Use GDAL Warp to resample the noisy BAG onto the clean grid's
    pixel grid (same resolution, same extent, same CRS).

    Args:
        noisy_path: Path to the noisy BAG file
        clean_grid: BathymetricGrid of the clean surface

    Returns:
        Tuple of (noisy_depth, noisy_uncertainty) arrays aligned to clean grid
    """
    import tempfile
    import os as _os

    bounds = clean_grid.bounds
    x_res = clean_grid.resolution[0]
    y_res = clean_grid.resolution[1]

    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        warp_opts = gdal.WarpOptions(
            format='GTiff',
            outputBounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
            xRes=x_res,
            yRes=y_res,
            resampleAlg='bilinear',
            dstNodata=1.0e6,
        )
        gdal.Warp(tmp_path, str(noisy_path), options=warp_opts)

        warped_ds = gdal.Open(tmp_path)
        noisy_depth = warped_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        noisy_uncert = None
        if warped_ds.RasterCount >= 2:
            noisy_uncert = warped_ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        warped_ds = None
    finally:
        if _os.path.exists(tmp_path):
            _os.remove(tmp_path)

    return noisy_depth, noisy_uncert


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
    adaptive_threshold: bool = False,
    remove_offset: bool = True,
    regression_mode: bool = False,
):
    """
    Compute ground truth from clean/noisy pair.
    
    Handles surveys with different extents and resolutions.
    
    Two modes:
        Classification (default): produces labels (0=seafloor, 2=noise) using
        a threshold. Band 1 of the output is the labels band.
    
        Regression (regression_mode=True): produces continuous correction
        targets without any thresholding. Band 1 of the output is a valid
        mask. Band 2 is the correction target. The model trained on this
        learns to predict the correction at every cell, including near-zero
        corrections for cells where cleaning barely changed the depth.
    
    Args:
        clean_path: Path to clean survey
        noisy_path: Path to noisy survey
        output_dir: Directory for output files
        noise_threshold: Minimum depth difference to classify as noise (meters).
                         Only used in classification mode with fixed threshold.
        vr_bag_mode: How to load VR BAGs
        adaptive_threshold: Classification mode only. If True, compute threshold
                           from the data using Otsu's method.
        remove_offset: If True, subtract the median difference. Applies in
                      both modes. Set to False when both surfaces share the
                      same vertical datum.
        regression_mode: If True, skip thresholding and produce continuous
                        correction targets for regression training.
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
    res_mismatch = abs(clean_res[0] - noisy_res[0]) > 0.01 or abs(clean_res[1] - noisy_res[1]) > 0.01
    
    if res_mismatch:
        # Warp noisy grid to match clean grid's resolution and extent
        logger.info(
            f"Resolution mismatch: clean={clean_res[0]:.3f}m, noisy={noisy_res[0]:.3f}m. "
            f"Warping noisy grid to match clean grid."
        )
        noisy_depth, noisy_uncert = warp_noisy_to_clean(noisy_path, clean_grid)
        clean_depth = clean_grid.depth
        clean_uncert = clean_grid.uncertainty
        transform = clean_grid.transform
        crs = clean_grid.crs

        # Trim to matching shape (warp may differ by 1 pixel)
        min_rows = min(clean_depth.shape[0], noisy_depth.shape[0])
        min_cols = min(clean_depth.shape[1], noisy_depth.shape[1])
        clean_depth = clean_depth[:min_rows, :min_cols]
        noisy_depth = noisy_depth[:min_rows, :min_cols]
        if clean_uncert is not None:
            clean_uncert = clean_uncert[:min_rows, :min_cols]
        if noisy_uncert is not None:
            noisy_uncert = noisy_uncert[:min_rows, :min_cols]

        logger.info(f"Warped grid shape: {clean_depth.shape}")
    else:
        # Same resolution -- extract overlapping regions directly
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
    raw_difference = noisy_depth - clean_depth
    
    # Valid mask (both surveys have data)
    # Use actual nodata values from the grids, plus common BAG nodata sentinels
    clean_nodata = clean_grid.nodata_value
    noisy_nodata = noisy_grid.nodata_value if not res_mismatch else 1.0e6  # warp uses 1e6

    valid_clean = np.isfinite(clean_depth)
    if clean_nodata is not None and np.isfinite(clean_nodata):
        valid_clean &= (clean_depth != clean_nodata)
    # Also check common BAG nodata sentinels
    valid_clean &= (np.abs(clean_depth) < 1.0e5)

    valid_noisy = np.isfinite(noisy_depth)
    if noisy_nodata is not None and np.isfinite(noisy_nodata):
        valid_noisy &= (noisy_depth != noisy_nodata)
    valid_noisy &= (np.abs(noisy_depth) < 1.0e5)

    valid_mask = valid_clean & valid_noisy

    logger.info(f"  Clean nodata value: {clean_nodata}")
    logger.info(f"  Noisy nodata value: {noisy_nodata}")
    logger.info(f"  Valid cells: {np.sum(valid_mask):,} / {valid_mask.size:,}")
    
    # Remove systematic offset between surveys (optional)
    valid_diff = raw_difference[valid_mask]
    systematic_offset = np.median(valid_diff)

    if remove_offset:
        logger.info(f"Detected systematic offset: {systematic_offset:.3f}m (will be removed)")
        difference = raw_difference - systematic_offset
    else:
        logger.info(f"Median difference: {systematic_offset:.3f}m (offset removal disabled)")
        difference = raw_difference.copy()
    
    # Determine noise threshold (classification mode only)
    if regression_mode:
        logger.info("Regression mode: skipping threshold computation")
        noise_threshold = None
    elif adaptive_threshold:
        abs_diffs = np.abs(difference[valid_mask])
        noise_threshold = compute_adaptive_threshold(abs_diffs)
        logger.info(f"Adaptive threshold (Otsu): {noise_threshold:.4f}m")
    else:
        logger.info(f"Fixed noise threshold: {noise_threshold}m")
    
    # Build labels (classification) or valid mask (regression)
    valid_count = int(np.sum(valid_mask))

    if regression_mode:
        # Band 1 in regression mode is a valid mask, not classification labels
        # 1 = valid for training, 0 = invalid (excluded)
        labels = valid_mask.astype(np.int32)

        # Statistics for regression
        valid_diffs = difference[valid_mask]
        abs_valid_diffs = np.abs(valid_diffs)

        logger.info(f"Regression target statistics:")
        logger.info(f"  Valid cells: {valid_count:,}")
        logger.info(f"  Correction magnitude:")
        logger.info(f"    Mean:    {np.mean(abs_valid_diffs):.4f}m")
        logger.info(f"    Median:  {np.median(abs_valid_diffs):.4f}m")
        logger.info(f"    90th %:  {np.percentile(abs_valid_diffs, 90):.4f}m")
        logger.info(f"    99th %:  {np.percentile(abs_valid_diffs, 99):.4f}m")
        logger.info(f"    Max:     {np.max(abs_valid_diffs):.4f}m")
        logger.info(f"  Correction direction:")
        shoal_pct = 100.0 * np.mean(valid_diffs < 0)  # noisy shallower than clean
        deep_pct = 100.0 * np.mean(valid_diffs > 0)   # noisy deeper than clean
        logger.info(f"    Shoal direction: {shoal_pct:.1f}%")
        logger.info(f"    Deep direction:  {deep_pct:.1f}%")

        # No noise_count/seafloor_count in regression mode
        noise_count = 0
        seafloor_count = valid_count
        noise_mask = np.zeros_like(valid_mask)
    else:
        # Classification mode: produce labels
        labels = np.full(clean_depth.shape, CLASS_SEAFLOOR, dtype=np.int32)

        # Mark noise where difference exceeds threshold
        noise_mask = np.abs(difference) > noise_threshold
        labels[noise_mask & valid_mask] = CLASS_NOISE

        # Mark invalid areas
        labels[~valid_mask] = -1  # NoData

        # Statistics
        noise_count = int(np.sum(labels == CLASS_NOISE))
        seafloor_count = int(np.sum(labels == CLASS_SEAFLOOR))

        logger.info(f"Ground truth statistics:")
        logger.info(f"  Valid cells: {valid_count:,}")
        logger.info(f"  Noise cells: {noise_count:,} ({100*noise_count/valid_count:.2f}%)")
        logger.info(f"  Seafloor cells: {seafloor_count:,} ({100*seafloor_count/valid_count:.2f}%)")

        if np.any(noise_mask & valid_mask):
            logger.info(f"  Mean noise magnitude: {np.mean(np.abs(difference[noise_mask & valid_mask])):.3f}m")
        logger.info(f"  Max noise magnitude: {np.max(np.abs(difference[noise_mask & valid_mask])):.3f}m")
    
    # Verify offset removal worked (classification mode only)
    if not regression_mode:
        seafloor_diff = difference[(labels == CLASS_SEAFLOOR) & valid_mask]
        if len(seafloor_diff) > 0:
            logger.info(f"  Seafloor mean diff (should be ~0): {np.mean(seafloor_diff):.3f}m")
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    survey_name = clean_path.stem.replace('_clean', '').replace('_Clean', '')
    suffix = '_regression' if regression_mode else '_ground_truth'
    output_path = output_dir / f"{survey_name}{suffix}.tif"
    
    driver = gdal.GetDriverByName('GTiff')
    height, width = labels.shape
    
    ds = driver.Create(
        str(output_path),
        width, height, 5,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    
    ds.SetGeoTransform(transform)
    if crs:
        ds.SetProjection(crs)
    
    # Band 1: Labels (classification) or valid mask (regression)
    band = ds.GetRasterBand(1)
    band.WriteArray(labels.astype(np.float32))
    if regression_mode:
        band.SetDescription('valid_mask')
        band.SetNoDataValue(0)
    else:
        band.SetDescription('labels')
        band.SetNoDataValue(-1)
    
    # Band 2: Difference / correction target
    band = ds.GetRasterBand(2)
    diff_out = difference.copy()
    diff_out[~valid_mask] = np.nan
    band.WriteArray(diff_out)
    band.SetDescription('correction' if regression_mode else 'difference')
    
    # Band 3: Noisy depth
    band = ds.GetRasterBand(3)
    band.WriteArray(noisy_depth)
    band.SetDescription('noisy_depth')
    
    # Band 4: Clean depth
    band = ds.GetRasterBand(4)
    band.WriteArray(clean_depth)
    band.SetDescription('clean_depth')
    
    # Band 5: Uncertainty (from noisy survey)
    band = ds.GetRasterBand(5)
    if noisy_uncert is not None:
        uncert_out = noisy_uncert.copy()
        uncert_out[~valid_mask] = np.nan
        band.WriteArray(uncert_out)
    else:
        # No uncertainty available, fill with NaN
        band.WriteArray(np.full((height, width), np.nan, dtype=np.float32))
    band.SetDescription('uncertainty')
    
    ds.FlushCache()
    ds = None
    
    logger.info(f"Saved ground truth: {output_path}")
    
    # Save statistics as JSON
    import json
    stats = {
        'clean_survey': str(clean_path),
        'noisy_survey': str(noisy_path),
        'mode': 'regression' if regression_mode else 'classification',
        'offset_removed': remove_offset,
        'systematic_offset': float(systematic_offset),
        'grid_shape': list(clean_depth.shape),
        'valid_cells': int(valid_count),
    }

    if regression_mode:
        valid_diffs = difference[valid_mask]
        abs_valid_diffs = np.abs(valid_diffs)
        stats['correction_stats'] = {
            'mean_magnitude': float(np.mean(abs_valid_diffs)),
            'median_magnitude': float(np.median(abs_valid_diffs)),
            'p90_magnitude': float(np.percentile(abs_valid_diffs, 90)),
            'p99_magnitude': float(np.percentile(abs_valid_diffs, 99)),
            'max_magnitude': float(np.max(abs_valid_diffs)),
            'shoal_direction_pct': float(100.0 * np.mean(valid_diffs < 0)),
            'deep_direction_pct': float(100.0 * np.mean(valid_diffs > 0)),
        }
    else:
        stats['noise_threshold'] = float(noise_threshold)
        stats['threshold_mode'] = 'adaptive' if adaptive_threshold else 'fixed'
        stats['noise_cells'] = int(noise_count)
        stats['noise_percentage'] = float(100 * noise_count / valid_count) if valid_count > 0 else 0
        stats['seafloor_cells'] = int(seafloor_count)
        if np.any(noise_mask & valid_mask):
            stats['mean_noise_magnitude'] = float(np.mean(np.abs(difference[noise_mask & valid_mask])))
            stats['max_noise_magnitude'] = float(np.max(np.abs(difference[noise_mask & valid_mask])))

    stats_suffix = '_regression_stats.json' if regression_mode else '_ground_truth_stats.json'
    stats_path = output_dir / f"{survey_name}{stats_suffix}"
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
                        help='Fixed noise threshold in meters (default: 0.15). '
                             'Ignored when --adaptive-threshold is set.')
    parser.add_argument('--adaptive-threshold', action='store_true',
                        help='Compute noise threshold from the data using Otsu\'s method '
                             'instead of using the fixed --noise-threshold value. '
                             'Finds the natural break between CUBE variability and noise.')
    parser.add_argument('--no-offset', action='store_true',
                        help='Skip median offset removal. Use when both surfaces '
                             'are in the same vertical datum.')
    parser.add_argument('--regression-mode', action='store_true',
                        help='Produce continuous correction targets for regression '
                             'training instead of binary classification labels. '
                             'No threshold is applied. Band 1 of the output becomes '
                             'a valid mask, band 2 is the correction target in meters.')
    parser.add_argument('--vr-bag-mode', default='resampled',
                        choices=['resampled', 'base', 'refinements'])
    
    args = parser.parse_args()
    
    compute_ground_truth(
        args.clean,
        args.noisy,
        args.output_dir,
        args.noise_threshold,
        args.vr_bag_mode,
        adaptive_threshold=args.adaptive_threshold,
        remove_offset=not args.no_offset,
        regression_mode=args.regression_mode,
    )


if __name__ == '__main__':
    main()

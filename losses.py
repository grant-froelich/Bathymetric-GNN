#!/usr/bin/env python3
"""
scripts/analyze_noise_patterns.py

Analyze noise patterns from ground truth to improve synthetic noise generation.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
from pathlib import Path
import numpy as np
from osgeo import gdal
import json
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def analyze_ground_truth(path: Path) -> dict:
    """Analyze noise patterns in ground truth file."""
    
    ds = gdal.Open(str(path))
    labels = ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
    difference = ds.GetRasterBand(2).ReadAsArray()
    noisy_depth = ds.GetRasterBand(3).ReadAsArray()
    clean_depth = ds.GetRasterBand(4).ReadAsArray()
    ds = None
    
    valid = labels >= 0
    noise_mask = labels == 2
    
    analysis = {
        'file': str(path),
        'shape': list(labels.shape),
    }
    
    if not np.any(noise_mask & valid):
        logger.warning(f"No noise found in {path}")
        return analysis
    
    # 1. Noise magnitude distribution
    noise_magnitudes = np.abs(difference[noise_mask & valid])
    analysis['magnitude'] = {
        'mean': float(np.mean(noise_magnitudes)),
        'std': float(np.std(noise_magnitudes)),
        'median': float(np.median(noise_magnitudes)),
        'p90': float(np.percentile(noise_magnitudes, 90)),
        'p99': float(np.percentile(noise_magnitudes, 99)),
        'max': float(np.max(noise_magnitudes)),
    }
    
    # 2. Noise sign distribution
    noise_diffs = difference[noise_mask & valid]
    analysis['sign'] = {
        'positive_ratio': float(np.mean(noise_diffs > 0)),
        'negative_ratio': float(np.mean(noise_diffs < 0)),
        'mean_positive': float(np.mean(noise_diffs[noise_diffs > 0])) if np.any(noise_diffs > 0) else 0,
        'mean_negative': float(np.mean(noise_diffs[noise_diffs < 0])) if np.any(noise_diffs < 0) else 0,
    }
    
    # 3. Depth-dependent noise
    depth_bins = [0, 10, 20, 50, 100, 200, 500, 1000, 10000]
    depth_noise = {}
    for i in range(len(depth_bins) - 1):
        lo, hi = depth_bins[i], depth_bins[i + 1]
        depth_range = (-noisy_depth >= lo) & (-noisy_depth < hi) & noise_mask & valid
        total_in_range = np.sum((-noisy_depth >= lo) & (-noisy_depth < hi) & valid)
        if np.sum(depth_range) > 100 and total_in_range > 0:
            depth_noise[f'{lo}-{hi}m'] = {
                'count': int(np.sum(depth_range)),
                'mean_magnitude': float(np.mean(np.abs(difference[depth_range]))),
                'noise_rate': float(np.sum(depth_range) / total_in_range),
            }
    analysis['depth_dependent'] = depth_noise
    
    # 4. Spatial clustering
    labeled, num_clusters = ndimage.label(noise_mask)
    if num_clusters > 0:
        cluster_sizes = ndimage.sum(noise_mask, labeled, range(1, num_clusters + 1))
        
        analysis['clustering'] = {
            'num_clusters': int(num_clusters),
            'mean_cluster_size': float(np.mean(cluster_sizes)),
            'median_cluster_size': float(np.median(cluster_sizes)),
            'max_cluster_size': int(np.max(cluster_sizes)),
            'isolated_noise_ratio': float(np.sum(cluster_sizes == 1) / len(cluster_sizes)),
        }
        
        # Size distribution
        size_bins = [1, 2, 5, 10, 50, 100, 1000, 10000]
        size_dist = {}
        for i in range(len(size_bins) - 1):
            lo, hi = size_bins[i], size_bins[i + 1]
            count = np.sum((cluster_sizes >= lo) & (cluster_sizes < hi))
            size_dist[f'{lo}-{hi}'] = int(count)
        analysis['clustering']['size_distribution'] = size_dist
    
    # 5. Row/column patterns
    col_valid = np.maximum(np.sum(valid, axis=0), 1)
    noise_by_col = np.sum(noise_mask, axis=0) / col_valid
    col_quartiles = np.array_split(noise_by_col, 4)
    
    analysis['swath_pattern'] = {
        'left_quarter_noise_rate': float(np.mean(col_quartiles[0])),
        'center_left_noise_rate': float(np.mean(col_quartiles[1])),
        'center_right_noise_rate': float(np.mean(col_quartiles[2])),
        'right_quarter_noise_rate': float(np.mean(col_quartiles[3])),
    }
    
    # 6. Local roughness context
    def local_std(x):
        return np.nanstd(x) if len(x) > 0 else 0
    
    roughness = ndimage.generic_filter(
        clean_depth.astype(np.float64), 
        local_std, 
        size=5, 
        mode='constant', 
        cval=np.nan
    )
    
    noise_roughness = roughness[noise_mask & valid]
    seafloor_roughness = roughness[(labels == 0) & valid]
    
    analysis['roughness_context'] = {
        'noise_mean_roughness': float(np.nanmean(noise_roughness)),
        'noise_median_roughness': float(np.nanmedian(noise_roughness)),
        'seafloor_mean_roughness': float(np.nanmean(seafloor_roughness)),
        'seafloor_median_roughness': float(np.nanmedian(seafloor_roughness)),
    }
    
    return analysis


def print_analysis(analysis: dict):
    """Print analysis in readable format."""
    print("\n" + "=" * 60)
    print(f"NOISE PATTERN ANALYSIS: {Path(analysis['file']).name}")
    print("=" * 60)
    
    print(f"\nGrid shape: {analysis['shape']}")
    
    if 'magnitude' not in analysis:
        print("No noise found in this file.")
        return
    
    print("\n1. Noise Magnitude:")
    m = analysis['magnitude']
    print(f"   Mean: {m['mean']:.3f}m, Std: {m['std']:.3f}m")
    print(f"   Median: {m['median']:.3f}m, 90th percentile: {m['p90']:.3f}m")
    print(f"   Max: {m['max']:.3f}m")
    
    print("\n2. Noise Direction:")
    s = analysis['sign']
    print(f"   Deepening (positive): {100*s['positive_ratio']:.1f}% (mean: {s['mean_positive']:.3f}m)")
    print(f"   Shoaling (negative): {100*s['negative_ratio']:.1f}% (mean: {s['mean_negative']:.3f}m)")
    
    print("\n3. Depth-dependent noise rate:")
    for depth_range, stats in analysis.get('depth_dependent', {}).items():
        print(f"   {depth_range}: {100*stats['noise_rate']:.2f}% noise rate, {stats['mean_magnitude']:.3f}m mean magnitude")
    
    if 'clustering' in analysis:
        print("\n4. Spatial Clustering:")
        c = analysis['clustering']
        print(f"   Number of clusters: {c['num_clusters']:,}")
        print(f"   Mean cluster size: {c['mean_cluster_size']:.1f} cells")
        print(f"   Isolated points: {100*c['isolated_noise_ratio']:.1f}%")
    
    print("\n5. Swath Pattern (noise rate by column quartile):")
    sp = analysis['swath_pattern']
    print(f"   Left (outer): {100*sp['left_quarter_noise_rate']:.2f}%")
    print(f"   Center-left: {100*sp['center_left_noise_rate']:.2f}%")
    print(f"   Center-right: {100*sp['center_right_noise_rate']:.2f}%")
    print(f"   Right (outer): {100*sp['right_quarter_noise_rate']:.2f}%")
    
    print("\n6. Roughness Context:")
    r = analysis['roughness_context']
    print(f"   Noise in areas with roughness: {r['noise_mean_roughness']:.4f} (mean)")
    print(f"   Seafloor in areas with roughness: {r['seafloor_mean_roughness']:.4f} (mean)")
    if r['seafloor_mean_roughness'] > 0:
        ratio = r['noise_mean_roughness'] / r['seafloor_mean_roughness']
        print(f"   Ratio: {ratio:.2f}x (>1 means noise in rougher areas)")


def main():
    parser = argparse.ArgumentParser(description='Analyze noise patterns in ground truth')
    parser.add_argument('--input', type=Path, required=True, nargs='+',
                        help='Ground truth file(s)')
    parser.add_argument('--output', type=Path, help='Save combined analysis to JSON')
    
    args = parser.parse_args()
    
    all_analyses = []
    for path in args.input:
        if path.exists():
            logger.info(f"Analyzing: {path}")
            analysis = analyze_ground_truth(path)
            print_analysis(analysis)
            all_analyses.append(analysis)
        else:
            logger.warning(f"File not found: {path}")
    
    if args.output and all_analyses:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_analyses, f, indent=2)
        logger.info(f"Saved analysis: {args.output}")


if __name__ == '__main__':
    main()

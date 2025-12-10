# Bathymetric GNN Training Plan

Detailed plan for training a production-quality model for bathymetric noise detection.

## Current State

- Model trained on synthetic noise only
- 50 epochs, 5 clean VR BAGs, 100 tiles
- Accuracy on synthetic data: 63.2%
- Real-world performance: Unknown (100% classified as noise, but low confidence indicates model uncertainty)

## Directory Structure

```
bathymetric-gnn/
├── data/
│   ├── raw/
│   │   ├── clean/              # Clean reference surveys
│   │   ├── noisy/              # Corresponding noisy versions
│   │   └── features/           # Surveys with known features
│   ├── processed/
│   │   ├── labels/             # Generated ground truth labels
│   │   ├── train/              # Training tiles
│   │   └── validation/         # Validation tiles
│   └── external/
│       └── wrecks/             # NOAA wreck database exports
├── outputs/
│   ├── models/                 # Saved model checkpoints
│   ├── metrics/                # Training metrics logs
│   └── analysis/               # Noise pattern analysis
└── scripts/
    ├── prepare_ground_truth.py
    ├── analyze_noise_patterns.py
    ├── prepare_feature_labels.py
    └── train_v2.py
```

---

## Phase 1: Establish Ground Truth

**Duration:** 1-2 weeks  
**Goal:** Create labeled validation data from real clean/noisy survey pairs

### Step 1.1: Organize Survey Pairs

Place matching surveys in the data directory:

```
data/raw/clean/survey_001_clean.bag
data/raw/noisy/survey_001_noisy.bag
data/raw/clean/survey_002_clean.bag
data/raw/noisy/survey_002_noisy.bag
```

### Step 1.2: Create Ground Truth Script

```python
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
    # Positive = noisy is deeper than clean (noise added depth)
    # Negative = noisy is shallower than clean (noise removed depth)
    difference = noisy_depth - clean_depth
    
    # Valid mask (both surveys have data)
    nodata = 1.0e6
    valid_clean = (clean_depth != nodata) & np.isfinite(clean_depth)
    valid_noisy = (noisy_depth != nodata) & np.isfinite(noisy_depth)
    valid_mask = valid_clean & valid_noisy
    
    # Classify based on difference
    # Start with seafloor (default)
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
    logger.info(f"  Mean noise magnitude: {np.mean(np.abs(difference[noise_mask & valid_mask])):.3f}m")
    logger.info(f"  Max noise magnitude: {np.max(np.abs(difference[noise_mask & valid_mask])):.3f}m")
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    survey_name = clean_path.stem.replace('_clean', '')
    
    # Save as GeoTIFF with multiple bands
    output_path = output_dir / f"{survey_name}_ground_truth.tif"
    
    driver = gdal.GetDriverByName('GTiff')
    height, width = labels.shape
    
    # 4 bands: labels, difference, noisy_depth, clean_depth
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
    
    # Band 2: Difference (correction that was applied)
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
    
    # Also save statistics as JSON
    import json
    stats = {
        'clean_survey': str(clean_path),
        'noisy_survey': str(noisy_path),
        'noise_threshold': noise_threshold,
        'grid_shape': list(clean_depth.shape),
        'valid_cells': int(valid_count),
        'noise_cells': int(noise_count),
        'noise_percentage': float(100 * noise_count / valid_count),
        'seafloor_cells': int(seafloor_count),
        'mean_noise_magnitude': float(np.mean(np.abs(difference[noise_mask & valid_mask]))),
        'max_noise_magnitude': float(np.max(np.abs(difference[noise_mask & valid_mask]))),
        'noise_depth_distribution': {
            'mean': float(np.mean(noisy_depth[noise_mask & valid_mask])),
            'std': float(np.std(noisy_depth[noise_mask & valid_mask])),
            'min': float(np.min(noisy_depth[noise_mask & valid_mask])),
            'max': float(np.max(noisy_depth[noise_mask & valid_mask])),
        }
    }
    
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
```

### Step 1.3: Run Ground Truth Generation

```cmd
:: For each survey pair
python scripts/prepare_ground_truth.py ^
    --clean data/raw/clean/survey_001_clean.bag ^
    --noisy data/raw/noisy/survey_001_noisy.bag ^
    --output-dir data/processed/labels ^
    --noise-threshold 0.15 ^
    --vr-bag-mode resampled

:: Repeat for all pairs
python scripts/prepare_ground_truth.py ^
    --clean data/raw/clean/survey_002_clean.bag ^
    --noisy data/raw/noisy/survey_002_noisy.bag ^
    --output-dir data/processed/labels
```

### Step 1.4: Evaluate Current Model Against Ground Truth

```python
#!/usr/bin/env python3
"""
scripts/evaluate_model.py

Evaluate model predictions against ground truth.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
from pathlib import Path
import numpy as np
from osgeo import gdal
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = ['seafloor', 'feature', 'noise']


def load_ground_truth(path: Path):
    """Load ground truth GeoTIFF."""
    ds = gdal.Open(str(path))
    labels = ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
    ds = None
    return labels


def load_predictions(path: Path):
    """Load model predictions GeoTIFF."""
    ds = gdal.Open(str(path))
    
    # Find bands
    classification = None
    confidence = None
    
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        desc = band.GetDescription().lower()
        if 'class' in desc:
            classification = band.ReadAsArray()
        elif 'confid' in desc:
            confidence = band.ReadAsArray()
    
    ds = None
    return classification, confidence


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray = None):
    """Compute per-class and overall metrics."""
    
    # Valid mask (ground truth has label)
    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if confidence is not None:
        confidence = confidence[valid]
    
    metrics = {
        'total_samples': int(len(y_true)),
        'overall_accuracy': float(np.mean(y_true == y_pred)),
    }
    
    # Per-class metrics
    for class_idx, class_name in enumerate(CLASS_NAMES):
        true_pos = np.sum((y_true == class_idx) & (y_pred == class_idx))
        false_pos = np.sum((y_true != class_idx) & (y_pred == class_idx))
        false_neg = np.sum((y_true == class_idx) & (y_pred != class_idx))
        true_neg = np.sum((y_true != class_idx) & (y_pred != class_idx))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'true_positives': int(true_pos),
            'false_positives': int(false_pos),
            'false_negatives': int(false_neg),
            'true_negatives': int(true_neg),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(y_true == class_idx)),
        }
    
    # Confusion matrix
    num_classes = len(CLASS_NAMES)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion[i, j] = np.sum((y_true == i) & (y_pred == j))
    
    metrics['confusion_matrix'] = confusion.tolist()
    
    # Confidence analysis
    if confidence is not None:
        metrics['confidence'] = {
            'mean': float(np.mean(confidence)),
            'std': float(np.std(confidence)),
            'mean_correct': float(np.mean(confidence[y_true == y_pred])),
            'mean_incorrect': float(np.mean(confidence[y_true != y_pred])),
        }
        
        # Accuracy at different confidence thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            mask = confidence >= thresh
            if np.sum(mask) > 0:
                acc = np.mean(y_true[mask] == y_pred[mask])
                coverage = np.mean(mask)
                metrics['confidence'][f'accuracy_at_{thresh}'] = float(acc)
                metrics['confidence'][f'coverage_at_{thresh}'] = float(coverage)
    
    return metrics


def print_metrics(metrics: dict):
    """Print metrics in readable format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nTotal samples: {metrics['total_samples']:,}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    
    print("\nPer-class metrics:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    
    for class_name in CLASS_NAMES:
        m = metrics[class_name]
        print(f"{class_name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10,}")
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    print(f"{'':>12} " + " ".join(f"{name:>10}" for name in CLASS_NAMES) + "  <- Predicted")
    
    cm = np.array(metrics['confusion_matrix'])
    for i, class_name in enumerate(CLASS_NAMES):
        row = " ".join(f"{cm[i,j]:>10,}" for j in range(len(CLASS_NAMES)))
        print(f"{class_name:>12} {row}")
    print("Actual ^")
    
    if 'confidence' in metrics:
        print("\nConfidence Analysis:")
        print("-" * 60)
        c = metrics['confidence']
        print(f"Mean confidence: {c['mean']:.4f}")
        print(f"Mean confidence (correct): {c['mean_correct']:.4f}")
        print(f"Mean confidence (incorrect): {c['mean_incorrect']:.4f}")
        print("\nAccuracy vs Coverage at confidence thresholds:")
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            acc_key = f'accuracy_at_{thresh}'
            cov_key = f'coverage_at_{thresh}'
            if acc_key in c:
                print(f"  >= {thresh}: Accuracy {c[acc_key]:.4f}, Coverage {c[cov_key]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model against ground truth')
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Save metrics to JSON')
    
    args = parser.parse_args()
    
    logger.info(f"Loading ground truth: {args.ground_truth}")
    labels = load_ground_truth(args.ground_truth)
    
    logger.info(f"Loading predictions: {args.predictions}")
    predictions, confidence = load_predictions(args.predictions)
    
    if labels.shape != predictions.shape:
        raise ValueError(f"Shape mismatch: GT {labels.shape} vs Pred {predictions.shape}")
    
    metrics = compute_metrics(labels, predictions, confidence)
    print_metrics(metrics)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {args.output}")


if __name__ == '__main__':
    main()
```

### Step 1.5: Run Evaluation

```cmd
:: First run inference on the noisy survey
python scripts/inference.py ^
    --input data/raw/noisy/survey_001_noisy.bag ^
    --model outputs/final_model.pt ^
    --output outputs/predictions/survey_001_predictions.tif ^
    --vr-bag-mode resampled ^
    --min-valid-ratio 0.01

:: Then evaluate against ground truth
python scripts/evaluate_model.py ^
    --ground-truth data/processed/labels/survey_001_ground_truth.tif ^
    --predictions outputs/predictions/survey_001_predictions.tif ^
    --output outputs/metrics/survey_001_evaluation.json
```

---

## Phase 2: Analyze Noise Patterns

**Duration:** 1 week  
**Goal:** Understand real noise characteristics to improve synthetic generation

### Step 2.1: Noise Pattern Analysis Script

```python
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
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    
    # Masks
    valid = labels >= 0
    noise_mask = labels == 2
    
    analysis = {
        'file': str(path),
        'shape': list(labels.shape),
    }
    
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
    
    # 2. Noise sign distribution (shoaling vs deepening)
    noise_diffs = difference[noise_mask & valid]
    analysis['sign'] = {
        'positive_ratio': float(np.mean(noise_diffs > 0)),  # Noisy deeper than clean
        'negative_ratio': float(np.mean(noise_diffs < 0)),  # Noisy shallower than clean
        'mean_positive': float(np.mean(noise_diffs[noise_diffs > 0])) if np.any(noise_diffs > 0) else 0,
        'mean_negative': float(np.mean(noise_diffs[noise_diffs < 0])) if np.any(noise_diffs < 0) else 0,
    }
    
    # 3. Depth-dependent noise
    depth_bins = [0, 10, 20, 50, 100, 200, 500, 1000, 10000]
    depth_noise = {}
    for i in range(len(depth_bins) - 1):
        lo, hi = depth_bins[i], depth_bins[i + 1]
        # Use absolute depth (negate since BAG stores as negative)
        depth_range = (-noisy_depth >= lo) & (-noisy_depth < hi) & noise_mask & valid
        if np.sum(depth_range) > 100:
            depth_noise[f'{lo}-{hi}m'] = {
                'count': int(np.sum(depth_range)),
                'mean_magnitude': float(np.mean(np.abs(difference[depth_range]))),
                'noise_rate': float(np.sum(depth_range) / np.sum((-noisy_depth >= lo) & (-noisy_depth < hi) & valid)),
            }
    analysis['depth_dependent'] = depth_noise
    
    # 4. Spatial clustering
    # Label connected components of noise
    labeled, num_clusters = ndimage.label(noise_mask)
    cluster_sizes = ndimage.sum(noise_mask, labeled, range(1, num_clusters + 1))
    
    analysis['clustering'] = {
        'num_clusters': int(num_clusters),
        'mean_cluster_size': float(np.mean(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
        'median_cluster_size': float(np.median(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
        'max_cluster_size': int(np.max(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
        'isolated_noise_ratio': float(np.sum(cluster_sizes == 1) / len(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
    }
    
    # Size distribution
    size_bins = [1, 2, 5, 10, 50, 100, 1000, 10000]
    size_dist = {}
    for i in range(len(size_bins) - 1):
        lo, hi = size_bins[i], size_bins[i + 1]
        count = np.sum((cluster_sizes >= lo) & (cluster_sizes < hi))
        size_dist[f'{lo}-{hi}'] = int(count)
    analysis['clustering']['size_distribution'] = size_dist
    
    # 5. Row/column patterns (detect swath artifacts)
    # Check if noise is concentrated in certain columns (outer beams)
    noise_by_col = np.sum(noise_mask, axis=0) / np.maximum(np.sum(valid, axis=0), 1)
    col_quartiles = np.array_split(noise_by_col, 4)
    
    analysis['swath_pattern'] = {
        'left_quarter_noise_rate': float(np.mean(col_quartiles[0])),
        'center_left_noise_rate': float(np.mean(col_quartiles[1])),
        'center_right_noise_rate': float(np.mean(col_quartiles[2])),
        'right_quarter_noise_rate': float(np.mean(col_quartiles[3])),
    }
    
    # 6. Local roughness context
    # Is noise more common in rough or smooth areas?
    from scipy.ndimage import generic_filter
    
    def local_std(x):
        return np.std(x) if len(x) > 0 else 0
    
    # Compute local roughness (std in 5x5 window)
    roughness = generic_filter(clean_depth, local_std, size=5, mode='constant', cval=np.nan)
    
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
    for depth_range, stats in analysis['depth_dependent'].items():
        print(f"   {depth_range}: {100*stats['noise_rate']:.2f}% noise rate, {stats['mean_magnitude']:.3f}m mean magnitude")
    
    print("\n4. Spatial Clustering:")
    c = analysis['clustering']
    print(f"   Number of clusters: {c['num_clusters']:,}")
    print(f"   Mean cluster size: {c['mean_cluster_size']:.1f} cells")
    print(f"   Isolated points: {100*c['isolated_noise_ratio']:.1f}%")
    print(f"   Size distribution: {c['size_distribution']}")
    
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
    ratio = r['noise_mean_roughness'] / r['seafloor_mean_roughness'] if r['seafloor_mean_roughness'] > 0 else 0
    print(f"   Ratio: {ratio:.2f}x (>1 means noise in rougher areas)")


def main():
    parser = argparse.ArgumentParser(description='Analyze noise patterns in ground truth')
    parser.add_argument('--input', type=Path, required=True, nargs='+',
                        help='Ground truth file(s)')
    parser.add_argument('--output', type=Path, help='Save combined analysis to JSON')
    
    args = parser.parse_args()
    
    all_analyses = []
    for path in args.input:
        logger.info(f"Analyzing: {path}")
        analysis = analyze_ground_truth(path)
        print_analysis(analysis)
        all_analyses.append(analysis)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_analyses, f, indent=2)
        logger.info(f"Saved analysis: {args.output}")
    
    # Print summary if multiple files
    if len(all_analyses) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY ACROSS ALL FILES")
        print("=" * 60)
        
        mean_magnitude = np.mean([a['magnitude']['mean'] for a in all_analyses])
        mean_cluster_size = np.mean([a['clustering']['mean_cluster_size'] for a in all_analyses])
        mean_isolated = np.mean([a['clustering']['isolated_noise_ratio'] for a in all_analyses])
        
        print(f"Average noise magnitude: {mean_magnitude:.3f}m")
        print(f"Average cluster size: {mean_cluster_size:.1f} cells")
        print(f"Average isolated noise: {100*mean_isolated:.1f}%")


if __name__ == '__main__':
    main()
```

### Step 2.2: Run Analysis

```cmd
:: Analyze all ground truth files
python scripts/analyze_noise_patterns.py ^
    --input data/processed/labels/*.tif ^
    --output outputs/analysis/noise_patterns.json
```

### Step 2.3: Update Synthetic Noise Generator

Based on analysis results, update `data/synthetic_noise.py` parameters:

```python
# Example updates based on analysis findings

# If analysis shows outer beam noise is 2x center:
swath_noise_profile = [2.0, 1.0, 1.0, 2.0]  # Left, center-left, center-right, right

# If analysis shows 30% isolated noise, 70% clustered:
cluster_probability = 0.70

# If analysis shows depth-dependent noise:
def get_noise_magnitude(depth):
    if depth < 20:
        return 0.15  # Shallow
    elif depth < 100:
        return 0.25  # Mid
    else:
        return 0.40  # Deep
```

---

## Phase 3: Feature Class Training

**Duration:** 2-3 weeks  
**Goal:** Add explicit feature examples (shipwrecks, rocks, structures)

### Step 3.1: Collect Feature Locations

**Source 1: NOAA Wrecks Database**

Download from: https://nauticalcharts.noaa.gov/data/wrecks-and-obstructions.html

```python
#!/usr/bin/env python3
"""
scripts/prepare_feature_labels.py

Create feature class labels from external sources.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
from pathlib import Path
import numpy as np
from osgeo import gdal, ogr, osr
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import BathymetricLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_wrecks_shapefile(shapefile_path: Path):
    """Load wreck locations from shapefile."""
    ds = ogr.Open(str(shapefile_path))
    layer = ds.GetLayer()
    
    wrecks = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom:
            x, y = geom.GetX(), geom.GetY()
            wrecks.append({
                'x': x,
                'y': y,
                'name': feature.GetField('VESSLTERMS') or 'Unknown',
                'depth': feature.GetField('DEPTH') or 0,
            })
    
    ds = None
    return wrecks


def create_feature_mask_from_slope(
    depth: np.ndarray,
    slope_threshold: float = 15.0,  # degrees
    min_cluster_size: int = 100,
) -> np.ndarray:
    """
    Create feature mask based on local slope.
    
    High slope areas are likely rocky terrain or features.
    """
    from scipy import ndimage
    
    # Compute gradients
    gy, gx = np.gradient(depth)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    
    # Threshold
    feature_mask = slope > slope_threshold
    
    # Remove small clusters (noise)
    labeled, num_features = ndimage.label(feature_mask)
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) < min_cluster_size:
            feature_mask[labeled == i] = False
    
    return feature_mask


def add_feature_labels(
    survey_path: Path,
    output_path: Path,
    wrecks_shapefile: Path = None,
    slope_threshold: float = 15.0,
    wreck_radius: int = 50,  # pixels
    vr_bag_mode: str = 'resampled',
):
    """
    Create labels with feature class from slope analysis and wreck locations.
    """
    loader = BathymetricLoader(vr_bag_mode=vr_bag_mode)
    grid = loader.load(survey_path)
    
    depth = grid.depth
    valid = (depth != 1.0e6) & np.isfinite(depth)
    
    # Initialize with seafloor
    labels = np.zeros(depth.shape, dtype=np.int32)
    labels[~valid] = -1
    
    # Add features from slope
    logger.info("Computing slope-based features...")
    slope_features = create_feature_mask_from_slope(depth, slope_threshold)
    labels[slope_features & valid] = 1  # Feature class
    
    slope_feature_count = np.sum(slope_features & valid)
    logger.info(f"Slope-based features: {slope_feature_count:,} cells")
    
    # Add features from wreck locations
    if wrecks_shapefile and wrecks_shapefile.exists():
        logger.info(f"Loading wrecks from: {wrecks_shapefile}")
        wrecks = load_wrecks_shapefile(wrecks_shapefile)
        
        # Convert wreck coordinates to pixel coordinates
        gt = grid.geotransform
        wreck_count = 0
        
        for wreck in wrecks:
            # Geographic to pixel
            col = int((wreck['x'] - gt[0]) / gt[1])
            row = int((wreck['y'] - gt[3]) / gt[5])
            
            # Check if in bounds
            if 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
                # Create circular mask around wreck
                yy, xx = np.ogrid[:depth.shape[0], :depth.shape[1]]
                dist = np.sqrt((xx - col)**2 + (yy - row)**2)
                wreck_mask = dist <= wreck_radius
                
                labels[wreck_mask & valid] = 1
                wreck_count += 1
        
        logger.info(f"Added {wreck_count} wrecks from database")
    
    # Statistics
    feature_count = np.sum(labels == 1)
    seafloor_count = np.sum(labels == 0)
    logger.info(f"Final labels: {seafloor_count:,} seafloor, {feature_count:,} feature")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        str(output_path),
        depth.shape[1], depth.shape[0], 2,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    
    ds.SetGeoTransform(grid.geotransform)
    if grid.crs:
        ds.SetProjection(grid.crs)
    
    band = ds.GetRasterBand(1)
    band.WriteArray(labels.astype(np.float32))
    band.SetDescription('labels')
    band.SetNoDataValue(-1)
    
    band = ds.GetRasterBand(2)
    band.WriteArray(depth)
    band.SetDescription('depth')
    
    ds.FlushCache()
    ds = None
    
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create feature labels')
    parser.add_argument('--survey', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wrecks', type=Path, help='NOAA wrecks shapefile')
    parser.add_argument('--slope-threshold', type=float, default=15.0)
    parser.add_argument('--wreck-radius', type=int, default=50)
    parser.add_argument('--vr-bag-mode', default='resampled')
    
    args = parser.parse_args()
    
    add_feature_labels(
        args.survey,
        args.output,
        args.wrecks,
        args.slope_threshold,
        args.wreck_radius,
        args.vr_bag_mode,
    )


if __name__ == '__main__':
    main()
```

### Step 3.2: Generate Feature Labels

```cmd
:: For surveys with known features
python scripts/prepare_feature_labels.py ^
    --survey data/raw/features/rocky_survey.bag ^
    --output data/processed/labels/rocky_survey_features.tif ^
    --slope-threshold 15.0 ^
    --vr-bag-mode resampled

:: Include wrecks database
python scripts/prepare_feature_labels.py ^
    --survey data/raw/features/harbor_survey.bag ^
    --output data/processed/labels/harbor_survey_features.tif ^
    --wrecks data/external/wrecks/wrecks.shp ^
    --wreck-radius 50
```

---

## Phase 4: Training Infrastructure

**Duration:** 1 week  
**Goal:** Set up proper training with validation and metrics

### Step 4.1: Create Training Dataset

```python
#!/usr/bin/env python3
"""
scripts/create_training_dataset.py

Combine ground truth and feature labels into training tiles.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
from pathlib import Path
import numpy as np
from osgeo import gdal
import json
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import TileManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def create_tiles_from_labeled_survey(
    depth_path: Path,
    labels_path: Path,
    output_dir: Path,
    tile_size: int = 512,
    overlap: int = 64,
    min_valid_ratio: float = 0.3,
):
    """Create training tiles from labeled survey."""
    
    # Load depth
    ds = gdal.Open(str(depth_path))
    depth = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    
    # Load labels
    ds = gdal.Open(str(labels_path))
    labels = ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
    ds = None
    
    valid = labels >= 0
    
    # Create tiles
    tiles = []
    height, width = depth.shape
    
    for y in range(0, height - tile_size + 1, tile_size - overlap):
        for x in range(0, width - tile_size + 1, tile_size - overlap):
            tile_depth = depth[y:y+tile_size, x:x+tile_size]
            tile_labels = labels[y:y+tile_size, x:x+tile_size]
            tile_valid = valid[y:y+tile_size, x:x+tile_size]
            
            valid_ratio = np.mean(tile_valid)
            if valid_ratio >= min_valid_ratio:
                tiles.append({
                    'depth': tile_depth,
                    'labels': tile_labels,
                    'valid': tile_valid,
                    'position': (y, x),
                    'source': str(depth_path.name),
                })
    
    logger.info(f"Created {len(tiles)} tiles from {depth_path.name}")
    
    # Save tiles
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, tile in enumerate(tiles):
        tile_path = output_dir / f"{depth_path.stem}_tile_{i:04d}.npz"
        np.savez_compressed(
            tile_path,
            depth=tile['depth'],
            labels=tile['labels'],
            valid=tile['valid'],
            position=tile['position'],
            source=tile['source'],
        )
    
    return len(tiles)


def split_dataset(
    tiles_dir: Path,
    train_dir: Path,
    val_dir: Path,
    val_ratio: float = 0.2,
    random_seed: int = 42,
):
    """Split tiles into train/validation sets."""
    
    tile_files = list(Path(tiles_dir).glob('*.npz'))
    
    # Group by source survey (geographic split)
    sources = {}
    for f in tile_files:
        data = np.load(f)
        source = str(data['source'])
        if source not in sources:
            sources[source] = []
        sources[source].append(f)
    
    # Split by source (entire surveys go to train or val)
    source_names = list(sources.keys())
    train_sources, val_sources = train_test_split(
        source_names, test_size=val_ratio, random_state=random_seed
    )
    
    # Move files
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    
    for source in train_sources:
        for f in sources[source]:
            shutil.copy(f, train_dir / f.name)
    
    for source in val_sources:
        for f in sources[source]:
            shutil.copy(f, val_dir / f.name)
    
    logger.info(f"Train: {len(train_sources)} surveys, Val: {len(val_sources)} surveys")
    logger.info(f"Train tiles: {sum(len(sources[s]) for s in train_sources)}")
    logger.info(f"Val tiles: {sum(len(sources[s]) for s in val_sources)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled-dir', type=Path, required=True,
                        help='Directory with labeled surveys')
    parser.add_argument('--depth-dir', type=Path, required=True,
                        help='Directory with depth surveys')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    
    args = parser.parse_args()
    
    # Create tiles
    tiles_dir = args.output_dir / 'tiles'
    
    for labels_file in args.labeled_dir.glob('*.tif'):
        # Find matching depth file
        depth_name = labels_file.stem.replace('_ground_truth', '').replace('_features', '')
        depth_file = args.depth_dir / f"{depth_name}.bag"
        
        if not depth_file.exists():
            depth_file = args.depth_dir / f"{depth_name}.tif"
        
        if depth_file.exists():
            create_tiles_from_labeled_survey(
                depth_file, labels_file, tiles_dir, args.tile_size
            )
    
    # Split into train/val
    split_dataset(
        tiles_dir,
        args.output_dir / 'train',
        args.output_dir / 'validation',
        args.val_ratio,
    )


if __name__ == '__main__':
    main()
```

### Step 4.2: Create Dataset

```cmd
python scripts/create_training_dataset.py ^
    --labeled-dir data/processed/labels ^
    --depth-dir data/raw/noisy ^
    --output-dir data/processed ^
    --tile-size 512 ^
    --val-ratio 0.2
```

### Step 4.3: Train with Validation

```cmd
:: Train v2 model with real data
python scripts/train.py ^
    --train-dir data/processed/train ^
    --val-dir data/processed/validation ^
    --output-dir outputs/models/v2 ^
    --epochs 100 ^
    --batch-size 4 ^
    --learning-rate 0.001 ^
    --early-stopping-patience 10
```

---

## Phase 5: Iterative Refinement

**Duration:** Ongoing  
**Goal:** Continuous improvement from deployment feedback

### Step 5.1: Export Low-Confidence Regions for Review

```python
#!/usr/bin/env python3
"""
scripts/export_for_review.py

Export low-confidence regions for human review.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path
import numpy as np
from osgeo import gdal

def export_review_regions(
    predictions_path: Path,
    output_path: Path,
    confidence_low: float = 0.4,
    confidence_high: float = 0.7,
    min_region_size: int = 100,
):
    """Export regions where model is uncertain."""
    from scipy import ndimage
    
    ds = gdal.Open(str(predictions_path))
    
    # Load bands
    classification = None
    confidence = None
    depth = None
    
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        desc = band.GetDescription().lower()
        if 'class' in desc:
            classification = band.ReadAsArray()
        elif 'confid' in desc:
            confidence = band.ReadAsArray()
        elif 'depth' in desc:
            depth = band.ReadAsArray()
    
    gt = ds.GetGeoTransform()
    crs = ds.GetProjection()
    ds = None
    
    # Find uncertain regions
    uncertain = (confidence >= confidence_low) & (confidence <= confidence_high)
    
    # Label connected regions
    labeled, num_regions = ndimage.label(uncertain)
    
    # Filter by size
    review_mask = np.zeros_like(uncertain)
    regions = []
    
    for i in range(1, num_regions + 1):
        region_mask = labeled == i
        if np.sum(region_mask) >= min_region_size:
            review_mask |= region_mask
            
            # Get region bounds
            rows, cols = np.where(region_mask)
            regions.append({
                'id': len(regions) + 1,
                'size': int(np.sum(region_mask)),
                'mean_confidence': float(np.mean(confidence[region_mask])),
                'bounds': {
                    'min_row': int(rows.min()),
                    'max_row': int(rows.max()),
                    'min_col': int(cols.min()),
                    'max_col': int(cols.max()),
                },
            })
    
    print(f"Found {len(regions)} regions for review")
    
    # Save review mask
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        str(output_path),
        classification.shape[1], classification.shape[0], 3,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW']
    )
    
    out_ds.SetGeoTransform(gt)
    if crs:
        out_ds.SetProjection(crs)
    
    out_ds.GetRasterBand(1).WriteArray(review_mask.astype(np.float32))
    out_ds.GetRasterBand(1).SetDescription('review_mask')
    
    out_ds.GetRasterBand(2).WriteArray(confidence)
    out_ds.GetRasterBand(2).SetDescription('confidence')
    
    out_ds.GetRasterBand(3).WriteArray(classification)
    out_ds.GetRasterBand(3).SetDescription('classification')
    
    out_ds.FlushCache()
    out_ds = None
    
    # Save region list
    import json
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(regions, f, indent=2)
    
    print(f"Saved: {output_path}")
    print(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confidence-low', type=float, default=0.4)
    parser.add_argument('--confidence-high', type=float, default=0.7)
    parser.add_argument('--min-region-size', type=int, default=100)
    
    args = parser.parse_args()
    
    export_review_regions(
        args.predictions,
        args.output,
        args.confidence_low,
        args.confidence_high,
        args.min_region_size,
    )


if __name__ == '__main__':
    main()
```

### Step 5.2: Export and Review

```cmd
:: Export uncertain regions
python scripts/export_for_review.py ^
    --predictions outputs/predictions/survey_001_predictions.tif ^
    --output outputs/review/survey_001_review.tif ^
    --confidence-low 0.4 ^
    --confidence-high 0.7
```

### Step 5.3: Incorporate Corrections

After human review, save corrections as additional training data:

```cmd
:: Add corrections to training set
python scripts/add_corrections.py ^
    --original outputs/predictions/survey_001_predictions.tif ^
    --corrections outputs/review/survey_001_corrections.tif ^
    --output data/processed/labels/survey_001_corrected.tif
```

---

## Summary: Command Sequence

```cmd
:: ========================================
:: PHASE 1: Ground Truth
:: ========================================

:: Generate ground truth from clean/noisy pairs
python scripts/prepare_ground_truth.py --clean data/raw/clean/survey_001_clean.bag --noisy data/raw/noisy/survey_001_noisy.bag --output-dir data/processed/labels

:: Run inference on noisy survey
python scripts/inference.py --input data/raw/noisy/survey_001_noisy.bag --model outputs/final_model.pt --output outputs/predictions/survey_001_pred.tif --vr-bag-mode resampled --min-valid-ratio 0.01

:: Evaluate against ground truth
python scripts/evaluate_model.py --ground-truth data/processed/labels/survey_001_ground_truth.tif --predictions outputs/predictions/survey_001_pred.tif --output outputs/metrics/survey_001_eval.json

:: ========================================
:: PHASE 2: Noise Analysis
:: ========================================

:: Analyze noise patterns
python scripts/analyze_noise_patterns.py --input data/processed/labels/*.tif --output outputs/analysis/noise_patterns.json

:: ========================================
:: PHASE 3: Feature Labels
:: ========================================

:: Create feature labels from slope and wrecks
python scripts/prepare_feature_labels.py --survey data/raw/features/rocky_survey.bag --output data/processed/labels/rocky_features.tif --slope-threshold 15.0

:: ========================================
:: PHASE 4: Training
:: ========================================

:: Create training dataset
python scripts/create_training_dataset.py --labeled-dir data/processed/labels --depth-dir data/raw/noisy --output-dir data/processed --tile-size 512 --val-ratio 0.2

:: Train v2 model
python scripts/train.py --train-dir data/processed/train --val-dir data/processed/validation --output-dir outputs/models/v2 --epochs 100

:: ========================================
:: PHASE 5: Deployment & Refinement
:: ========================================

:: Run inference
python scripts/inference_vr_native.py --input new_survey.bag --model outputs/models/v2/best_model.pt --output cleaned_survey.bag

:: Export for review
python scripts/export_for_review.py --predictions outputs/predictions/new_survey.tif --output outputs/review/new_survey_review.tif
```

---

## Metrics to Track

| Metric | Target | Current |
|--------|--------|---------|
| Noise Precision | > 0.90 | Unknown |
| Noise Recall | > 0.85 | Unknown |
| Feature Precision | > 0.80 | Unknown |
| Feature Recall | > 0.70 | Unknown |
| Overall Accuracy | > 0.90 | 0.63 (synthetic) |
| Mean Confidence (correct) | > 0.85 | Unknown |
| Mean Confidence (incorrect) | < 0.50 | Unknown |

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Ground Truth | 3-5 labeled survey pairs, baseline metrics |
| 3 | Noise Analysis | Updated synthetic noise parameters |
| 4-5 | Feature Labels | Feature training data from slopes + wrecks |
| 6 | Infrastructure | Train/val split, metrics logging |
| 7-8 | Training v2 | Model trained on real data |
| 9+ | Refinement | Continuous improvement from feedback |

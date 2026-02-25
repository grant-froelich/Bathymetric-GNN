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
    
    classification = None
    confidence = None
    
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        desc = band.GetDescription().lower()
        if 'class' in desc:
            classification = band.ReadAsArray()
        elif 'confid' in desc:
            confidence = band.ReadAsArray()
    
    # If no description, assume band order
    if classification is None and ds.RasterCount >= 3:
        classification = ds.GetRasterBand(3).ReadAsArray()
    if confidence is None and ds.RasterCount >= 4:
        confidence = ds.GetRasterBand(4).ReadAsArray()
    
    ds = None
    return classification, confidence


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray = None):
    """Compute per-class and overall metrics."""
    
    # Valid mask
    valid = (y_true >= 0) & (y_pred >= 0) & np.isfinite(y_pred)
    y_true = y_true[valid].astype(np.int32)
    y_pred = y_pred[valid].astype(np.int32)
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
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'true_positives': int(true_pos),
            'false_positives': int(false_pos),
            'false_negatives': int(false_neg),
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
    if confidence is not None and len(confidence) > 0:
        correct = y_true == y_pred
        incorrect = ~correct
        
        metrics['confidence'] = {
            'mean': float(np.mean(confidence)),
            'std': float(np.std(confidence)),
            'mean_correct': float(np.mean(confidence[correct])) if np.any(correct) else 0,
            'mean_incorrect': float(np.mean(confidence[incorrect])) if np.any(incorrect) else 0,
        }
        
        # Accuracy at thresholds
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            mask = confidence >= thresh
            if np.sum(mask) > 0:
                metrics['confidence'][f'accuracy_at_{thresh}'] = float(np.mean(y_true[mask] == y_pred[mask]))
                metrics['confidence'][f'coverage_at_{thresh}'] = float(np.mean(mask))
    
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
        if class_name in metrics:
            m = metrics[class_name]
            print(f"{class_name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10,}")
    
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("-" * 60)
    print(f"{'':>12} " + " ".join(f"{name:>10}" for name in CLASS_NAMES))
    
    cm = np.array(metrics['confusion_matrix'])
    for i, class_name in enumerate(CLASS_NAMES):
        row = " ".join(f"{cm[i,j]:>10,}" for j in range(len(CLASS_NAMES)))
        print(f"{class_name:>12} {row}")
    
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
    
    if predictions is None:
        raise ValueError("Could not load predictions from file")
    
    if labels.shape != predictions.shape:
        raise ValueError(f"Shape mismatch: GT {labels.shape} vs Pred {predictions.shape}")
    
    metrics = compute_metrics(labels, predictions, confidence)
    print_metrics(metrics)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {args.output}")


if __name__ == '__main__':
    main()

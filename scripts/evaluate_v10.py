#!/usr/bin/env python3
"""
scripts/evaluate_v10.py

Evaluate a trained V10 regression model using the V10Metrics module.

Loads a checkpoint, runs inference on regression-mode ground truth files,
denormalizes predictions (model outputs are in local_std units), and computes
per-magnitude-bucket MAE, hazardous error rate, and recovery RMSE.

Usage:
    python scripts/evaluate_v10.py --checkpoint outputs/best_model.pt --ground-truth-dir ground-truth-val/
    python scripts/evaluate_v10.py --checkpoint outputs/best_model.pt --ground-truth-dir ground-truth-val/ --output eval_results.json
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import torch before numpy to avoid DLL conflicts on Windows
import torch

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data import GraphBuilder
from models import BathymetricGNN
from training import GroundTruthDataset
from training.metrics import compute_v10_metrics, V10Metrics


def load_model(checkpoint_path: Path, device: str):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    in_channels = checkpoint.get('in_channels', 8)
    edge_dim = checkpoint.get('edge_dim', 3)
    
    model = BathymetricGNN(
        in_channels=in_channels,
        hidden_channels=config.model.gnn_hidden_channels,
        num_gnn_layers=config.model.gnn_num_layers,
        gnn_type=config.model.gnn_type,
        heads=config.model.gnn_heads,
        num_classes=config.model.num_classes,
        predict_correction=config.model.predict_correction,
        dropout=config.model.gnn_dropout,
        edge_dim=edge_dim,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', -1)
    logger.info(f"Loaded model from epoch {epoch + 1}, best val loss {checkpoint.get('best_val_loss', 'N/A')}")
    
    return model, config


def evaluate(
    checkpoint_path: Path,
    ground_truth_dir: Path,
    device: str = 'cuda',
    tile_size: int = 256,
    overlap: int = 32,
    model_version: str = 'V10',
    iho_order: str = None,
    tvu_a: float = None,
    tvu_b: float = None,
) -> V10Metrics:
    """Evaluate a V10 model on regression-mode ground truth files."""
    
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    model, config = load_model(checkpoint_path, device)
    
    # Collect regression-mode ground truth files
    gt_files = sorted(ground_truth_dir.glob("*_regression.tif"))
    if not gt_files:
        logger.error(f"No regression-mode ground truth files found in {ground_truth_dir}")
        sys.exit(1)
    logger.info(f"Found {len(gt_files)} regression ground truth files")
    
    graph_builder = GraphBuilder(
        connectivity=config.graph.connectivity,
        edge_features=config.graph.edge_features,
    )
    
    dataset = GroundTruthDataset(
        ground_truth_paths=gt_files,
        graph_builder=graph_builder,
        tile_size=tile_size,
        overlap=overlap,
        min_valid_ratio=config.tile.min_valid_ratio,
    )
    
    logger.info(f"Dataset contains {len(dataset)} tiles")
    
    # Accumulate predictions and targets across all tiles (in meters)
    all_pred_meters = []
    all_target_meters = []
    all_noisy = []
    all_clean = []
    all_local_std = []
    
    logger.info("Running inference...")
    with torch.no_grad():
        for i in range(len(dataset)):
            graph = dataset[i]
            tile = dataset.tiles[i]
            
            if graph.num_nodes == 0:
                continue
            
            graph = graph.to(device)
            outputs = model(graph)
            
            if 'correction' not in outputs:
                logger.error("Model does not output corrections; not a correction model")
                sys.exit(1)
            
            # Model prediction is normalized (local_std units). Denormalize to meters.
            pred_norm = outputs['correction'].detach().cpu().numpy()
            local_std = graph.local_std.detach().cpu().numpy()
            pred_meters = pred_norm * np.maximum(local_std, 0.01)
            
            # Targets and reference depths in meters, from the tile arrays
            rows = graph.valid_rows.detach().cpu().numpy()
            cols = graph.valid_cols.detach().cpu().numpy()
            target_meters = tile['difference'][rows, cols]
            noisy = tile['noisy_depth'][rows, cols]
            clean = tile['clean_depth'][rows, cols]
            
            all_pred_meters.append(pred_meters)
            all_target_meters.append(target_meters)
            all_noisy.append(noisy)
            all_clean.append(clean)
            all_local_std.append(np.maximum(local_std, 0.01))
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(dataset)} tiles")
    
    if not all_pred_meters:
        logger.error("No valid predictions produced")
        sys.exit(1)
    
    pred_meters = np.concatenate(all_pred_meters)
    target_meters = np.concatenate(all_target_meters)
    noisy = np.concatenate(all_noisy)
    clean = np.concatenate(all_clean)
    local_std = np.concatenate(all_local_std)
    valid_mask = np.ones(len(pred_meters), dtype=bool)
    
    survey_name = ground_truth_dir.name
    
    metrics = compute_v10_metrics(
        predicted_correction=pred_meters,
        target_correction=target_meters,
        valid_mask=valid_mask,
        local_std=local_std,
        noisy_depth=noisy,
        clean_depth=clean,
        survey_name=survey_name,
        model_version=model_version,
        iho_order=iho_order,
        tvu_a=tvu_a,
        tvu_b=tvu_b,
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate a V10 regression model')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--ground-truth-dir', type=Path, required=True,
                        help='Directory with regression-mode ground truth files')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--model-version', default='V10', help='Label for the model version')
    parser.add_argument('--iho-order', default=None,
                        help='Order label for the TVU breach metric. IHO S-44: '
                             'exclusive, special, 1a, 1b, 2. NOAA HSSD OCS Quality '
                             'Metric: exceptional, critical, general1, general2, '
                             'general3, general4. Omit to skip the TVU metric.')
    parser.add_argument('--tvu-a', type=float, default=None,
                        help='Explicit TVU constant term a in meters (overrides --iho-order)')
    parser.add_argument('--tvu-b', type=float, default=None,
                        help='Explicit TVU depth-scaled term b (overrides --iho-order)')
    parser.add_argument('--output', type=Path, default=None, help='Optional path to save metrics JSON')
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        ground_truth_dir=args.ground_truth_dir,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        model_version=args.model_version,
        iho_order=args.iho_order,
        tvu_a=args.tvu_a,
        tvu_b=args.tvu_b,
    )
    
    print()
    print(metrics.summary())
    print()
    
    if args.output:
        metrics.save_json(args.output)
        logger.info(f"Saved metrics to {args.output}")


if __name__ == '__main__':
    main()

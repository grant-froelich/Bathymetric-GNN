#!/usr/bin/env python3
"""
Training script for Bathymetric GNN.

Usage:
    python scripts/train.py --ground-truth-dir ground-truth-train/ --output-dir ./outputs
"""

import os
# Fix OpenMP conflict on Windows - must be before any other imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import torch BEFORE numpy to avoid DLL conflicts on Windows
import torch

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import Config
from data import GraphBuilder
from models import BathymetricGNN
from training import GroundTruthDataset, Trainer


def setup_logging(log_level: str = "INFO"):
    """Configure console logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )


def attach_log_file(output_dir: Path) -> Path:
    """
    Add a timestamped file handler to the root logger so all subsequent
    log lines are written to disk in addition to the console. Survives
    crashes, reboots, and lost terminal windows; the only thing missed
    is tqdm progress bars (those use direct stderr writes, not logging).
    """
    from datetime import datetime
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    
    return log_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Bathymetric GNN for denoising"
    )
    
    # Data arguments
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="Directory containing ground truth GeoTIFF files (from prepare_ground_truth.py)",
    )
    parser.add_argument(
        "--val-surveys",
        type=Path,
        default=None,
        help="Directory containing clean survey files for validation (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory for outputs (checkpoints, logs)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML file (optional)",
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    
    # Model arguments
    parser.add_argument("--gnn-type", choices=["GCN", "GAT", "GraphSAGE", "GIN"], default="GAT")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable bf16 mixed-precision training (autocast on the forward "
             "pass and loss). Speeds up the GAT matmuls on CUDA; no effect on "
             "CPU. Off by default.",
    )
    
    # Misc
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load or create config
    if args.config and args.config.exists():
        config = Config.load(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = Config()
    
    # Override config with command line args
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.tile.tile_size = args.tile_size
    config.tile.overlap = args.overlap
    config.model.gnn_type = args.gnn_type
    config.model.gnn_hidden_channels = args.hidden_channels
    config.model.gnn_num_layers = args.num_layers
    config.model.gnn_heads = args.heads
    config.device = args.device
    config.num_workers = args.num_workers
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Attach a log file in the output directory so progress survives
    # terminal crashes, reboots, and power loss.
    log_path = attach_log_file(args.output_dir)
    logger.info(f"Logging to file: {log_path}")
    
    # Save config
    config.save(args.output_dir / "config.yaml")
    
    # Initialize graph builder (used for both modes)
    graph_builder = GraphBuilder(
        connectivity=config.graph.connectivity,
        edge_features=config.graph.edge_features,
    )
    
    # Ground truth datasets (real noise from clean/noisy pairs)
    logger.info("Using GROUND TRUTH mode (real noise from clean/noisy pairs)")
    
    # Pick up both classification (_ground_truth.tif) and regression
    # (_regression.tif) files. GroundTruthDataset auto-detects the mode.
    gt_files = sorted(
        list(args.ground_truth_dir.glob("*_ground_truth.tif")) +
        list(args.ground_truth_dir.glob("*_regression.tif"))
    )
    if not gt_files:
        logger.error(f"No ground truth files found in {args.ground_truth_dir}")
        sys.exit(1)
    logger.info(f"Found {len(gt_files)} ground truth files")
    
    # Split into train/val if no separate val set provided
    if args.val_surveys:
        val_gt_files = sorted(
            list(args.val_surveys.glob("*_ground_truth.tif")) +
            list(args.val_surveys.glob("*_regression.tif"))
        )
        train_gt_files = gt_files
    elif len(gt_files) > 1:
        # Use last file for validation
        train_gt_files = gt_files[:-1]
        val_gt_files = gt_files[-1:]
        logger.info(f"Using {len(train_gt_files)} files for training, {len(val_gt_files)} for validation")
    else:
        train_gt_files = gt_files
        val_gt_files = []
    
    # Create datasets
    logger.info("Creating training dataset from ground truth...")
    train_dataset = GroundTruthDataset(
        ground_truth_paths=train_gt_files,
        graph_builder=graph_builder,
        tile_size=config.tile.tile_size,
        overlap=config.tile.overlap,
        min_valid_ratio=config.tile.min_valid_ratio,
    )
    
    val_dataset = None
    if val_gt_files:
        logger.info("Creating validation dataset from ground truth...")
        val_dataset = GroundTruthDataset(
            ground_truth_paths=val_gt_files,
            graph_builder=graph_builder,
            tile_size=config.tile.tile_size,
            overlap=config.tile.overlap,
            min_valid_ratio=config.tile.min_valid_ratio,
        )
    
    # Determine input dimensions from first sample
    sample = train_dataset[0]
    in_channels = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else None
    
    logger.info(f"Input channels: {in_channels}, Edge features: {edge_dim}")
    
    # Create model
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        use_amp=args.amp,
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train()
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

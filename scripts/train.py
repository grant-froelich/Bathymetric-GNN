#!/usr/bin/env python3
"""
Training script for Bathymetric GNN.

Usage:
    python scripts/train.py --clean-surveys /path/to/clean/data --output-dir ./outputs
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
from data import (
    TileManager,
    GraphBuilder,
    SyntheticNoiseGenerator,
)
from models import BathymetricGNN
from training import BathymetricGraphDataset, Trainer


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Bathymetric GNN for denoising"
    )
    
    # Data arguments
    parser.add_argument(
        "--clean-surveys",
        type=Path,
        required=True,
        help="Directory containing clean survey files for training",
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
    
    # Data loading arguments
    parser.add_argument(
        "--vr-bag-mode",
        choices=["refinements", "resampled", "base"],
        default="resampled",
        help="How to handle Variable Resolution BAGs (default: resampled)",
    )
    
    # Model arguments
    parser.add_argument("--gnn-type", choices=["GCN", "GAT", "GraphSAGE", "GIN"], default="GAT")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Misc
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def find_survey_files(directory: Path) -> list:
    """Find all supported survey files in directory."""
    extensions = {'.bag', '.tif', '.tiff'}
    files = []
    
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"**/*{ext}"))
    
    return sorted(set(files))


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
    
    # Save config
    config.save(args.output_dir / "config.yaml")
    
    # Find training files
    train_files = find_survey_files(args.clean_surveys)
    if not train_files:
        logger.error(f"No survey files found in {args.clean_surveys}")
        sys.exit(1)
    logger.info(f"Found {len(train_files)} training surveys")
    
    # Find validation files
    val_files = []
    if args.val_surveys:
        val_files = find_survey_files(args.val_surveys)
        logger.info(f"Found {len(val_files)} validation surveys")
    
    # Initialize components
    tile_manager = TileManager(
        tile_size=config.tile.tile_size,
        overlap=config.tile.overlap,
        min_valid_ratio=config.tile.min_valid_ratio,
    )
    
    graph_builder = GraphBuilder(
        connectivity=config.graph.connectivity,
        edge_features=config.graph.edge_features,
    )
    
    noise_generator = SyntheticNoiseGenerator(
        enable_gaussian=config.noise.enable_gaussian,
        enable_spikes=config.noise.enable_spikes,
        enable_blobs=config.noise.enable_blobs,
        enable_systematic=config.noise.enable_systematic,
        seed=args.seed,
    )
    
    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = BathymetricGraphDataset(
        survey_paths=train_files,
        tile_manager=tile_manager,
        graph_builder=graph_builder,
        noise_generator=noise_generator,
        augment=True,
        vr_bag_mode=args.vr_bag_mode,
    )
    
    val_dataset = None
    if val_files:
        logger.info("Creating validation dataset...")
        val_dataset = BathymetricGraphDataset(
            survey_paths=val_files,
            tile_manager=tile_manager,
            graph_builder=graph_builder,
            noise_generator=noise_generator,
            augment=False,  # No augmentation for validation
            vr_bag_mode=args.vr_bag_mode,
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
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train()
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

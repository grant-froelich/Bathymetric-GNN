#!/usr/bin/env python3
"""
Inference script for Bathymetric GNN.

Usage:
    python scripts/inference.py --input survey.bag --model model.pt --output cleaned.bag
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

from config import Config
from models import BathymetricPipeline


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
        description="Run inference with trained Bathymetric GNN"
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input bathymetric file (BAG, GeoTIFF)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (optional, will use model's config if not provided)",
    )
    
    # Processing options
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size for processing (default: from config)",
    )
    parser.add_argument(
        "--auto-correct-threshold",
        type=float,
        default=None,
        help="Confidence threshold for auto-correction (default: 0.85)",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="Confidence threshold for flagging review (default: 0.6)",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=None,
        help="Minimum ratio of valid data to process a tile (default: 0.1). Lower to process sparser data.",
    )
    
    # Output options
    parser.add_argument(
        "--export-extras",
        action="store_true",
        default=True,
        help="Export confidence and classification layers",
    )
    parser.add_argument(
        "--no-export-extras",
        action="store_false",
        dest="export_extras",
        help="Only export cleaned depth",
    )
    
    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    
    # VR BAG handling
    parser.add_argument(
        "--vr-bag-mode",
        type=str,
        choices=["refinements", "resampled", "base"],
        default="resampled",
        help="How to handle Variable Resolution BAGs (default: resampled)",
    )
    
    # Misc
    parser.add_argument("--log-level", type=str, default="INFO")
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.model.exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    if args.config and args.config.exists():
        config = Config.load(args.config)
    else:
        # Try to load config from model directory
        model_dir = args.model.parent
        config_path = model_dir / "config.yaml"
        if config_path.exists():
            config = Config.load(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            config = Config()
            logger.info("Using default config")
    
    # Override config with command line args
    config.device = args.device
    
    if args.tile_size:
        config.tile.tile_size = args.tile_size
    
    if args.min_valid_ratio is not None:
        config.tile.min_valid_ratio = args.min_valid_ratio
    
    if args.auto_correct_threshold:
        config.inference.auto_correct_threshold = args.auto_correct_threshold
    
    if args.review_threshold:
        config.inference.review_threshold = args.review_threshold
    
    # Create pipeline
    pipeline = BathymetricPipeline(config, vr_bag_mode=args.vr_bag_mode)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    pipeline.load_model(args.model)
    
    # Process
    logger.info(f"Processing {args.input}")
    results = pipeline.process(
        input_path=args.input,
        output_path=args.output,
        export_extras=args.export_extras,
    )
    
    logger.info(f"Output saved to {args.output}")
    
    # Summary statistics
    if 'classification' in results:
        import numpy as np
        classification = results['classification']
        confidence = results['confidence']
        
        valid = ~np.isnan(classification)
        
        logger.info("Results summary:")
        logger.info(f"  - Noise detected: {np.sum(classification[valid] == 2)}")
        logger.info(f"  - Features preserved: {np.sum(classification[valid] == 1)}")
        logger.info(f"  - Mean confidence: {np.nanmean(confidence):.3f}")
        logger.info(f"  - Low confidence cells: {np.sum(confidence[valid] < config.inference.review_threshold)}")


if __name__ == "__main__":
    main()

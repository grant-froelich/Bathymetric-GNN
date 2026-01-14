#!/usr/bin/env python3
"""
Native BAG inference script.

Processes BAG files without resampling:
- VR BAGs: Iterates through refinement grids, preserving multi-resolution structure
- SR BAGs: Processes the full elevation grid directly

Both modes output a corrected BAG and optional sidecar GeoTIFF.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models.gnn import BathymetricGNN
from data import GraphBuilder
from data.vr_bag import (
    detect_bag_type,
    VRBagHandler, SRBagHandler,
    SidecarBuilder,
)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native BAG inference - preserves BAG structure (VR or SR)"
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input BAG file (VR or SR)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output BAG file (copy with corrections applied)",
    )
    
    # Optional config
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config YAML file",
    )
    
    # Processing options
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.01,
        help="Minimum ratio of valid data to process a grid (default: 0.01)",
    )
    parser.add_argument(
        "--auto-correct-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for auto-correction (default: 0.85)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    
    # Output options
    parser.add_argument(
        "--export-sidecar",
        action="store_true",
        default=True,
        help="Export sidecar GeoTIFF with classification/confidence",
    )
    parser.add_argument(
        "--no-export-sidecar",
        action="store_false",
        dest="export_sidecar",
    )
    
    parser.add_argument("--log-level", type=str, default="INFO")
    
    return parser.parse_args()


class NativeProcessor:
    """Processes BAG grids through GNN."""
    
    CLASS_NOISE = 2
    
    def __init__(
        self,
        model: BathymetricGNN,
        graph_builder: GraphBuilder,
        device: torch.device,
        auto_correct_threshold: float = 0.85,
    ):
        self.model = model
        self.graph_builder = graph_builder
        self.device = device
        self.auto_correct_threshold = auto_correct_threshold
        
        self.model.eval()
    
    def process_grid(
        self,
        depth: np.ndarray,
        uncertainty: np.ndarray,
        resolution: tuple,
    ) -> tuple:
        """
        Process a single grid through the GNN.
        
        Args:
            depth: 2D depth array
            uncertainty: 2D uncertainty array
            resolution: (x_res, y_res) tuple
            
        Returns:
            (classification, confidence, correction) arrays
        """
        # Create valid mask
        nodata = 1.0e6
        valid_mask = (depth != nodata) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            # Return empty results
            return (
                np.zeros_like(depth, dtype=np.float32),
                np.zeros_like(depth, dtype=np.float32),
                np.zeros_like(depth, dtype=np.float32),
            )
        
        # Build graph
        graph = self.graph_builder.build_graph(
            depth=depth,
            valid_mask=valid_mask,
            uncertainty=uncertainty,
            resolution=resolution,
        )
        
        if graph.num_nodes == 0:
            return (
                np.zeros_like(depth, dtype=np.float32),
                np.zeros_like(depth, dtype=np.float32),
                np.zeros_like(depth, dtype=np.float32),
            )
        
        # Run inference
        graph = graph.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.predict(
                graph,
                auto_correct_threshold=self.auto_correct_threshold,
            )
        
        # Convert back to grids
        classification = self.graph_builder.graph_to_grid(
            graph.cpu(),
            outputs['predicted_class'].cpu().float(),
            fill_value=0.0,
        )
        
        confidence = self.graph_builder.graph_to_grid(
            graph.cpu(),
            outputs['confidence'].cpu(),
            fill_value=0.0,
        )
        
        correction = np.zeros_like(depth, dtype=np.float32)
        if 'correction' in outputs:
            correction = self.graph_builder.graph_to_grid(
                graph.cpu(),
                outputs['correction'].cpu(),
                fill_value=0.0,
            )
        
        return classification, confidence, correction


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
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Detect BAG type
    bag_type = detect_bag_type(args.input)
    logger.info(f"Detected BAG type: {bag_type}")
    
    # Load config
    if args.config and args.config.exists():
        config = Config.load(args.config)
    else:
        model_dir = args.model.parent
        config_path = model_dir / "config.yaml"
        if config_path.exists():
            config = Config.load(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            config = Config()
            logger.info("Using default config")
    
    # Setup device with Blackwell compatibility
    device = torch.device("cpu")
    if args.device == "cuda" and torch.cuda.is_available():
        try:
            test = torch.zeros(1).cuda()
            _ = test + 1
            device = torch.device("cuda")
        except RuntimeError as e:
            if "no kernel image" in str(e):
                logger.warning(
                    "GPU not supported by PyTorch (likely Blackwell/RTX 50). "
                    "Falling back to CPU."
                )
            else:
                raise
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    model_config = checkpoint['config'].model
    in_channels = checkpoint.get('in_channels', 7)
    edge_dim = checkpoint.get('edge_dim', 3)
    
    model = BathymetricGNN(
        in_channels=in_channels,
        hidden_channels=model_config.gnn_hidden_channels,
        num_gnn_layers=model_config.gnn_num_layers,
        gnn_type=model_config.gnn_type,
        heads=model_config.gnn_heads,
        num_classes=model_config.num_classes,
        predict_correction=model_config.predict_correction,
        dropout=0.0,
        edge_dim=edge_dim,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create processor
    graph_builder = GraphBuilder(
        connectivity=config.graph.connectivity,
        edge_features=config.graph.edge_features,
    )
    
    processor = NativeProcessor(
        model=model,
        graph_builder=graph_builder,
        device=device,
        auto_correct_threshold=args.auto_correct_threshold,
    )
    
    # Open BAG with appropriate handler
    logger.info(f"Opening BAG: {args.input}")
    
    if bag_type == 'VR':
        handler = VRBagHandler(args.input)
    else:
        handler = SRBagHandler(args.input)
    
    info = handler.get_refinement_info()
    logger.info(f"BAG structure:")
    logger.info(f"  - Grid shape: {info['base_shape']}")
    if bag_type == 'VR':
        logger.info(f"  - Refined cells: {info['num_refined_cells']:,}")
        logger.info(f"  - Total nodes: {info['total_refinement_nodes']:,}")
        logger.info(f"  - Resolutions: {info['unique_resolutions']} meters")
    else:
        logger.info(f"  - Total cells: {info['total_refinement_nodes']:,}")
        logger.info(f"  - Resolution: {info['unique_resolutions'][0]} meters")
    
    # Copy BAG and open for writing
    logger.info(f"Creating output: {args.output}")
    writer = handler.copy_and_open_for_writing(args.output)
    
    # Initialize sidecar builder if requested
    sidecar = None
    if args.export_sidecar:
        sidecar = SidecarBuilder(handler)
    
    # Process grids
    stats = {
        'grids_processed': 0,
        'cells_processed': 0,
        'cells_classified_noise': 0,
        'cells_corrected': 0,
        'total_confidence': 0.0,
    }
    
    try:
        for i, grid in enumerate(handler.iterate_refinements(args.min_valid_ratio)):
            # Process through GNN
            classification, confidence, correction = processor.process_grid(
                grid.depth,
                grid.uncertainty,
                grid.resolution,
            )
            
            # Add to sidecar if building one
            if sidecar is not None:
                sidecar.add_refinement_results(grid, classification, confidence, correction)
            
            # Apply corrections
            corrected_depth = grid.depth.copy()
            corrected_uncertainty = grid.uncertainty.copy()
            
            # Only correct high-confidence noise
            noise_mask = (classification == processor.CLASS_NOISE) & grid.valid_mask
            high_conf = confidence >= args.auto_correct_threshold
            apply_mask = noise_mask & high_conf
            
            if np.any(apply_mask):
                corrected_depth[apply_mask] += correction[apply_mask]
                stats['cells_corrected'] += int(np.sum(apply_mask))
                
                # Scale uncertainty by confidence
                scale = 2.0 - confidence[apply_mask]
                corrected_uncertainty[apply_mask] *= scale
            
            # Write back
            writer.update_refinement_batch(
                grid,
                corrected_depth,
                corrected_uncertainty,
            )
            
            # Update stats
            stats['grids_processed'] += 1
            stats['cells_processed'] += grid.num_valid
            stats['cells_classified_noise'] += int(np.sum(noise_mask))
            stats['total_confidence'] += float(np.sum(confidence[grid.valid_mask]))
            
            # Progress logging (more frequent for VR)
            if bag_type == 'VR' and (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{info['num_refined_cells']} refinement grids")
    
    finally:
        writer.close()
    
    # Calculate final stats
    mean_confidence = (
        stats['total_confidence'] / stats['cells_processed']
        if stats['cells_processed'] > 0 else 0
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"BAG type: {bag_type}")
    logger.info(f"Grids processed: {stats['grids_processed']:,}")
    logger.info(f"Total cells processed: {stats['cells_processed']:,}")
    logger.info(f"Cells classified as noise: {stats['cells_classified_noise']:,} "
                f"({100*stats['cells_classified_noise']/max(1,stats['cells_processed']):.1f}%)")
    logger.info(f"Cells auto-corrected: {stats['cells_corrected']:,}")
    logger.info(f"Mean confidence: {mean_confidence:.3f}")
    logger.info("=" * 60)
    
    logger.info(f"Output saved: {args.output}")
    
    # Export sidecar GeoTIFF if requested
    if sidecar is not None:
        sidecar_path = args.output.with_name(args.output.stem + '_gnn_outputs.tif')
        sidecar.save(sidecar_path)


if __name__ == "__main__":
    main()

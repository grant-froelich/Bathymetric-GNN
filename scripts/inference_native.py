#!/usr/bin/env python3
"""
Native BAG inference script.

Processes BAG files without resampling:
- VR BAGs: Iterates through refinement grids, preserving multi-resolution structure
- SR BAGs: Processes the full elevation grid directly

Auto-detects BAG type and uses the appropriate handler.
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

try:
    from torch_geometric.data import Batch as PyGBatch
    PYG_BATCH_AVAILABLE = True
except ImportError:
    PYG_BATCH_AVAILABLE = False

from config import Config
from config.constants import CORRECTION_NORM_FLOOR
from models.gnn import BathymetricGNN
from data import GraphBuilder, VRBagHandler, VRBagWriter, SRBagHandler, SRBagWriter
from data.vr_bag import SidecarBuilder, detect_bag_type


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native BAG inference - auto-detects VR/SR and preserves structure"
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
        help="Minimum ratio of valid data to process a refinement grid (default: 0.01)",
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
        help="Export sidecar GeoTIFF with classification/confidence (resampled)",
    )
    parser.add_argument(
        "--no-export-sidecar",
        action="store_false",
        dest="export_sidecar",
    )
    
    parser.add_argument("--log-level", type=str, default="INFO")
    
    return parser.parse_args()


class NativeVRProcessor:
    """Processes VR BAG refinement grids through GNN.
    
    Supports both single-grid and batched processing. Batched mode
    accumulates multiple small grids into a single PyTorch Geometric
    Batch for a single forward pass, reducing per-grid overhead for
    VR BAGs with thousands of tiny refinement grids.
    """
    
    CLASS_NOISE = 2
    # Maximum number of nodes to accumulate before flushing a batch
    BATCH_NODE_BUDGET = 50000
    
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
        
        # Check model's expected input channels
        # The model's feature_extractor.mlp[0] is the first Linear layer
        try:
            self.expected_in_channels = model.feature_extractor.mlp[0].in_features
            logger.info(f"Model expects {self.expected_in_channels} input features")
        except (AttributeError, IndexError):
            logger.warning("Could not detect model input channels; will use all available features")
            self.expected_in_channels = None
        
        # Batching support
        self._batch_graphs = []
        self._batch_metadata = []  # (depth_shape, has_local_std)
        self._batch_node_count = 0
    
    def _build_graph_for_grid(self, depth, uncertainty, resolution, nodata):
        """Build a graph from a single grid, handling uncertainty channel selection."""
        valid_mask = (depth != nodata) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return None, valid_mask
        
        use_uncertainty = uncertainty
        if self.expected_in_channels == 7:
            use_uncertainty = None
        
        graph = self.graph_builder.build_graph(
            depth=depth,
            valid_mask=valid_mask,
            uncertainty=use_uncertainty,
            resolution=resolution,
        )
        
        if graph.num_nodes == 0:
            return None, valid_mask
        
        return graph, valid_mask
    
    def _extract_results_from_outputs(self, graph, outputs, depth_shape):
        """Convert model outputs back to grids for a single graph."""
        classification = self.graph_builder.graph_to_grid(
            graph, outputs['predicted_class'].float(), fill_value=0.0,
        )
        confidence = self.graph_builder.graph_to_grid(
            graph, outputs['confidence'], fill_value=0.0,
        )
        
        correction = np.zeros(depth_shape, dtype=np.float32)
        if 'correction' in outputs:
            norm_correction = self.graph_builder.graph_to_grid(
                graph, outputs['correction'], fill_value=0.0,
            )
            local_std_grid = self.graph_builder.graph_to_grid(
                graph,
                graph.local_std if hasattr(graph, 'local_std') else
                    torch.zeros(graph.num_nodes),
                fill_value=0.0,
            )
            local_std_grid = np.maximum(local_std_grid, CORRECTION_NORM_FLOOR)
            correction = norm_correction * local_std_grid
        
        return classification, confidence, correction
    
    def process_grid(
        self,
        depth: np.ndarray,
        uncertainty: np.ndarray,
        resolution: tuple,
        nodata: float = 1.0e6,
    ) -> tuple:
        """
        Process a single refinement grid (unbatched).
        
        Args:
            depth: 2D depth array
            uncertainty: 2D uncertainty array
            resolution: (x_res, y_res) tuple
            nodata: NoData sentinel value
            
        Returns:
            (classification, confidence, correction) arrays
        """
        empty = (
            np.zeros_like(depth, dtype=np.float32),
            np.zeros_like(depth, dtype=np.float32),
            np.zeros_like(depth, dtype=np.float32),
        )
        
        graph, valid_mask = self._build_graph_for_grid(depth, uncertainty, resolution, nodata)
        if graph is None:
            return empty
        
        # Run inference on single graph
        graph = graph.to(self.device)
        with torch.no_grad():
            outputs = self.model.predict(
                graph,
                auto_correct_threshold=self.auto_correct_threshold,
            )
        
        # Move back to CPU for grid conversion
        graph_cpu = graph.cpu()
        outputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
        
        return self._extract_results_from_outputs(graph_cpu, outputs_cpu, depth.shape)
    
    def add_to_batch(self, depth, uncertainty, resolution, nodata=1.0e6):
        """Accumulate a grid into the pending batch.
        
        Returns None if the grid was added to the batch (results pending).
        Returns results tuple immediately for grids with no valid data.
        Call flush_batch() to process all accumulated grids.
        """
        empty = (
            np.zeros_like(depth, dtype=np.float32),
            np.zeros_like(depth, dtype=np.float32),
            np.zeros_like(depth, dtype=np.float32),
        )
        
        graph, valid_mask = self._build_graph_for_grid(depth, uncertainty, resolution, nodata)
        if graph is None:
            return empty  # Immediately return empty results
        
        self._batch_graphs.append(graph)
        self._batch_metadata.append(depth.shape)
        self._batch_node_count += graph.num_nodes
        return None  # Results pending
    
    @property
    def batch_ready(self):
        """True if accumulated batch has enough nodes to justify a forward pass."""
        return self._batch_node_count >= self.BATCH_NODE_BUDGET
    
    @property
    def batch_pending(self):
        """True if there are any graphs waiting in the batch."""
        return len(self._batch_graphs) > 0
    
    def flush_batch(self):
        """Process all accumulated graphs in a single batched forward pass.
        
        Returns:
            List of (classification, confidence, correction) tuples,
            one per grid in the order they were added.
        """
        if not self._batch_graphs:
            return []
        
        graphs = self._batch_graphs
        metadata = self._batch_metadata
        self._batch_graphs = []
        self._batch_metadata = []
        self._batch_node_count = 0
        
        # If only one graph or PyG Batch unavailable, process individually
        if len(graphs) == 1 or not PYG_BATCH_AVAILABLE:
            results = []
            for graph, shape in zip(graphs, metadata):
                graph = graph.to(self.device)
                with torch.no_grad():
                    outputs = self.model.predict(
                        graph, auto_correct_threshold=self.auto_correct_threshold,
                    )
                graph_cpu = graph.cpu()
                outputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                results.append(self._extract_results_from_outputs(graph_cpu, outputs_cpu, shape))
            return results
        
        # Batch all graphs into a single forward pass
        batch = PyGBatch.from_data_list(graphs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.predict(
                batch,
                auto_correct_threshold=self.auto_correct_threshold,
            )
        
        # Move everything to CPU for unbatching
        batch_cpu = batch.cpu()
        outputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
        batch_indices = batch_cpu.batch.numpy()
        
        # Unbatch: split per-node tensors back into per-graph results
        results = []
        for graph_idx, (orig_graph, shape) in enumerate(zip(graphs, metadata)):
            node_mask = batch_indices == graph_idx
            
            # Build a per-graph outputs dict by slicing the batched tensors
            graph_outputs = {}
            for key, val in outputs_cpu.items():
                if isinstance(val, torch.Tensor) and val.shape[0] == len(batch_indices):
                    graph_outputs[key] = val[node_mask]
                else:
                    graph_outputs[key] = val
            
            # Reconstruct a minimal graph with correct metadata for graph_to_grid
            # We use the original (CPU) graph which has grid_shape, valid_rows, valid_cols
            results.append(self._extract_results_from_outputs(orig_graph, graph_outputs, shape))
        
        return results


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
    
    # Validate thresholds
    if not (0.0 <= args.auto_correct_threshold <= 1.0):
        logger.error(f"--auto-correct-threshold must be between 0.0 and 1.0, got {args.auto_correct_threshold}")
        sys.exit(1)
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
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
    in_channels = checkpoint.get('in_channels', 7)  # Default node features
    edge_dim = checkpoint.get('edge_dim', 3)  # Default edge features
    
    model = BathymetricGNN(
        in_channels=in_channels,
        hidden_channels=model_config.gnn_hidden_channels,
        num_gnn_layers=model_config.gnn_num_layers,
        gnn_type=model_config.gnn_type,
        heads=model_config.gnn_heads,
        num_classes=model_config.num_classes,
        predict_correction=model_config.predict_correction,
        dropout=0.0,  # No dropout during inference
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
    
    processor = NativeVRProcessor(
        model=model,
        graph_builder=graph_builder,
        device=device,
        auto_correct_threshold=args.auto_correct_threshold,
    )
    
    # Detect BAG type and open with appropriate handler
    bag_type = detect_bag_type(args.input)
    logger.info(f"Detected BAG type: {bag_type}")
    
    if bag_type == 'VR':
        handler = VRBagHandler(args.input)
    else:
        handler = SRBagHandler(args.input)
    
    info = handler.get_refinement_info()
    logger.info(f"BAG structure ({bag_type}):")
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
    
    # Process refinement grids
    stats = {
        'grids_processed': 0,
        'cells_processed': 0,
        'cells_classified_noise': 0,
        'cells_corrected': 0,
        'total_confidence': 0.0,
    }
    
    # Get nodata value from handler
    handler_nodata = getattr(handler, 'NODATA', 1.0e6)
    total_grids = info.get('num_refined_cells', info.get('total_refinement_nodes', 0))
    
    try:
        # Accumulate grids for batched processing
        pending_grids = []  # Stores (grid, immediate_result_or_None) for batch flush
        
        def apply_results(grid, classification, confidence, correction):
            """Apply inference results for a single grid."""
            if sidecar is not None:
                sidecar.add_refinement_results(grid, classification, confidence, correction)
            
            corrected_depth = grid.depth.copy()
            corrected_uncertainty = grid.uncertainty.copy()
            
            noise_mask = (classification == processor.CLASS_NOISE) & grid.valid_mask
            high_conf = confidence >= args.auto_correct_threshold
            apply_mask = noise_mask & high_conf
            
            if np.any(apply_mask):
                corrected_depth[apply_mask] -= correction[apply_mask]
                stats['cells_corrected'] += int(np.sum(apply_mask))
                scale = 2.0 - confidence[apply_mask]
                corrected_uncertainty[apply_mask] *= scale
            
            writer.update_refinement_batch(grid, corrected_depth, corrected_uncertainty)
            
            stats['grids_processed'] += 1
            stats['cells_processed'] += grid.num_valid
            stats['cells_classified_noise'] += int(np.sum(noise_mask))
            stats['total_confidence'] += float(np.sum(confidence[grid.valid_mask]))
        
        def flush_pending():
            """Flush accumulated batch and apply all results."""
            if not pending_grids:
                return
            batch_results = processor.flush_batch()
            batch_idx = 0
            for grid, immediate in pending_grids:
                if immediate is not None:
                    # Grid had no valid data; results were returned immediately
                    apply_results(grid, *immediate)
                else:
                    apply_results(grid, *batch_results[batch_idx])
                    batch_idx += 1
            pending_grids.clear()
        
        for i, grid in enumerate(handler.iterate_refinements(args.min_valid_ratio)):
            immediate = processor.add_to_batch(
                grid.depth, grid.uncertainty, grid.resolution, nodata=handler_nodata,
            )
            pending_grids.append((grid, immediate))
            
            # Flush when batch is large enough
            if processor.batch_ready:
                flush_pending()
            
            if (i + 1) % 100 == 0 or (i + 1) == total_grids:
                pct = 100 * (i + 1) / max(1, total_grids)
                logger.info(
                    f"Processed {i + 1:,}/{total_grids:,} grids ({pct:.1f}%) - "
                    f"{stats['cells_corrected']:,} cells corrected so far"
                )
        
        # Flush any remaining grids
        flush_pending()
    
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
    if bag_type == 'VR':
        logger.info(f"Refinement grids processed: {stats['grids_processed']:,}")
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

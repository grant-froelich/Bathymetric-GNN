"""
Inference pipeline for bathymetric GNN processing.

Handles:
- Loading data
- Tile-based processing
- Graph construction
- Model inference
- Result merging
- Output generation
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from config import Config
from data import (
    BathymetricGrid,
    BathymetricLoader,
    BathymetricWriter,
    TileManager,
    TileMerger,
    GraphBuilder,
)
from .gnn import BathymetricGNN

logger = logging.getLogger(__name__)


class BathymetricPipeline:
    """
    Complete inference pipeline for bathymetric denoising.
    
    Usage:
        pipeline = BathymetricPipeline(config)
        pipeline.load_model("model.pt")
        results = pipeline.process("input.bag", "output.bag")
    """
    
    def __init__(self, config: Config):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        self.loader = BathymetricLoader()
        self.writer = BathymetricWriter()
        self.tile_manager = TileManager(
            tile_size=config.tile.tile_size,
            overlap=config.tile.overlap,
            min_valid_ratio=config.tile.min_valid_ratio,
        )
        self.graph_builder = GraphBuilder(
            connectivity=config.graph.connectivity,
            edge_features=config.graph.edge_features,
        )
        
        # Model (loaded later)
        self.model: Optional[BathymetricGNN] = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Pipeline initialized, device: {self.device}")
    
    def load_model(self, model_path: Union[str, Path]):
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint or use current config
        model_config = checkpoint.get('model_config', self.config.model)
        
        # Determine input channels from checkpoint
        in_channels = checkpoint.get('in_channels', 7)  # Default node features
        edge_dim = checkpoint.get('edge_dim', 3)  # Default edge features
        
        # Create model
        self.model = BathymetricGNN(
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
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
    
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        export_extras: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Process a bathymetric file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            export_extras: Whether to export confidence/classification layers
            
        Returns:
            Dictionary of result arrays
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        logger.info(f"Processing: {input_path}")
        
        # Load input data
        grid = self.loader.load(input_path)
        logger.info(f"Loaded grid: {grid.shape}, resolution: {grid.resolution}")
        
        # Initialize output merger
        merger = TileMerger(self.tile_manager)
        merger.initialize(
            grid_shape=grid.shape,
            channels=['cleaned_depth', 'classification', 'confidence', 'correction'],
        )
        
        # Process tiles
        num_tiles = 0
        _, _, specs = self.tile_manager.compute_tile_grid(grid.shape)
        
        for tile in self.tile_manager.iterate_tiles(grid, skip_empty=True):
            tile_results = self._process_tile(tile, grid)
            
            # Find the spec for this tile
            spec = next(s for s in specs 
                       if s.tile_row == tile.tile_row and s.tile_col == tile.tile_col)
            
            merger.add_tile(spec, tile_results)
            num_tiles += 1
            
            if num_tiles % 10 == 0:
                logger.info(f"Processed {num_tiles}/{len(specs)} tiles")
        
        logger.info(f"Processed {num_tiles} tiles")
        
        # Finalize results
        results = merger.finalize()
        
        # Apply corrections to get cleaned depth
        cleaned_depth = self._apply_corrections(grid, results)
        results['cleaned_depth'] = cleaned_depth
        
        # Create output grid
        output_grid = BathymetricGrid(
            depth=cleaned_depth,
            uncertainty=grid.uncertainty,  # Preserve original uncertainty
            nodata_value=grid.nodata_value,
            transform=grid.transform,
            crs=grid.crs,
            resolution=grid.resolution,
            bounds=grid.bounds,
            source_path=input_path,
        )
        
        # Save output
        additional_bands = None
        if export_extras:
            additional_bands = {
                'classification': results['classification'],
                'confidence': results['confidence'],
            }
        
        self.writer.save(output_grid, output_path, additional_bands=additional_bands)
        logger.info(f"Saved output: {output_path}")
        
        # Generate summary
        self._log_summary(grid, results)
        
        return results
    
    def _process_tile(
        self,
        tile,
        grid: BathymetricGrid,
    ) -> Dict[str, np.ndarray]:
        """Process a single tile."""
        # Build graph from tile
        graph = self.graph_builder.build_graph(
            depth=tile.data,
            valid_mask=tile.valid_mask,
            uncertainty=tile.uncertainty,
            resolution=grid.resolution,
        )
        
        if graph.num_nodes == 0:
            # Return empty results for invalid tile
            return {
                'cleaned_depth': tile.data,
                'classification': np.zeros(tile.shape, dtype=np.float32),
                'confidence': np.zeros(tile.shape, dtype=np.float32),
                'correction': np.zeros(tile.shape, dtype=np.float32),
            }
        
        # Move to device
        graph = graph.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model.predict(
                graph,
                auto_correct_threshold=self.config.inference.auto_correct_threshold,
                review_threshold=self.config.inference.review_threshold,
            )
        
        # Convert outputs back to grids
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
        
        correction = np.zeros(tile.shape, dtype=np.float32)
        if 'correction' in outputs:
            correction = self.graph_builder.graph_to_grid(
                graph.cpu(),
                outputs['correction'].cpu(),
                fill_value=0.0,
            )
        
        return {
            'cleaned_depth': tile.data,  # Original, corrections applied later
            'classification': classification,
            'confidence': confidence,
            'correction': correction,
        }
    
    def _apply_corrections(
        self,
        grid: BathymetricGrid,
        results: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Apply corrections to generate cleaned depth.
        
        Correction strategy:
        - High confidence noise: apply correction
        - High confidence feature: preserve original
        - Low confidence: blend based on confidence
        """
        original = grid.depth.copy()
        classification = results['classification']
        confidence = results['confidence']
        correction = results['correction']
        
        cleaned = original.copy()
        
        # Get noise mask
        is_noise = classification == BathymetricGNN.CLASS_NOISE
        high_confidence = confidence > self.config.inference.auto_correct_threshold
        
        # Apply corrections to high-confidence noise
        apply_mask = is_noise & high_confidence & grid.valid_mask
        cleaned[apply_mask] = original[apply_mask] - correction[apply_mask]
        
        logger.info(
            f"Applied corrections to {np.sum(apply_mask)} cells "
            f"({100*np.sum(apply_mask)/np.sum(grid.valid_mask):.1f}% of valid)"
        )
        
        return cleaned
    
    def _log_summary(
        self,
        grid: BathymetricGrid,
        results: Dict[str, np.ndarray],
    ):
        """Log processing summary."""
        valid_mask = grid.valid_mask
        num_valid = np.sum(valid_mask)
        
        classification = results['classification']
        confidence = results['confidence']
        
        # Count classifications
        num_seafloor = np.sum((classification == 0) & valid_mask)
        num_feature = np.sum((classification == 1) & valid_mask)
        num_noise = np.sum((classification == 2) & valid_mask)
        
        # Confidence statistics
        mean_conf = np.mean(confidence[valid_mask])
        low_conf = np.sum((confidence < self.config.inference.review_threshold) & valid_mask)
        
        logger.info("=" * 50)
        logger.info("Processing Summary")
        logger.info("=" * 50)
        logger.info(f"Total valid cells: {num_valid:,}")
        logger.info(f"Classifications:")
        logger.info(f"  - Smooth seafloor: {num_seafloor:,} ({100*num_seafloor/num_valid:.1f}%)")
        logger.info(f"  - Features: {num_feature:,} ({100*num_feature/num_valid:.1f}%)")
        logger.info(f"  - Noise: {num_noise:,} ({100*num_noise/num_valid:.1f}%)")
        logger.info(f"Mean confidence: {mean_conf:.3f}")
        logger.info(f"Cells needing review: {low_conf:,} ({100*low_conf/num_valid:.1f}%)")
        logger.info("=" * 50)

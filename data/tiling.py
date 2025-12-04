"""
Tile management for large bathymetric grids.

Handles:
- Splitting large grids into overlapping tiles
- Merging processed tiles back into full grid
- Edge blending to avoid seams
- Memory-efficient iteration
"""

import logging
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np

from .loaders import BathymetricGrid

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """A tile extracted from a larger grid."""
    data: np.ndarray                         # Tile depth data
    uncertainty: Optional[np.ndarray]        # Tile uncertainty (if available)
    row_start: int                           # Start row in full grid
    col_start: int                           # Start column in full grid
    row_end: int                             # End row in full grid (exclusive)
    col_end: int                             # End column in full grid (exclusive)
    tile_row: int                            # Tile row index
    tile_col: int                            # Tile column index
    valid_mask: np.ndarray                   # Boolean mask of valid data
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    @property
    def valid_ratio(self) -> float:
        return np.sum(self.valid_mask) / self.valid_mask.size


@dataclass
class TileSpec:
    """Specification for a tile without loading data."""
    row_start: int
    col_start: int
    row_end: int
    col_end: int
    tile_row: int
    tile_col: int


class TileManager:
    """
    Manages tile-based processing of large bathymetric grids.
    
    Splits grids into overlapping tiles for processing and handles
    merging results back into a seamless output.
    """
    
    def __init__(
        self,
        tile_size: int = 1024,
        overlap: int = 128,
        min_valid_ratio: float = 0.1,
    ):
        """
        Initialize tile manager.
        
        Args:
            tile_size: Size of each tile (pixels)
            overlap: Overlap between adjacent tiles (pixels)
            min_valid_ratio: Minimum ratio of valid data to process a tile
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_valid_ratio = min_valid_ratio
        
        # Effective stride between tiles
        self.stride = tile_size - overlap
        
        if self.stride <= 0:
            raise ValueError("Tile size must be larger than overlap")
    
    def compute_tile_grid(
        self,
        grid_shape: Tuple[int, int]
    ) -> Tuple[int, int, List[TileSpec]]:
        """
        Compute tile specifications for a grid.
        
        Args:
            grid_shape: (height, width) of the full grid
            
        Returns:
            (num_tile_rows, num_tile_cols, list of TileSpecs)
        """
        height, width = grid_shape
        
        # Calculate number of tiles needed
        num_tile_rows = max(1, (height - self.overlap) // self.stride + 
                          (1 if (height - self.overlap) % self.stride > 0 else 0))
        num_tile_cols = max(1, (width - self.overlap) // self.stride +
                          (1 if (width - self.overlap) % self.stride > 0 else 0))
        
        specs = []
        
        for tile_row in range(num_tile_rows):
            for tile_col in range(num_tile_cols):
                row_start = tile_row * self.stride
                col_start = tile_col * self.stride
                
                row_end = min(row_start + self.tile_size, height)
                col_end = min(col_start + self.tile_size, width)
                
                # Adjust start for edge tiles to maintain tile_size when possible
                if row_end - row_start < self.tile_size and row_start > 0:
                    row_start = max(0, row_end - self.tile_size)
                if col_end - col_start < self.tile_size and col_start > 0:
                    col_start = max(0, col_end - self.tile_size)
                
                specs.append(TileSpec(
                    row_start=row_start,
                    col_start=col_start,
                    row_end=row_end,
                    col_end=col_end,
                    tile_row=tile_row,
                    tile_col=tile_col,
                ))
        
        logger.info(
            f"Grid {height}x{width} -> {num_tile_rows}x{num_tile_cols} = "
            f"{len(specs)} tiles"
        )
        
        return num_tile_rows, num_tile_cols, specs
    
    def extract_tile(
        self,
        grid: BathymetricGrid,
        spec: TileSpec,
    ) -> Tile:
        """
        Extract a single tile from the grid.
        
        Args:
            grid: Full bathymetric grid
            spec: Tile specification
            
        Returns:
            Tile object with data and metadata
        """
        data = grid.depth[spec.row_start:spec.row_end, spec.col_start:spec.col_end].copy()
        
        uncertainty = None
        if grid.uncertainty is not None:
            uncertainty = grid.uncertainty[
                spec.row_start:spec.row_end,
                spec.col_start:spec.col_end
            ].copy()
        
        # Compute valid mask
        valid_mask = np.isfinite(data)
        if grid.nodata_value is not None and not np.isnan(grid.nodata_value):
            valid_mask &= (data != grid.nodata_value)
        
        return Tile(
            data=data,
            uncertainty=uncertainty,
            row_start=spec.row_start,
            col_start=spec.col_start,
            row_end=spec.row_end,
            col_end=spec.col_end,
            tile_row=spec.tile_row,
            tile_col=spec.tile_col,
            valid_mask=valid_mask,
        )
    
    def iterate_tiles(
        self,
        grid: BathymetricGrid,
        skip_empty: bool = True,
    ) -> Generator[Tile, None, None]:
        """
        Iterate over tiles in a grid.
        
        Args:
            grid: Full bathymetric grid
            skip_empty: Skip tiles with insufficient valid data
            
        Yields:
            Tile objects
        """
        _, _, specs = self.compute_tile_grid(grid.shape)
        
        for spec in specs:
            tile = self.extract_tile(grid, spec)
            
            if skip_empty and tile.valid_ratio < self.min_valid_ratio:
                logger.debug(
                    f"Skipping tile ({tile.tile_row}, {tile.tile_col}) - "
                    f"valid ratio {tile.valid_ratio:.2%}"
                )
                continue
            
            yield tile
    
    def create_output_grid(
        self,
        grid_shape: Tuple[int, int],
        dtype: np.dtype = np.float32,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """Create an empty output grid."""
        return np.full(grid_shape, fill_value, dtype=dtype)
    
    def merge_tile(
        self,
        output: np.ndarray,
        tile_data: np.ndarray,
        spec: TileSpec,
        weight_grid: Optional[np.ndarray] = None,
    ):
        """
        Merge a processed tile into the output grid with blending.
        
        Args:
            output: Output grid to merge into
            tile_data: Processed tile data
            spec: Tile specification
            weight_grid: Optional weight accumulator for averaging
        """
        # Create blending weights (higher in center, lower at edges)
        weights = self._create_blend_weights(
            (spec.row_end - spec.row_start, spec.col_end - spec.col_start)
        )
        
        # Extract current region
        region = output[spec.row_start:spec.row_end, spec.col_start:spec.col_end]
        
        if weight_grid is not None:
            # Weighted averaging mode
            weight_region = weight_grid[spec.row_start:spec.row_end, spec.col_start:spec.col_end]
            
            # Handle NaN in tile_data
            valid_mask = np.isfinite(tile_data)
            
            # Update weights
            weight_region[valid_mask] += weights[valid_mask]
            
            # Update values (accumulate weighted sum)
            np.add.at(
                region,
                np.where(valid_mask),
                (tile_data * weights)[valid_mask]
            )
        else:
            # Simple overwrite with blending
            valid_mask = np.isfinite(tile_data)
            existing_valid = np.isfinite(region)
            
            # Where both are valid, blend
            both_valid = valid_mask & existing_valid
            new_only = valid_mask & ~existing_valid
            
            region[both_valid] = (
                region[both_valid] * (1 - weights[both_valid]) +
                tile_data[both_valid] * weights[both_valid]
            )
            region[new_only] = tile_data[new_only]
    
    def finalize_output(
        self,
        output: np.ndarray,
        weight_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Finalize the output grid after all tiles are merged.
        
        Args:
            output: Accumulated output grid
            weight_grid: Weight accumulator (if using weighted averaging)
            
        Returns:
            Finalized output grid
        """
        if weight_grid is not None:
            # Divide by weights to get average
            valid = weight_grid > 0
            output[valid] /= weight_grid[valid]
        
        return output
    
    def _create_blend_weights(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create blend weights that are higher in the center and taper at edges.
        
        This creates smooth transitions in overlap regions.
        """
        height, width = shape
        
        # Create 1D ramps
        row_weights = self._create_1d_blend(height)
        col_weights = self._create_1d_blend(width)
        
        # Combine into 2D weights
        weights = np.outer(row_weights, col_weights)
        
        return weights.astype(np.float32)
    
    def _create_1d_blend(self, size: int) -> np.ndarray:
        """Create 1D blend weights with ramps at edges."""
        weights = np.ones(size, dtype=np.float32)
        
        ramp_size = min(self.overlap, size // 4)
        
        if ramp_size > 0:
            # Linear ramp at start
            weights[:ramp_size] = np.linspace(0, 1, ramp_size)
            # Linear ramp at end
            weights[-ramp_size:] = np.linspace(1, 0, ramp_size)
        
        return weights


class TileMerger:
    """
    Handles merging of multiple output channels from tile processing.
    
    Supports merging:
    - Cleaned depth
    - Classification maps
    - Confidence maps
    - Any additional outputs
    """
    
    def __init__(self, tile_manager: TileManager):
        self.tile_manager = tile_manager
        self.outputs = {}
        self.weights = {}
    
    def initialize(
        self,
        grid_shape: Tuple[int, int],
        channels: List[str],
        dtypes: Optional[dict] = None,
    ):
        """
        Initialize output grids for specified channels.
        
        Args:
            grid_shape: Shape of the full output grid
            channels: List of channel names to create
            dtypes: Optional dict of channel -> dtype
        """
        if dtypes is None:
            dtypes = {}
        
        for channel in channels:
            dtype = dtypes.get(channel, np.float32)
            self.outputs[channel] = np.full(grid_shape, np.nan, dtype=dtype)
            self.weights[channel] = np.zeros(grid_shape, dtype=np.float32)
        
        logger.info(f"Initialized {len(channels)} output channels for {grid_shape}")
    
    def add_tile(
        self,
        spec: TileSpec,
        channel_data: dict,
    ):
        """
        Add processed tile data for all channels.
        
        Args:
            spec: Tile specification
            channel_data: Dict of channel_name -> tile_data
        """
        for channel, data in channel_data.items():
            if channel not in self.outputs:
                raise ValueError(f"Unknown channel: {channel}")
            
            self.tile_manager.merge_tile(
                self.outputs[channel],
                data,
                spec,
                self.weights[channel],
            )
    
    def finalize(self) -> dict:
        """
        Finalize all output channels.
        
        Returns:
            Dict of channel_name -> finalized array
        """
        results = {}
        
        for channel in self.outputs:
            results[channel] = self.tile_manager.finalize_output(
                self.outputs[channel],
                self.weights[channel],
            )
        
        # Clear internal state
        self.outputs = {}
        self.weights = {}
        
        return results

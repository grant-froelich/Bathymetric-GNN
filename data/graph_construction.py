"""
Graph construction for bathymetric data.

Converts gridded bathymetric data into graph representation suitable for GNN processing.

Graph structure:
- Nodes: Grid cells with valid depth values
- Edges: Spatial connections between neighboring cells
- Node features: Depth, local statistics, gradients
- Edge features: Distance, slope, aspect difference
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds PyTorch Geometric graph structures from gridded bathymetric data.
    """
    
    def __init__(
        self,
        connectivity: str = "8-connected",
        include_self_loops: bool = False,
        node_features: Optional[List[str]] = None,
        edge_features: Optional[List[str]] = None,
    ):
        """
        Initialize graph builder.
        
        Args:
            connectivity: "4-connected" or "8-connected"
            include_self_loops: Whether to include self-loop edges
            node_features: List of node features to compute
            edge_features: List of edge features to compute
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required. "
                "Install via: conda install pyg -c pyg"
            )
        
        self.connectivity = connectivity
        self.include_self_loops = include_self_loops
        
        # Default node features
        self.node_features = node_features or [
            "depth",
            "local_mean",
            "local_std",
            "gradient_x",
            "gradient_y",
            "gradient_magnitude",
            "curvature",
        ]
        
        # Default edge features
        self.edge_features = edge_features or [
            "distance",
            "depth_difference",
            "slope",
        ]
        
        # Neighbor offsets based on connectivity
        if connectivity == "4-connected":
            self.neighbor_offsets = [
                (-1, 0), (1, 0), (0, -1), (0, 1)
            ]
        elif connectivity == "8-connected":
            self.neighbor_offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
        else:
            raise ValueError(f"Unknown connectivity: {connectivity}")
    
    def build_graph(
        self,
        depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        uncertainty: Optional[np.ndarray] = None,
        resolution: Tuple[float, float] = (1.0, 1.0),
    ) -> Data:
        """
        Build a graph from gridded depth data.
        
        Args:
            depth: 2D depth array
            valid_mask: Boolean mask of valid cells (computed if not provided)
            uncertainty: Optional uncertainty array
            resolution: (x_resolution, y_resolution) in grid units
            
        Returns:
            PyTorch Geometric Data object
        """
        if valid_mask is None:
            valid_mask = np.isfinite(depth)
        
        logger.debug(f"Building graph from {depth.shape} grid, {np.sum(valid_mask)} valid cells")
        
        # Get valid cell coordinates
        valid_rows, valid_cols = np.where(valid_mask)
        num_nodes = len(valid_rows)
        
        if num_nodes == 0:
            logger.warning("No valid cells in grid")
            return self._create_empty_graph()
        
        # Create mapping from (row, col) to node index using a grid array
        # (Fix #5: numpy index grid replaces Python dict for O(1) array lookups)
        node_index_grid = np.full(depth.shape, -1, dtype=np.int64)
        node_index_grid[valid_rows, valid_cols] = np.arange(num_nodes)
        
        # Build edges
        edge_index, edge_coords = self._build_edges(
            valid_rows, valid_cols, node_index_grid, depth.shape
        )
        
        # Compute node features (also returns per-node local_std for correction normalization)
        node_features, node_local_std = self._compute_node_features(
            depth, valid_rows, valid_cols, uncertainty, valid_mask
        )
        
        # Compute edge features
        edge_features = self._compute_edge_features(
            depth, edge_coords, resolution
        )
        
        # Store grid position for later reconstruction
        pos = torch.tensor(
            np.stack([valid_cols, valid_rows], axis=1),
            dtype=torch.float32
        )
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            pos=pos,
        )
        
        # Store metadata for reconstruction
        data.grid_shape = depth.shape
        data.valid_rows = torch.tensor(valid_rows, dtype=torch.long)
        data.valid_cols = torch.tensor(valid_cols, dtype=torch.long)
        data.num_valid_cells = num_nodes
        
        # Store local_std per node for correction normalization/denormalization.
        # Training normalizes correction targets by local_std so the model learns
        # corrections in units of local variability rather than raw meters.
        # Inference denormalizes by multiplying predicted correction by local_std.
        data.local_std = node_local_std
        
        logger.debug(
            f"Built graph: {data.num_nodes} nodes, {data.num_edges} edges, "
            f"{data.x.shape[1]} node features, {data.edge_attr.shape[1]} edge features"
        )
        
        return data
    
    def _build_edges(
        self,
        valid_rows: np.ndarray,
        valid_cols: np.ndarray,
        node_index_grid: np.ndarray,
        grid_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """Build edge index tensor using vectorized numpy operations.
        
        Uses a node_index_grid (2D array mapping grid coords to node indices,
        -1 for invalid) instead of a Python dict for fast neighbor lookups.
        """
        height, width = grid_shape
        num_nodes = len(valid_rows)
        
        all_src = []
        all_tgt = []
        all_edge_coords = []
        node_indices = np.arange(num_nodes)
        
        for dr, dc in self.neighbor_offsets:
            # Compute neighbor coordinates for ALL valid nodes at once
            nr = valid_rows + dr
            nc = valid_cols + dc
            
            # Bounds check (vectorized)
            in_bounds = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
            
            # For out-of-bounds, clip to valid range to avoid index errors
            # (we'll filter these out with in_bounds mask)
            nr_safe = np.clip(nr, 0, height - 1)
            nc_safe = np.clip(nc, 0, width - 1)
            
            # Look up neighbor node indices from the grid
            neighbor_idx = node_index_grid[nr_safe, nc_safe]
            
            # Valid edge: in bounds AND neighbor is a valid node (index >= 0)
            valid_edge = in_bounds & (neighbor_idx >= 0)
            
            all_src.append(node_indices[valid_edge])
            all_tgt.append(neighbor_idx[valid_edge])
            
            # Store edge coordinates for feature computation
            src_r = valid_rows[valid_edge]
            src_c = valid_cols[valid_edge]
            tgt_r = nr[valid_edge]
            tgt_c = nc[valid_edge]
            all_edge_coords.extend(zip(src_r, src_c, tgt_r, tgt_c))
        
        # Add self loops if requested
        if self.include_self_loops:
            all_src.append(node_indices)
            all_tgt.append(node_indices)
            all_edge_coords.extend(zip(valid_rows, valid_cols, valid_rows, valid_cols))
        
        if len(all_src) > 0:
            src_array = np.concatenate(all_src)
            tgt_array = np.concatenate(all_tgt)
        else:
            src_array = np.array([], dtype=np.int64)
            tgt_array = np.array([], dtype=np.int64)
        
        edge_index = torch.tensor(
            np.stack([src_array, tgt_array]),
            dtype=torch.long
        )
        
        return edge_index, all_edge_coords
    
    def _compute_node_features(
        self,
        depth: np.ndarray,
        valid_rows: np.ndarray,
        valid_cols: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute features for each node using boundary-aware operations.
        
        All local statistics (mean, std, gradient, curvature) are computed
        using only valid neighbors. This prevents nodata values (1e6, NaN)
        from contaminating features near survey boundaries, which would
        otherwise create artificial signals the model mistakes for noise.
        
        Returns:
            Tuple of (node_features tensor, local_std tensor).
            local_std is returned separately for use in correction
            normalization/denormalization.
        """
        num_nodes = len(valid_rows)
        features = []
        
        # Build valid mask if not provided
        if valid_mask is None:
            valid_mask = np.isfinite(depth) & (np.abs(depth) < 1.0e5)
        
        # Precompute boundary-aware local statistics
        local_mean, local_std, valid_count = self._masked_local_stats(
            depth, valid_mask, size=5
        )
        
        # Fill invalid cells with local mean before computing gradient/curvature.
        # This prevents nodata values from creating false gradients at boundaries.
        # A cell at the survey edge will see gradients relative to the local
        # surface trend rather than a spike to nodata.
        depth_filled = np.where(valid_mask, depth, local_mean)
        depth_filled = np.nan_to_num(depth_filled, nan=0.0)
        
        grad_y, grad_x = np.gradient(depth_filled)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        curvature = self._compute_curvature(depth_filled, valid_mask)
        
        for feature_name in self.node_features:
            if feature_name == "depth":
                feat = depth[valid_rows, valid_cols]
            elif feature_name == "local_mean":
                feat = local_mean[valid_rows, valid_cols]
            elif feature_name == "local_std":
                feat = local_std[valid_rows, valid_cols]
            elif feature_name == "gradient_x":
                feat = grad_x[valid_rows, valid_cols]
            elif feature_name == "gradient_y":
                feat = grad_y[valid_rows, valid_cols]
            elif feature_name == "gradient_magnitude":
                feat = grad_mag[valid_rows, valid_cols]
            elif feature_name == "curvature":
                feat = curvature[valid_rows, valid_cols]
            elif feature_name == "uncertainty" and uncertainty is not None:
                feat = uncertainty[valid_rows, valid_cols]
            else:
                continue
            
            # Handle NaN values
            feat = np.nan_to_num(feat, nan=0.0)
            features.append(feat)
        
        # Add uncertainty if available and not already included
        if uncertainty is not None and "uncertainty" not in self.node_features:
            feat = uncertainty[valid_rows, valid_cols]
            feat = np.nan_to_num(feat, nan=0.0)
            features.append(feat)
        
        feature_matrix = np.stack(features, axis=1).astype(np.float32)
        
        # Extract per-node local_std for correction normalization
        node_local_std = local_std[valid_rows, valid_cols]
        node_local_std = np.nan_to_num(node_local_std, nan=0.0).astype(np.float32)
        
        return (
            torch.tensor(feature_matrix, dtype=torch.float32),
            torch.tensor(node_local_std, dtype=torch.float32),
        )
    
    def _compute_edge_features(
        self,
        depth: np.ndarray,
        edge_coords: List[Tuple],
        resolution: Tuple[float, float],
    ) -> torch.Tensor:
        """Compute features for each edge."""
        if len(edge_coords) == 0:
            return torch.zeros((0, len(self.edge_features)), dtype=torch.float32)
        
        features = []
        res_x, res_y = resolution
        
        for feature_name in self.edge_features:
            feat_values = []
            
            for src_r, src_c, tgt_r, tgt_c in edge_coords:
                if feature_name == "distance":
                    # Euclidean distance in real-world units
                    dx = (tgt_c - src_c) * res_x
                    dy = (tgt_r - src_r) * res_y
                    value = np.sqrt(dx**2 + dy**2)
                
                elif feature_name == "depth_difference":
                    value = depth[tgt_r, tgt_c] - depth[src_r, src_c]
                
                elif feature_name == "slope":
                    # Slope in degrees
                    dx = (tgt_c - src_c) * res_x
                    dy = (tgt_r - src_r) * res_y
                    dz = depth[tgt_r, tgt_c] - depth[src_r, src_c]
                    horizontal_dist = np.sqrt(dx**2 + dy**2)
                    if horizontal_dist > 0:
                        value = np.degrees(np.arctan(dz / horizontal_dist))
                    else:
                        value = 0.0
                
                else:
                    value = 0.0
                
                feat_values.append(value)
            
            feat_values = np.nan_to_num(feat_values, nan=0.0)
            features.append(feat_values)
        
        feature_matrix = np.stack(features, axis=1).astype(np.float32)
        
        return torch.tensor(feature_matrix, dtype=torch.float32)
    
    def _masked_local_stats(
        self,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        size: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute local mean and std using only valid neighbors.
        
        Uses a counting approach: sum valid values in each window and divide
        by the count of valid cells rather than the full kernel area. This
        prevents nodata cells from contaminating statistics near boundaries.
        
        Args:
            depth: 2D depth array (may contain nodata values)
            valid_mask: Boolean mask of valid cells
            size: Window size for local statistics
            
        Returns:
            Tuple of (local_mean, local_std, valid_count) arrays
        """
        # Zero out invalid cells so they don't contribute to sums
        depth_masked = np.where(valid_mask, depth, 0.0).astype(np.float64)
        valid_float = valid_mask.astype(np.float64)
        
        kernel_area = float(size * size)
        
        # uniform_filter computes the mean over the window, so multiply by
        # kernel_area to recover the sum. Use mode='constant', cval=0 so
        # cells outside the array boundary contribute nothing (not 'nearest'
        # which would replicate edge values).
        sum_vals = ndimage.uniform_filter(
            depth_masked, size=size, mode='constant', cval=0.0
        ) * kernel_area
        
        count = ndimage.uniform_filter(
            valid_float, size=size, mode='constant', cval=0.0
        ) * kernel_area
        
        # Avoid division by zero where no valid neighbors exist
        safe_count = np.maximum(count, 1.0)
        
        local_mean = (sum_vals / safe_count).astype(np.float32)
        
        # Masked standard deviation: E[x^2] - E[x]^2
        depth_sq_masked = np.where(valid_mask, depth.astype(np.float64)**2, 0.0)
        sum_sq = ndimage.uniform_filter(
            depth_sq_masked, size=size, mode='constant', cval=0.0
        ) * kernel_area
        
        mean_sq = sum_sq / safe_count
        variance = mean_sq - (sum_vals / safe_count)**2
        variance = np.maximum(variance, 0.0)  # Numerical stability
        local_std = np.sqrt(variance).astype(np.float32)
        
        return local_mean, local_std, count.astype(np.float32)
    
    def _compute_curvature(
        self,
        depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute surface curvature (Laplacian).
        
        Args:
            depth: 2D depth array (should be boundary-filled before calling)
            valid_mask: If provided, nodes with fewer than 3 valid neighbors
                in the 3x3 Laplacian kernel get curvature = 0.0 to avoid
                misleading edge artifacts.
        """
        curvature = ndimage.laplace(depth)
        
        if valid_mask is not None:
            # Count valid neighbors in the 3x3 Laplacian kernel
            kernel = np.ones((3, 3), dtype=np.float64)
            neighbor_count = ndimage.convolve(
                valid_mask.astype(np.float64), kernel, mode='constant', cval=0.0
            )
            # Zero out curvature where fewer than 3 valid cells in kernel
            curvature[neighbor_count < 3] = 0.0
        
        return curvature
    
    def _create_empty_graph(self) -> Data:
        """Create an empty graph for invalid tiles."""
        data = Data(
            x=torch.zeros((0, len(self.node_features)), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, len(self.edge_features)), dtype=torch.float32),
            pos=torch.zeros((0, 2), dtype=torch.float32),
        )
        data.local_std = torch.zeros(0, dtype=torch.float32)
        return data
    
    def graph_to_grid(
        self,
        data: Data,
        node_values: torch.Tensor,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        Convert node values back to grid format.
        
        Args:
            data: Original Data object (contains grid metadata)
            node_values: Tensor of values for each node
            fill_value: Value to use for invalid cells
            
        Returns:
            2D numpy array with values placed at original grid positions
        """
        if not hasattr(data, 'grid_shape'):
            raise ValueError("Data object missing grid_shape metadata")
        
        grid = np.full(data.grid_shape, fill_value, dtype=np.float32)
        
        if node_values.dim() == 1:
            # Single value per node
            values = node_values.detach().cpu().numpy()
            rows = data.valid_rows.numpy()
            cols = data.valid_cols.numpy()
            grid[rows, cols] = values
        else:
            # Multiple values per node - return list of grids
            raise ValueError(
                "For multi-channel node values, call graph_to_grid for each channel"
            )
        
        return grid


class MultiScaleGraphBuilder(GraphBuilder):
    """
    Builds multi-scale graph representations.
    
    Creates hierarchical graph structure with:
    - Fine scale: All valid cells
    - Coarse scales: Downsampled representations
    
    This allows the GNN to reason at multiple spatial scales.
    """
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        **kwargs
    ):
        """
        Initialize multi-scale graph builder.
        
        Args:
            scales: List of scale factors (1 = original, 2 = 2x downsampled, etc.)
            **kwargs: Arguments passed to parent GraphBuilder
        """
        super().__init__(**kwargs)
        self.scales = sorted(scales)
    
    def build_multiscale_graph(
        self,
        depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        uncertainty: Optional[np.ndarray] = None,
        resolution: Tuple[float, float] = (1.0, 1.0),
    ) -> Dict[int, Data]:
        """
        Build graphs at multiple scales.
        
        Args:
            depth: 2D depth array
            valid_mask: Boolean mask of valid cells
            uncertainty: Optional uncertainty array
            resolution: Grid resolution
            
        Returns:
            Dict mapping scale factor to Data object
        """
        if valid_mask is None:
            valid_mask = np.isfinite(depth)
        
        graphs = {}
        
        for scale in self.scales:
            if scale == 1:
                # Full resolution
                scaled_depth = depth
                scaled_mask = valid_mask
                scaled_unc = uncertainty
                scaled_res = resolution
            else:
                # Downsample
                scaled_depth = self._downsample(depth, scale)
                scaled_mask = self._downsample_mask(valid_mask, scale)
                scaled_unc = self._downsample(uncertainty, scale) if uncertainty is not None else None
                scaled_res = (resolution[0] * scale, resolution[1] * scale)
            
            graphs[scale] = self.build_graph(
                scaled_depth,
                scaled_mask,
                scaled_unc,
                scaled_res,
            )
            
            logger.debug(f"Scale {scale}: {graphs[scale].num_nodes} nodes")
        
        return graphs
    
    def _downsample(self, arr: np.ndarray, factor: int) -> np.ndarray:
        """Downsample array by averaging."""
        if arr is None:
            return None
        
        h, w = arr.shape
        new_h, new_w = h // factor, w // factor
        
        # Crop to exact multiple of factor
        cropped = arr[:new_h * factor, :new_w * factor]
        
        # Reshape and average
        reshaped = cropped.reshape(new_h, factor, new_w, factor)
        return np.nanmean(reshaped, axis=(1, 3))
    
    def _downsample_mask(self, mask: np.ndarray, factor: int) -> np.ndarray:
        """Downsample mask (True if majority of cells are valid)."""
        h, w = mask.shape
        new_h, new_w = h // factor, w // factor
        
        cropped = mask[:new_h * factor, :new_w * factor]
        reshaped = cropped.reshape(new_h, factor, new_w, factor)
        
        # Cell is valid if at least half of contributing cells are valid
        return np.mean(reshaped, axis=(1, 3)) >= 0.5

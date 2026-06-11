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

# Minimum cell footprint (meters) used as a floor before taking the log,
# to guard against zero or malformed resolution values producing -inf.
FOOTPRINT_FLOOR = 0.1


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
            "log_footprint",
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
            return self._create_empty_graph(include_uncertainty=uncertainty is not None)
        
        # Build edges (vectorized; no per-node Python loop)
        edge_index, edge_coords = self._build_edges(
            valid_rows, valid_cols, valid_mask, depth.shape
        )
        
        # Compute node features (also returns per-node local_std for correction normalization)
        node_features, node_local_std = self._compute_node_features(
            depth, valid_rows, valid_cols, uncertainty, valid_mask, resolution
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
        valid_mask: np.ndarray,
        grid_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Build edge index tensor (vectorized).
        
        For each neighbor offset, all candidate neighbors are evaluated at once
        via a node-id lookup grid, replacing the previous per-node Python loop
        and dict lookups. Edge ordering is offset-major (all edges for offset 0,
        then offset 1, ...) rather than the old node-major ordering; GNN message
        passing is permutation-invariant over edges, so this has no semantic
        effect. Equivalence to the loop implementation (same edge set, same
        per-edge features) is checked by scripts/verify_graph_equivalence.py.
        
        Returns:
            (edge_index tensor [2, E],
             edge_coords as four int arrays (src_r, src_c, tgt_r, tgt_c))
        """
        height, width = grid_shape
        num_nodes = len(valid_rows)
        
        # Node-id lookup grid: -1 for invalid cells, node index for valid ones
        node_id = np.full(grid_shape, -1, dtype=np.int64)
        node_id[valid_rows, valid_cols] = np.arange(num_nodes, dtype=np.int64)
        
        src_parts, tgt_parts = [], []
        src_r_parts, src_c_parts, tgt_r_parts, tgt_c_parts = [], [], [], []
        
        for dr, dc in self.neighbor_offsets:
            nr = valid_rows + dr
            nc = valid_cols + dc
            in_bounds = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
            
            neighbor_ids = np.full(num_nodes, -1, dtype=np.int64)
            neighbor_ids[in_bounds] = node_id[nr[in_bounds], nc[in_bounds]]
            
            has_edge = neighbor_ids >= 0
            if not np.any(has_edge):
                continue
            
            src_parts.append(np.nonzero(has_edge)[0].astype(np.int64))
            tgt_parts.append(neighbor_ids[has_edge])
            src_r_parts.append(valid_rows[has_edge])
            src_c_parts.append(valid_cols[has_edge])
            tgt_r_parts.append(nr[has_edge])
            tgt_c_parts.append(nc[has_edge])
        
        # Self loops if requested
        if self.include_self_loops:
            all_ids = np.arange(num_nodes, dtype=np.int64)
            src_parts.append(all_ids)
            tgt_parts.append(all_ids)
            src_r_parts.append(valid_rows)
            src_c_parts.append(valid_cols)
            tgt_r_parts.append(valid_rows)
            tgt_c_parts.append(valid_cols)
        
        if src_parts:
            source_nodes = np.concatenate(src_parts)
            target_nodes = np.concatenate(tgt_parts)
            edge_coords = (
                np.concatenate(src_r_parts),
                np.concatenate(src_c_parts),
                np.concatenate(tgt_r_parts),
                np.concatenate(tgt_c_parts),
            )
        else:
            source_nodes = np.zeros(0, dtype=np.int64)
            target_nodes = np.zeros(0, dtype=np.int64)
            empty = np.zeros(0, dtype=np.int64)
            edge_coords = (empty, empty, empty, empty)
        
        edge_index = torch.tensor(
            np.stack([source_nodes, target_nodes], axis=0),
            dtype=torch.long
        )
        
        return edge_index, edge_coords
    
    def _compute_node_features(
        self,
        depth: np.ndarray,
        valid_rows: np.ndarray,
        valid_cols: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
        resolution: Tuple[float, float] = (1.0, 1.0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute features for each node using boundary-aware operations.
        
        All local statistics (mean, std, gradient, curvature) are computed
        using only valid neighbors. This prevents nodata values (1e6, NaN)
        from contaminating features near survey boundaries, which would
        otherwise create artificial signals the model mistakes for noise.
        
        The log_footprint feature encodes the cell resolution (footprint) so
        the model can condition its behavior on scale. For SR surveys this is
        constant across all nodes; for VR surveys (once per-cell resolution is
        preserved through loading) it varies per node. Expressed as log2 of the
        footprint in meters so that equal resolution ratios are equal distances.
        
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
        curvature = self._compute_curvature(depth_filled)
        
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
            elif feature_name == "log_footprint":
                # Cell footprint in meters (use the coarser of x/y resolution).
                # Constant per SR survey; per-node once VR resolution is preserved.
                # log2 so equal resolution ratios map to equal feature distances.
                footprint = max(abs(resolution[0]), abs(resolution[1]))
                log_fp = float(np.log2(max(footprint, FOOTPRINT_FLOOR)))
                feat = np.full(num_nodes, log_fp, dtype=np.float32)
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
        edge_coords: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        resolution: Tuple[float, float],
    ) -> torch.Tensor:
        """Compute features for each edge (vectorized)."""
        src_r, src_c, tgt_r, tgt_c = edge_coords
        num_edges = len(src_r)
        
        if num_edges == 0:
            return torch.zeros((0, len(self.edge_features)), dtype=torch.float32)
        
        res_x, res_y = resolution
        
        # Shared geometry, computed once for all edges
        dx = (tgt_c - src_c).astype(np.float64) * res_x
        dy = (tgt_r - src_r).astype(np.float64) * res_y
        horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
        dz = (depth[tgt_r, tgt_c] - depth[src_r, src_c]).astype(np.float64)
        
        features = []
        for feature_name in self.edge_features:
            if feature_name == "distance":
                feat_values = horizontal_dist
            elif feature_name == "depth_difference":
                feat_values = dz
            elif feature_name == "slope":
                # Slope in degrees; 0 where horizontal distance is 0 (self loops)
                feat_values = np.where(
                    horizontal_dist > 0,
                    np.degrees(np.arctan(np.divide(
                        dz, horizontal_dist,
                        out=np.zeros_like(dz),
                        where=horizontal_dist > 0,
                    ))),
                    0.0,
                )
            else:
                feat_values = np.zeros(num_edges, dtype=np.float64)
            
            features.append(np.nan_to_num(feat_values, nan=0.0))
        
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
    
    def _local_std(self, arr: np.ndarray, size: int = 5) -> np.ndarray:
        """Compute local standard deviation using uniform filter.
        
        Note: This legacy method does NOT handle boundaries correctly.
        Use _masked_local_stats instead for boundary-aware computation.
        Kept for backward compatibility with any external callers.
        """
        arr_sq = ndimage.uniform_filter(arr**2, size=size, mode='nearest')
        arr_mean = ndimage.uniform_filter(arr, size=size, mode='nearest')
        variance = arr_sq - arr_mean**2
        variance = np.maximum(variance, 0)  # Numerical stability
        return np.sqrt(variance)
    
    def _compute_curvature(self, depth: np.ndarray) -> np.ndarray:
        """Compute surface curvature (Laplacian)."""
        return ndimage.laplace(depth)
    
    def _create_empty_graph(self, include_uncertainty: bool = False) -> Data:
        """Create an empty graph for invalid tiles.
        
        The feature width must match what build_graph produces for non-empty
        tiles (node_features plus an appended uncertainty channel when
        uncertainty data is present), or batching an empty graph with real
        ones would fail on mismatched dimensions.
        """
        num_features = len(self.node_features)
        if include_uncertainty and "uncertainty" not in self.node_features:
            num_features += 1
        
        data = Data(
            x=torch.zeros((0, num_features), dtype=torch.float32),
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

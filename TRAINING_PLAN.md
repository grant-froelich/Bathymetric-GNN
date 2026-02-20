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
        
        # Create mapping from (row, col) to node index
        coord_to_node = {}
        for node_idx, (r, c) in enumerate(zip(valid_rows, valid_cols)):
            coord_to_node[(r, c)] = node_idx
        
        # Build edges
        edge_index, edge_coords = self._build_edges(
            valid_rows, valid_cols, coord_to_node, depth.shape
        )
        
        # Compute node features
        node_features = self._compute_node_features(
            depth, valid_rows, valid_cols, uncertainty
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
        
        logger.debug(
            f"Built graph: {data.num_nodes} nodes, {data.num_edges} edges, "
            f"{data.x.shape[1]} node features, {data.edge_attr.shape[1]} edge features"
        )
        
        return data
    
    def _build_edges(
        self,
        valid_rows: np.ndarray,
        valid_cols: np.ndarray,
        coord_to_node: dict,
        grid_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """Build edge index tensor."""
        height, width = grid_shape
        
        source_nodes = []
        target_nodes = []
        edge_coords = []  # Store (src_row, src_col, tgt_row, tgt_col) for feature computation
        
        for node_idx, (r, c) in enumerate(zip(valid_rows, valid_cols)):
            for dr, dc in self.neighbor_offsets:
                nr, nc = r + dr, c + dc
                
                # Check bounds
                if 0 <= nr < height and 0 <= nc < width:
                    # Check if neighbor is valid
                    if (nr, nc) in coord_to_node:
                        neighbor_idx = coord_to_node[(nr, nc)]
                        source_nodes.append(node_idx)
                        target_nodes.append(neighbor_idx)
                        edge_coords.append((r, c, nr, nc))
        
        # Add self loops if requested
        if self.include_self_loops:
            for node_idx, (r, c) in enumerate(zip(valid_rows, valid_cols)):
                source_nodes.append(node_idx)
                target_nodes.append(node_idx)
                edge_coords.append((r, c, r, c))
        
        edge_index = torch.tensor(
            [source_nodes, target_nodes],
            dtype=torch.long
        )
        
        return edge_index, edge_coords
    
    def _compute_node_features(
        self,
        depth: np.ndarray,
        valid_rows: np.ndarray,
        valid_cols: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Compute features for each node."""
        num_nodes = len(valid_rows)
        features = []
        
        # Precompute derived arrays
        local_mean = ndimage.uniform_filter(depth, size=5, mode='nearest')
        local_std = self._local_std(depth, size=5)
        grad_y, grad_x = np.gradient(depth)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        curvature = self._compute_curvature(depth)
        
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
        
        return torch.tensor(feature_matrix, dtype=torch.float32)
    
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
    
    def _local_std(self, arr: np.ndarray, size: int = 5) -> np.ndarray:
        """Compute local standard deviation using uniform filter."""
        arr_sq = ndimage.uniform_filter(arr**2, size=size, mode='nearest')
        arr_mean = ndimage.uniform_filter(arr, size=size, mode='nearest')
        variance = arr_sq - arr_mean**2
        variance = np.maximum(variance, 0)  # Numerical stability
        return np.sqrt(variance)
    
    def _compute_curvature(self, depth: np.ndarray) -> np.ndarray:
        """Compute surface curvature (Laplacian)."""
        return ndimage.laplace(depth)
    
    def _create_empty_graph(self) -> Data:
        """Create an empty graph for invalid tiles."""
        return Data(
            x=torch.zeros((0, len(self.node_features)), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, len(self.edge_features)), dtype=torch.float32),
            pos=torch.zeros((0, 2), dtype=torch.float32),
        )
    
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

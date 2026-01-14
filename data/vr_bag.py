"""
Native BAG handler for both Variable Resolution (VR) and Single Resolution (SR) BAGs.

Processes BAG files without resampling:
- VR BAGs: Iterates through refinement grids, preserving multi-resolution structure
- SR BAGs: Processes the full elevation grid directly

Both modes support writing corrections back to the original BAG format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Union
from dataclasses import dataclass
import shutil

import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

logger = logging.getLogger(__name__)


def detect_bag_type(path: Path) -> str:
    """
    Detect whether a BAG file is Variable Resolution (VR) or Single Resolution (SR).
    
    Args:
        path: Path to BAG file
        
    Returns:
        'VR' for Variable Resolution BAG
        'SR' for Single Resolution BAG
        
    Raises:
        ValueError if file is not a valid BAG
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required for BAG handling")
    
    with h5py.File(str(path), 'r') as f:
        if 'BAG_root' not in f:
            raise ValueError(f"Not a valid BAG file: {path}")
        
        root = f['BAG_root']
        
        # VR BAGs have varres_refinements dataset
        if 'varres_refinements' in root and 'varres_metadata' in root:
            # Check if there are actual refinements
            varres_metadata = root['varres_metadata'][:]
            dims_x = varres_metadata['dimensions_x']
            if np.any(dims_x > 0):
                return 'VR'
        
        # Otherwise it's an SR BAG
        return 'SR'


@dataclass
class RefinementGrid:
    """A single refinement grid from a VR BAG."""
    # Location in base grid
    base_row: int
    base_col: int
    
    # Refinement data
    depth: np.ndarray           # 2D depth array
    uncertainty: np.ndarray     # 2D uncertainty array
    
    # Metadata
    resolution: Tuple[float, float]  # (x_res, y_res) in meters
    dimensions: Tuple[int, int]      # (rows, cols)
    sw_corner: Tuple[float, float]   # Offset within base cell
    
    # Index into varres_refinements
    start_index: int
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.depth.shape
    
    @property
    def valid_mask(self) -> np.ndarray:
        """Return mask of valid (non-nodata) cells."""
        nodata = 1.0e6
        return (self.depth != nodata) & np.isfinite(self.depth)
    
    @property
    def num_valid(self) -> int:
        return int(np.sum(self.valid_mask))


class VRBagHandler:
    """
    Handler for native VR BAG processing.
    
    Allows iterating through refinement grids, processing them,
    and writing corrections back while preserving VR structure.
    """
    
    NODATA = 1.0e6
    INVALID_INDEX = 4294967295  # Max uint32, indicates no refinement
    
    def __init__(self, path: Path):
        """
        Initialize VR BAG handler.
        
        Args:
            path: Path to VR BAG file
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for VR BAG handling")
        
        self.path = Path(path)
        self._validate_vr_bag()
        
        # Cache metadata
        with h5py.File(str(self.path), 'r') as f:
            root = f['BAG_root']
            self.base_shape = root['elevation'].shape
            self.varres_metadata = root['varres_metadata'][:]
            
            # Get geospatial info
            elevation = root['elevation']
            self.min_depth = elevation.attrs.get('Minimum Elevation Value', None)
            self.max_depth = elevation.attrs.get('Maximum Elevation Value', None)
        
        # Get geotransform using GDAL
        self._load_geospatial_info()
    
    def _load_geospatial_info(self):
        """Load geospatial information using GDAL."""
        try:
            from osgeo import gdal
            ds = gdal.Open(str(self.path))
            if ds:
                self.geotransform = ds.GetGeoTransform()
                self.crs = ds.GetProjection()
                ds = None
            else:
                self.geotransform = None
                self.crs = None
        except ImportError:
            self.geotransform = None
            self.crs = None
    
    @property
    def base_cell_size(self) -> Tuple[float, float]:
        """Size of base grid cells in CRS units."""
        if self.geotransform:
            return (abs(self.geotransform[1]), abs(self.geotransform[5]))
        # Fallback: estimate from metadata
        res_x = self.varres_metadata['resolution_x']
        dims_x = self.varres_metadata['dimensions_x']
        valid = dims_x > 0
        if np.any(valid):
            # Base cell size = refinement_resolution * refinement_dimensions
            max_size_x = np.max(res_x[valid] * dims_x[valid])
            return (float(max_size_x), float(max_size_x))
        return (50.0, 50.0)  # Default
    
    @property
    def finest_resolution(self) -> float:
        """Finest refinement resolution in CRS units."""
        res_x = self.varres_metadata['resolution_x']
        valid = res_x > 0
        if np.any(valid):
            return float(np.min(res_x[valid]))
        return 1.0
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Geographic bounds (min_x, min_y, max_x, max_y)."""
        if self.geotransform:
            min_x = self.geotransform[0]
            max_y = self.geotransform[3]
            max_x = min_x + self.base_shape[1] * self.geotransform[1]
            min_y = max_y + self.base_shape[0] * self.geotransform[5]
            return (min_x, min_y, max_x, max_y)
        return (0, 0, self.base_shape[1] * 50, self.base_shape[0] * 50)
    
    @property
    def resampled_shape(self) -> Tuple[int, int]:
        """Shape of grid if resampled to finest resolution."""
        bounds = self.bounds
        res = self.finest_resolution
        width = int(np.ceil((bounds[2] - bounds[0]) / res))
        height = int(np.ceil((bounds[3] - bounds[1]) / res))
        return (height, width)
    
    def _validate_vr_bag(self):
        """Check that file is a valid VR BAG."""
        with h5py.File(str(self.path), 'r') as f:
            if 'BAG_root' not in f:
                raise ValueError(f"Not a valid BAG file: {self.path}")
            
            root = f['BAG_root']
            if 'varres_refinements' not in root:
                raise ValueError(f"Not a VR BAG (no varres_refinements): {self.path}")
            
            if 'varres_metadata' not in root:
                raise ValueError(f"Not a VR BAG (no varres_metadata): {self.path}")
    
    @property
    def num_refinement_cells(self) -> int:
        """Number of base cells that have refinements."""
        dims_x = self.varres_metadata['dimensions_x']
        return int(np.sum(dims_x > 0))
    
    @property
    def total_refinement_nodes(self) -> int:
        """Total number of refinement grid cells."""
        dims_x = self.varres_metadata['dimensions_x']
        dims_y = self.varres_metadata['dimensions_y']
        return int(np.sum(dims_x.astype(np.int64) * dims_y.astype(np.int64)))
    
    def get_refinement_info(self) -> Dict:
        """Get summary information about refinements."""
        dims_x = self.varres_metadata['dimensions_x']
        dims_y = self.varres_metadata['dimensions_y']
        res_x = self.varres_metadata['resolution_x']
        
        has_refinement = dims_x > 0
        
        return {
            'base_shape': self.base_shape,
            'num_refined_cells': int(np.sum(has_refinement)),
            'total_refinement_nodes': self.total_refinement_nodes,
            'unique_dimensions': sorted(set(zip(
                dims_x[has_refinement].flatten(),
                dims_y[has_refinement].flatten()
            ))),
            'unique_resolutions': sorted(set(res_x[has_refinement].flatten())),
        }
    
    def iterate_refinements(
        self,
        min_valid_ratio: float = 0.0,
    ) -> Generator[RefinementGrid, None, None]:
        """
        Iterate through all refinement grids.
        
        Args:
            min_valid_ratio: Skip grids with less than this ratio of valid data
            
        Yields:
            RefinementGrid objects
        """
        with h5py.File(str(self.path), 'r') as f:
            root = f['BAG_root']
            refinements = root['varres_refinements']
            
            # Refinements stored as (1, N) array of structured records
            ref_data = refinements[0, :]
            
            # Iterate through base grid
            for row in range(self.base_shape[0]):
                for col in range(self.base_shape[1]):
                    meta = self.varres_metadata[row, col]
                    
                    dims_x = int(meta['dimensions_x'])
                    dims_y = int(meta['dimensions_y'])
                    
                    if dims_x == 0 or dims_y == 0:
                        continue  # No refinement for this cell
                    
                    start_idx = int(meta['index'])
                    num_cells = dims_x * dims_y
                    
                    # Extract refinement data
                    ref_slice = ref_data[start_idx:start_idx + num_cells]
                    
                    # Reshape to 2D grid (row-major order)
                    depth = ref_slice['depth'].reshape(dims_y, dims_x)
                    uncertainty = ref_slice['depth_uncrt'].reshape(dims_y, dims_x)
                    
                    grid = RefinementGrid(
                        base_row=row,
                        base_col=col,
                        depth=depth.copy(),
                        uncertainty=uncertainty.copy(),
                        resolution=(float(meta['resolution_x']), float(meta['resolution_y'])),
                        dimensions=(dims_y, dims_x),
                        sw_corner=(float(meta['sw_corner_x']), float(meta['sw_corner_y'])),
                        start_index=start_idx,
                    )
                    
                    # Check valid ratio
                    valid_ratio = grid.num_valid / grid.depth.size
                    if valid_ratio >= min_valid_ratio:
                        yield grid
    
    def copy_and_open_for_writing(self, output_path: Path) -> 'VRBagWriter':
        """
        Copy BAG to output location and open for writing.
        
        Args:
            output_path: Path for output BAG
            
        Returns:
            VRBagWriter for modifying the copy
        """
        logger.info(f"Copying VR BAG: {self.path} -> {output_path}")
        shutil.copy(str(self.path), str(output_path))
        return VRBagWriter(output_path)


class VRBagWriter:
    """
    Writer for modifying VR BAG refinement data.
    
    Used in conjunction with VRBagHandler to apply corrections.
    """
    
    NODATA = 1.0e6
    
    def __init__(self, path: Path):
        """
        Open VR BAG for writing.
        
        Args:
            path: Path to VR BAG file (should be a copy)
        """
        self.path = Path(path)
        self._file = h5py.File(str(self.path), 'r+')
        self._root = self._file['BAG_root']
        self._refinements = self._root['varres_refinements']
        self._uncertainty = None
        if 'uncertainty' in self._root:
            self._uncertainty = self._root['uncertainty']
        
        # Load metadata for validation
        self._metadata = self._root['varres_metadata'][:]
        
        self._corrections_applied = 0
        self._uncertainty_updates = 0
    
    def update_refinement(
        self,
        grid: RefinementGrid,
        corrected_depth: np.ndarray,
        corrected_uncertainty: Optional[np.ndarray] = None,
    ):
        """
        Write corrected depth values back to refinement grid.
        
        Args:
            grid: Original RefinementGrid (for location info)
            corrected_depth: New depth values (same shape as grid.depth)
            corrected_uncertainty: New uncertainty values (optional)
        """
        if corrected_depth.shape != grid.shape:
            raise ValueError(
                f"Shape mismatch: corrected {corrected_depth.shape} vs grid {grid.shape}"
            )
        
        start_idx = grid.start_index
        num_cells = grid.dimensions[0] * grid.dimensions[1]
        
        # Flatten to 1D for writing
        flat_depth = corrected_depth.flatten()
        
        # Update depth values
        for i, depth in enumerate(flat_depth):
            self._refinements[0, start_idx + i, 'depth'] = depth
        
        # Track changes
        changed = corrected_depth != grid.depth
        self._corrections_applied += int(np.sum(changed & grid.valid_mask))
        
        # Update uncertainty if provided
        if corrected_uncertainty is not None:
            flat_uncert = corrected_uncertainty.flatten()
            for i, uncert in enumerate(flat_uncert):
                self._refinements[0, start_idx + i, 'depth_uncrt'] = uncert
            self._uncertainty_updates += int(np.sum(
                (corrected_uncertainty != grid.uncertainty) & grid.valid_mask
            ))
    
    def update_refinement_batch(
        self,
        grid: RefinementGrid,
        corrected_depth: np.ndarray,
        corrected_uncertainty: Optional[np.ndarray] = None,
    ):
        """
        Write corrected depth values using batch update (faster).
        
        Args:
            grid: Original RefinementGrid
            corrected_depth: New depth values
            corrected_uncertainty: New uncertainty values (optional)
        """
        if corrected_depth.shape != grid.shape:
            raise ValueError(
                f"Shape mismatch: corrected {corrected_depth.shape} vs grid {grid.shape}"
            )
        
        start_idx = grid.start_index
        num_cells = grid.dimensions[0] * grid.dimensions[1]
        end_idx = start_idx + num_cells
        
        # Read current data
        current = self._refinements[0, start_idx:end_idx]
        
        # Update depth
        current['depth'] = corrected_depth.flatten()
        
        # Update uncertainty if provided
        if corrected_uncertainty is not None:
            current['depth_uncrt'] = corrected_uncertainty.flatten()
        
        # Write back
        self._refinements[0, start_idx:end_idx] = current
        
        # Track changes
        changed = corrected_depth != grid.depth
        self._corrections_applied += int(np.sum(changed & grid.valid_mask))
    
    def close(self):
        """Close file and log summary."""
        if self._file is not None:
            self._file.close()
            self._file = None
            
        logger.info(f"VR BAG modifications complete:")
        logger.info(f"  - Depth corrections applied: {self._corrections_applied:,}")
        if self._uncertainty_updates > 0:
            logger.info(f"  - Uncertainty updates: {self._uncertainty_updates:,}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Single Resolution (SR) BAG Support
# =============================================================================

@dataclass
class SRGrid:
    """Represents the full grid from an SR BAG (analogous to RefinementGrid)."""
    depth: np.ndarray           # 2D depth array
    uncertainty: np.ndarray     # 2D uncertainty array
    resolution: Tuple[float, float]  # (x_res, y_res) in meters
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.depth.shape
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.depth.shape
    
    @property
    def valid_mask(self) -> np.ndarray:
        """Return mask of valid (non-nodata) cells."""
        nodata = 1.0e6
        return (self.depth != nodata) & np.isfinite(self.depth)
    
    @property
    def num_valid(self) -> int:
        return int(np.sum(self.valid_mask))


class SRBagHandler:
    """
    Handler for Single Resolution BAG processing.
    
    Provides a similar interface to VRBagHandler for consistency,
    but operates on the single elevation/uncertainty grid.
    """
    
    NODATA = 1.0e6
    
    def __init__(self, path: Path):
        """
        Initialize SR BAG handler.
        
        Args:
            path: Path to SR BAG file
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for BAG handling")
        
        self.path = Path(path)
        self._validate_sr_bag()
        
        # Cache metadata
        with h5py.File(str(self.path), 'r') as f:
            root = f['BAG_root']
            self.shape = root['elevation'].shape
            self.base_shape = self.shape  # For compatibility with VR interface
            
            # Read elevation attributes
            elevation = root['elevation']
            self.min_depth = elevation.attrs.get('Minimum Elevation Value', None)
            self.max_depth = elevation.attrs.get('Maximum Elevation Value', None)
        
        # Get geospatial info using GDAL
        self._load_geospatial_info()
    
    def _validate_sr_bag(self):
        """Check that file is a valid SR BAG."""
        with h5py.File(str(self.path), 'r') as f:
            if 'BAG_root' not in f:
                raise ValueError(f"Not a valid BAG file: {self.path}")
            
            root = f['BAG_root']
            if 'elevation' not in root:
                raise ValueError(f"Not a valid BAG (no elevation): {self.path}")
    
    def _load_geospatial_info(self):
        """Load geospatial information using GDAL."""
        try:
            from osgeo import gdal
            ds = gdal.Open(str(self.path))
            if ds:
                self.geotransform = ds.GetGeoTransform()
                self.crs = ds.GetProjection()
                ds = None
            else:
                self.geotransform = None
                self.crs = None
        except ImportError:
            self.geotransform = None
            self.crs = None
    
    @property
    def resolution(self) -> Tuple[float, float]:
        """Grid resolution in CRS units (x_res, y_res)."""
        if self.geotransform:
            return (abs(self.geotransform[1]), abs(self.geotransform[5]))
        return (1.0, 1.0)  # Default fallback
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Geographic bounds (min_x, min_y, max_x, max_y)."""
        if self.geotransform:
            min_x = self.geotransform[0]
            max_y = self.geotransform[3]
            max_x = min_x + self.shape[1] * self.geotransform[1]
            min_y = max_y + self.shape[0] * self.geotransform[5]
            return (min_x, min_y, max_x, max_y)
        return (0, 0, self.shape[1], self.shape[0])
    
    def get_refinement_info(self) -> Dict:
        """Get summary information (compatible with VR interface)."""
        return {
            'base_shape': self.shape,
            'num_refined_cells': 1,  # SR has single grid
            'total_refinement_nodes': int(self.shape[0] * self.shape[1]),
            'unique_dimensions': [self.shape],
            'unique_resolutions': [self.resolution[0]],
            'bag_type': 'SR',
        }
    
    def iterate_refinements(
        self,
        min_valid_ratio: float = 0.0,
    ) -> Generator[SRGrid, None, None]:
        """
        Yield the single SR grid (compatible with VR iteration interface).
        
        Args:
            min_valid_ratio: Skip if less than this ratio of valid data
            
        Yields:
            Single SRGrid object
        """
        with h5py.File(str(self.path), 'r') as f:
            root = f['BAG_root']
            
            depth = root['elevation'][:]
            
            # Uncertainty may be in different locations
            if 'uncertainty' in root:
                uncertainty = root['uncertainty'][:]
            elif 'Uncertainty' in root:
                uncertainty = root['Uncertainty'][:]
            else:
                # Create default uncertainty
                uncertainty = np.ones_like(depth) * 0.5
            
            grid = SRGrid(
                depth=depth.copy(),
                uncertainty=uncertainty.copy(),
                resolution=self.resolution,
            )
            
            # Check valid ratio
            valid_ratio = grid.num_valid / max(1, grid.depth.size)
            if valid_ratio >= min_valid_ratio:
                yield grid
    
    def copy_and_open_for_writing(self, output_path: Path) -> 'SRBagWriter':
        """
        Copy BAG and open for writing corrections.
        
        Args:
            output_path: Path for output BAG
            
        Returns:
            SRBagWriter for the copied BAG
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy original file
        shutil.copy2(self.path, output_path)
        logger.info(f"Copied SR BAG to: {output_path}")
        
        return SRBagWriter(output_path)


class SRBagWriter:
    """
    Writer for SR BAG corrections.
    
    Writes corrected depth values back to the elevation dataset.
    """
    
    def __init__(self, path: Path):
        """
        Open SR BAG for writing.
        
        Args:
            path: Path to SR BAG file (must exist)
        """
        self.path = Path(path)
        self._file = h5py.File(str(self.path), 'r+')
        self._elevation = self._file['BAG_root']['elevation']
        
        # Try to get uncertainty dataset
        root = self._file['BAG_root']
        if 'uncertainty' in root:
            self._uncertainty = root['uncertainty']
        elif 'Uncertainty' in root:
            self._uncertainty = root['Uncertainty']
        else:
            self._uncertainty = None
        
        self._corrections_applied = 0
        self._uncertainty_updates = 0
        
        logger.info(f"Opened SR BAG for writing: {self.path}")
    
    def update_grid(
        self,
        grid: SRGrid,
        corrected_depth: np.ndarray,
        corrected_uncertainty: Optional[np.ndarray] = None,
    ):
        """
        Write corrected depth values to SR BAG.
        
        Args:
            grid: Original SRGrid (for shape validation)
            corrected_depth: New depth values
            corrected_uncertainty: New uncertainty values (optional)
        """
        if corrected_depth.shape != grid.shape:
            raise ValueError(
                f"Shape mismatch: corrected {corrected_depth.shape} vs grid {grid.shape}"
            )
        
        # Track changes before writing
        changed = corrected_depth != grid.depth
        self._corrections_applied += int(np.sum(changed & grid.valid_mask))
        
        # Write elevation
        self._elevation[:] = corrected_depth
        
        # Write uncertainty if provided and dataset exists
        if corrected_uncertainty is not None and self._uncertainty is not None:
            self._uncertainty[:] = corrected_uncertainty
            self._uncertainty_updates += int(np.sum(grid.valid_mask))
    
    # Alias for compatibility with VR interface
    def update_refinement_batch(
        self,
        grid: SRGrid,
        corrected_depth: np.ndarray,
        corrected_uncertainty: Optional[np.ndarray] = None,
    ):
        """Alias for update_grid (compatible with VR writer interface)."""
        self.update_grid(grid, corrected_depth, corrected_uncertainty)
    
    def close(self):
        """Close file and log summary."""
        if self._file is not None:
            self._file.close()
            self._file = None
            
        logger.info(f"SR BAG modifications complete:")
        logger.info(f"  - Depth corrections applied: {self._corrections_applied:,}")
        if self._uncertainty_updates > 0:
            logger.info(f"  - Uncertainty updates: {self._uncertainty_updates:,}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class SidecarBuilder:
    """
    Builds a sidecar GeoTIFF from native BAG processing results.
    
    Works with both VR and SR BAGs:
    - VR: Accumulates results at finest resolution from refinement grids
    - SR: Uses native grid resolution directly
    
    Uses GDAL to get exact georeferencing that matches how the BAG appears.
    """
    
    def __init__(self, handler: Union[VRBagHandler, SRBagHandler]):
        """
        Initialize sidecar builder.
        
        Args:
            handler: VRBagHandler or SRBagHandler for the source BAG
        """
        self.handler = handler
        self.is_vr = isinstance(handler, VRBagHandler)
        
        # Get georeferencing from GDAL
        self._init_from_gdal()
        
        # Initialize output arrays
        self.classification = np.full(self.shape, np.nan, dtype=np.float32)
        self.confidence = np.full(self.shape, np.nan, dtype=np.float32)
        self.correction = np.full(self.shape, np.nan, dtype=np.float32)
        self.valid_mask = np.zeros(self.shape, dtype=np.float32)
        
        bag_type = "VR" if self.is_vr else "SR"
        logger.info(f"SidecarBuilder initialized ({bag_type}): {self.shape} at {self.resolution}m")
    
    def _init_from_gdal(self):
        """Get georeferencing from GDAL."""
        try:
            from osgeo import gdal
            gdal.UseExceptions()
        except ImportError:
            raise ImportError("GDAL required for SidecarBuilder")
        
        path_str = str(self.handler.path)
        
        if self.is_vr:
            # For VR BAGs, use GDAL's resampled view
            open_options = ['MODE=RESAMPLED_GRID']
            ds = gdal.OpenEx(path_str, gdal.OF_RASTER | gdal.OF_READONLY, open_options=open_options)
            
            if ds is None:
                ds = gdal.Open(path_str)
            
            if ds is None:
                raise RuntimeError(f"Could not open BAG: {self.handler.path}")
            
            # Check if we got the resampled view by comparing dimensions
            if ds.RasterXSize == self.handler.base_shape[1] and ds.RasterYSize == self.handler.base_shape[0]:
                subdatasets = ds.GetSubDatasets()
                ds = None
                
                for sd_path, sd_desc in subdatasets:
                    if 'resampled' in sd_desc.lower() or 'supergrids' in sd_desc.lower():
                        ds = gdal.Open(sd_path)
                        if ds is not None:
                            break
            
            if ds is None:
                raise RuntimeError(f"Could not open resampled view: {self.handler.path}")
        else:
            # For SR BAGs, just open directly
            ds = gdal.Open(path_str)
            if ds is None:
                raise RuntimeError(f"Could not open BAG: {self.handler.path}")
        
        self.shape = (ds.RasterYSize, ds.RasterXSize)
        self.geotransform = ds.GetGeoTransform()
        self.crs = ds.GetProjection()
        
        # Resolution from geotransform
        self.resolution = abs(self.geotransform[1])
        
        # Calculate bounds from geotransform
        min_x = self.geotransform[0]
        max_y = self.geotransform[3]
        max_x = min_x + self.shape[1] * self.geotransform[1]
        min_y = max_y + self.shape[0] * self.geotransform[5]
        self.bounds = (min_x, min_y, max_x, max_y)
        
        ds = None
    
    def add_refinement_results(
        self,
        grid: Union[RefinementGrid, SRGrid],
        classification: np.ndarray,
        confidence: np.ndarray,
        correction: np.ndarray,
    ):
        """
        Add results from a processed grid.
        
        Args:
            grid: The grid (RefinementGrid for VR, SRGrid for SR)
            classification: Classification array
            confidence: Confidence array
            correction: Correction array
        """
        if not self.is_vr:
            # SR BAG: Direct copy (shapes should match)
            if classification.shape != self.shape:
                raise ValueError(
                    f"SR grid shape {classification.shape} doesn't match sidecar {self.shape}"
                )
            self.classification[:] = classification
            self.confidence[:] = confidence
            self.correction[:] = correction
            self.valid_mask[grid.valid_mask] = 1.0
            return
        
        # VR BAG: Map refinement grid to resampled output
        # Compute base cell size from resampled extent and base grid dimensions
        base_shape = self.handler.base_shape  # (512, 512) typically
        
        # Base cell size in ground units
        base_cell_width = (self.bounds[2] - self.bounds[0]) / base_shape[1]
        base_cell_height = (self.bounds[3] - self.bounds[1]) / base_shape[0]
        
        # Geographic position of this base cell's SW corner
        # Base grid row 0 is at the south (bottom), col 0 is at the west (left)
        base_x = self.bounds[0] + grid.base_col * base_cell_width
        base_y = self.bounds[1] + grid.base_row * base_cell_height
        
        # Add sw_corner offset (position of refinement grid within base cell)
        ref_x = base_x + grid.sw_corner[0]
        ref_y = base_y + grid.sw_corner[1]
        
        # Refinement grid size in ground units
        ref_width = grid.dimensions[1] * grid.resolution[0]
        ref_height = grid.dimensions[0] * grid.resolution[1]
        
        # Convert geographic coordinates to pixel coordinates
        # Note: geotransform origin is at top-left (northwest)
        gt = self.geotransform
        
        # Pixel column = (x - origin_x) / pixel_width
        out_col_start = int(round((ref_x - gt[0]) / gt[1]))
        
        # Pixel row = (y - origin_y) / pixel_height
        ref_top_y = ref_y + ref_height
        out_row_start = int(round((gt[3] - ref_top_y) / abs(gt[5])))
        
        # Refinement dimensions in output pixels
        ref_res = grid.resolution[0]
        scale = max(1, int(round(ref_res / self.resolution)))
        
        # Place each refinement cell into output grid
        # Refinement grid: row 0 is at south (bottom)
        # Output raster: row 0 is at north (top)
        for r in range(grid.dimensions[0]):
            for c in range(grid.dimensions[1]):
                # Flip row index (refinement bottom -> output top)
                out_r_base = out_row_start + (grid.dimensions[0] - 1 - r) * scale
                out_c_base = out_col_start + c * scale
                
                # Fill scaled region
                for dr in range(scale):
                    for dc in range(scale):
                        out_r = out_r_base + dr
                        out_c = out_c_base + dc
                        
                        # Bounds check
                        if 0 <= out_r < self.shape[0] and 0 <= out_c < self.shape[1]:
                            self.classification[out_r, out_c] = classification[r, c]
                            self.confidence[out_r, out_c] = confidence[r, c]
                            self.correction[out_r, out_c] = correction[r, c]
                            if grid.valid_mask[r, c]:
                                self.valid_mask[out_r, out_c] = 1.0
    
    def save(self, path: Path):
        """
        Save accumulated results as GeoTIFF.
        
        Args:
            path: Output path for GeoTIFF
        """
        try:
            from osgeo import gdal
        except ImportError:
            logger.error("GDAL required for GeoTIFF export")
            return
        
        logger.info(f"Saving sidecar GeoTIFF: {path}")
        
        height, width = self.shape
        num_bands = 4  # classification, confidence, correction, valid_mask
        
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(
            str(path),
            width,
            height,
            num_bands,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        
        try:
            # Use the geotransform we got from GDAL's resampled view
            if self.geotransform:
                ds.SetGeoTransform(self.geotransform)
            
            if self.crs:
                ds.SetProjection(self.crs)
            
            # Write bands
            bands = [
                ('classification', self.classification),
                ('confidence', self.confidence),
                ('correction', self.correction),
                ('valid_mask', self.valid_mask),
            ]
            
            for i, (name, data) in enumerate(bands, 1):
                band = ds.GetRasterBand(i)
                band.WriteArray(data)
                band.SetDescription(name)
            
            ds.FlushCache()
            
        finally:
            ds = None
        
        logger.info(f"Saved sidecar GeoTIFF with {num_bands} bands")


def process_vr_bag_native(
    input_path: Path,
    output_path: Path,
    process_func,
    min_valid_ratio: float = 0.01,
    confidence_for_uncertainty: bool = True,
) -> Dict:
    """
    Process a VR BAG natively without resampling.
    
    Args:
        input_path: Path to input VR BAG
        output_path: Path for output VR BAG
        process_func: Function that takes (depth, uncertainty, resolution) and returns
                     (classification, confidence, correction) arrays
        min_valid_ratio: Skip refinement grids with less valid data
        confidence_for_uncertainty: Scale uncertainty by confidence for corrected cells
        
    Returns:
        Dictionary with processing statistics
    """
    handler = VRBagHandler(input_path)
    
    info = handler.get_refinement_info()
    logger.info(f"VR BAG: {info['num_refined_cells']} refined cells, "
                f"{info['total_refinement_nodes']:,} total nodes")
    logger.info(f"Resolutions: {info['unique_resolutions']} meters")
    
    # Copy and open for writing
    writer = handler.copy_and_open_for_writing(output_path)
    
    stats = {
        'grids_processed': 0,
        'grids_skipped': 0,
        'cells_processed': 0,
        'cells_corrected': 0,
        'total_grids': info['num_refined_cells'],
    }
    
    try:
        for i, grid in enumerate(handler.iterate_refinements(min_valid_ratio)):
            # Process this refinement grid
            classification, confidence, correction = process_func(
                grid.depth,
                grid.uncertainty,
                grid.resolution,
            )
            
            # Apply corrections
            corrected_depth = grid.depth.copy()
            corrected_uncertainty = grid.uncertainty.copy()
            
            # Only correct high-confidence noise
            noise_mask = (classification == 2) & grid.valid_mask
            high_conf = confidence > 0.5
            apply_mask = noise_mask & high_conf
            
            if np.any(apply_mask):
                corrected_depth[apply_mask] += correction[apply_mask]
                stats['cells_corrected'] += int(np.sum(apply_mask))
                
                # Scale uncertainty by confidence
                if confidence_for_uncertainty:
                    scale = 2.0 - confidence[apply_mask]
                    corrected_uncertainty[apply_mask] *= scale
            
            # Write back
            writer.update_refinement_batch(
                grid,
                corrected_depth,
                corrected_uncertainty,
            )
            
            stats['grids_processed'] += 1
            stats['cells_processed'] += grid.num_valid
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{info['num_refined_cells']} refinement grids")
        
    finally:
        writer.close()
    
    logger.info(f"Native VR processing complete:")
    logger.info(f"  - Grids processed: {stats['grids_processed']:,}")
    logger.info(f"  - Cells processed: {stats['cells_processed']:,}")
    logger.info(f"  - Cells corrected: {stats['cells_corrected']:,}")
    
    return stats

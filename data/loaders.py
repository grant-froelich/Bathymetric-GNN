"""
Data loaders for bathymetric files.

Supports:
- ONSWG BAG format
- GeoTIFF
- ASCII Grid (ASC)
- XYZ point clouds (future)

Uses GDAL for file I/O to maintain compatibility with standard geospatial workflows.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

try:
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BathymetricGrid:
    """Container for bathymetric grid data with metadata."""
    depth: np.ndarray                        # 2D depth array (positive down or negative down)
    uncertainty: Optional[np.ndarray]        # 2D uncertainty array (if available)
    nodata_value: float                      # NoData value
    transform: Tuple[float, ...]             # Geotransform (origin_x, pixel_width, 0, origin_y, 0, pixel_height)
    crs: Optional[str]                       # Coordinate reference system (WKT or EPSG)
    resolution: Tuple[float, float]          # (x_resolution, y_resolution) in CRS units
    bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    source_path: Optional[Path]              # Original file path
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return (height, width) of the depth grid."""
        return self.depth.shape
    
    @property
    def valid_mask(self) -> np.ndarray:
        """Return boolean mask of valid (non-nodata) cells."""
        mask = np.isfinite(self.depth)
        if self.nodata_value is not None:
            mask &= (self.depth != self.nodata_value)
        return mask
    
    @property
    def valid_ratio(self) -> float:
        """Return ratio of valid cells to total cells."""
        return np.sum(self.valid_mask) / self.depth.size
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics on valid depth values."""
        valid_depths = self.depth[self.valid_mask]
        if len(valid_depths) == 0:
            return {}
        
        return {
            "min": float(np.min(valid_depths)),
            "max": float(np.max(valid_depths)),
            "mean": float(np.mean(valid_depths)),
            "std": float(np.std(valid_depths)),
            "median": float(np.median(valid_depths)),
            "count": int(len(valid_depths)),
            "valid_ratio": float(self.valid_ratio),
        }


class BathymetricLoader:
    """Load bathymetric data from various file formats."""
    
    SUPPORTED_FORMATS = {'.bag', '.tif', '.tiff', '.asc', '.xyz'}
    
    def __init__(self):
        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL is required for loading bathymetric data. "
                "Install via: conda install -c conda-forge gdal"
            )
        
        # Configure GDAL
        gdal.UseExceptions()
        gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')  # Disable .aux.xml files
    
    def load(self, path: Union[str, Path]) -> BathymetricGrid:
        """
        Load bathymetric data from file.
        
        Args:
            path: Path to bathymetric file
            
        Returns:
            BathymetricGrid containing depth data and metadata
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.bag':
            return self._load_bag(path)
        elif suffix in {'.tif', '.tiff'}:
            return self._load_geotiff(path)
        elif suffix == '.asc':
            return self._load_ascii(path)
        elif suffix == '.xyz':
            return self._load_xyz(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _load_bag(self, path: Path) -> BathymetricGrid:
        """Load ONSWG BAG format file."""
        logger.info(f"Loading BAG file: {path}")
        
        # BAG files are HDF5 with specific structure
        # GDAL can read them, but we may want direct H5 access for uncertainty
        
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise IOError(f"Failed to open BAG file: {path}")
        
        try:
            # Get depth band (typically band 1)
            depth_band = ds.GetRasterBand(1)
            depth = depth_band.ReadAsArray().astype(np.float32)
            nodata = depth_band.GetNoDataValue()
            
            # Get uncertainty band if available (typically band 2)
            uncertainty = None
            if ds.RasterCount >= 2:
                unc_band = ds.GetRasterBand(2)
                uncertainty = unc_band.ReadAsArray().astype(np.float32)
            
            # Get geospatial metadata
            transform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            
            # Calculate bounds
            width = ds.RasterXSize
            height = ds.RasterYSize
            min_x = transform[0]
            max_y = transform[3]
            max_x = min_x + width * transform[1]
            min_y = max_y + height * transform[5]  # transform[5] is negative
            
            return BathymetricGrid(
                depth=depth,
                uncertainty=uncertainty,
                nodata_value=nodata if nodata is not None else np.nan,
                transform=transform,
                crs=projection,
                resolution=(abs(transform[1]), abs(transform[5])),
                bounds=(min_x, min_y, max_x, max_y),
                source_path=path,
            )
            
        finally:
            ds = None  # Close dataset
    
    def _load_geotiff(self, path: Path) -> BathymetricGrid:
        """Load GeoTIFF format file."""
        logger.info(f"Loading GeoTIFF file: {path}")
        
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise IOError(f"Failed to open GeoTIFF file: {path}")
        
        try:
            depth_band = ds.GetRasterBand(1)
            depth = depth_band.ReadAsArray().astype(np.float32)
            nodata = depth_band.GetNoDataValue()
            
            # Check for uncertainty in band 2
            uncertainty = None
            if ds.RasterCount >= 2:
                unc_band = ds.GetRasterBand(2)
                uncertainty = unc_band.ReadAsArray().astype(np.float32)
            
            transform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            
            width = ds.RasterXSize
            height = ds.RasterYSize
            min_x = transform[0]
            max_y = transform[3]
            max_x = min_x + width * transform[1]
            min_y = max_y + height * transform[5]
            
            return BathymetricGrid(
                depth=depth,
                uncertainty=uncertainty,
                nodata_value=nodata if nodata is not None else np.nan,
                transform=transform,
                crs=projection,
                resolution=(abs(transform[1]), abs(transform[5])),
                bounds=(min_x, min_y, max_x, max_y),
                source_path=path,
            )
            
        finally:
            ds = None
    
    def _load_ascii(self, path: Path) -> BathymetricGrid:
        """Load ASCII grid format file."""
        logger.info(f"Loading ASCII grid file: {path}")
        
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise IOError(f"Failed to open ASCII file: {path}")
        
        try:
            depth_band = ds.GetRasterBand(1)
            depth = depth_band.ReadAsArray().astype(np.float32)
            nodata = depth_band.GetNoDataValue()
            
            transform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            
            width = ds.RasterXSize
            height = ds.RasterYSize
            min_x = transform[0]
            max_y = transform[3]
            max_x = min_x + width * transform[1]
            min_y = max_y + height * transform[5]
            
            return BathymetricGrid(
                depth=depth,
                uncertainty=None,
                nodata_value=nodata if nodata is not None else np.nan,
                transform=transform,
                crs=projection if projection else None,
                resolution=(abs(transform[1]), abs(transform[5])),
                bounds=(min_x, min_y, max_x, max_y),
                source_path=path,
            )
            
        finally:
            ds = None
    
    def _load_xyz(self, path: Path) -> BathymetricGrid:
        """
        Load XYZ point cloud and grid it.
        
        Note: This is a placeholder. Full implementation would need
        gridding parameters and interpolation options.
        """
        raise NotImplementedError(
            "XYZ point cloud loading requires gridding. "
            "Use a pre-gridded format or implement gridding logic."
        )


class BathymetricWriter:
    """Write bathymetric data to various file formats."""
    
    def __init__(self):
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required for writing bathymetric data.")
        gdal.UseExceptions()
    
    def save(
        self,
        grid: BathymetricGrid,
        path: Union[str, Path],
        format: Optional[str] = None,
        additional_bands: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Save bathymetric grid to file.
        
        Args:
            grid: BathymetricGrid to save
            path: Output file path
            format: Output format (inferred from extension if not specified)
            additional_bands: Dict of band_name -> array to include (e.g., confidence, classification)
        """
        path = Path(path)
        
        if format is None:
            format = path.suffix.lower()
        
        if format in {'.tif', '.tiff'}:
            self._save_geotiff(grid, path, additional_bands)
        elif format == '.bag':
            self._save_bag(grid, path, additional_bands)
        elif format == '.asc':
            self._save_ascii(grid, path)
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def _save_geotiff(
        self,
        grid: BathymetricGrid,
        path: Path,
        additional_bands: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Save as GeoTIFF with optional additional bands."""
        logger.info(f"Saving GeoTIFF: {path}")
        
        height, width = grid.shape
        num_bands = 1
        
        if grid.uncertainty is not None:
            num_bands += 1
        
        if additional_bands:
            num_bands += len(additional_bands)
        
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
            ds.SetGeoTransform(grid.transform)
            if grid.crs:
                ds.SetProjection(grid.crs)
            
            # Write depth band
            band_idx = 1
            depth_band = ds.GetRasterBand(band_idx)
            depth_band.WriteArray(grid.depth)
            depth_band.SetNoDataValue(grid.nodata_value)
            depth_band.SetDescription("Depth")
            band_idx += 1
            
            # Write uncertainty band if present
            if grid.uncertainty is not None:
                unc_band = ds.GetRasterBand(band_idx)
                unc_band.WriteArray(grid.uncertainty)
                unc_band.SetDescription("Uncertainty")
                band_idx += 1
            
            # Write additional bands
            if additional_bands:
                for name, data in additional_bands.items():
                    band = ds.GetRasterBand(band_idx)
                    band.WriteArray(data.astype(np.float32))
                    band.SetDescription(name)
                    band_idx += 1
            
            ds.FlushCache()
            
        finally:
            ds = None
        
        logger.info(f"Saved GeoTIFF with {num_bands} bands")
    
    def _save_bag(
        self,
        grid: BathymetricGrid,
        path: Path,
        additional_bands: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Save as BAG format.
        
        Note: Full BAG support requires proper metadata.
        This is a simplified implementation.
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for BAG output")
        
        logger.warning(
            "BAG output is simplified. For full BAG compliance, "
            "use official BAG tools or the source BAG as template."
        )
        
        # For now, fall back to GeoTIFF
        tiff_path = path.with_suffix('.tif')
        self._save_geotiff(grid, tiff_path, additional_bands)
        logger.info(f"Saved as GeoTIFF instead: {tiff_path}")
    
    def _save_ascii(self, grid: BathymetricGrid, path: Path):
        """Save as ASCII grid (depth only)."""
        logger.info(f"Saving ASCII grid: {path}")
        
        driver = gdal.GetDriverByName('AAIGrid')
        height, width = grid.shape
        
        ds = driver.Create(str(path), width, height, 1, gdal.GDT_Float32)
        
        try:
            ds.SetGeoTransform(grid.transform)
            if grid.crs:
                ds.SetProjection(grid.crs)
            
            band = ds.GetRasterBand(1)
            band.WriteArray(grid.depth)
            band.SetNoDataValue(grid.nodata_value)
            ds.FlushCache()
            
        finally:
            ds = None

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
    
    def __init__(self, vr_bag_mode: str = 'refinements'):
        """
        Initialize the loader.
        
        Args:
            vr_bag_mode: How to handle Variable Resolution BAGs:
                - 'refinements': Read refinement grids directly (default)
                - 'resampled': Resample to uniform grid using GDAL
                - 'base': Read only the base/coarse grid (not recommended)
        """
        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL is required for loading bathymetric data. "
                "Install via: conda install -c conda-forge gdal"
            )
        
        self.vr_bag_mode = vr_bag_mode
        
        # Configure GDAL
        gdal.UseExceptions()
        gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')  # Disable .aux.xml files
    
    def load(
        self, 
        path: Union[str, Path],
        vr_target_resolution: Optional[float] = None,
    ) -> BathymetricGrid:
        """
        Load bathymetric data from file.
        
        Args:
            path: Path to bathymetric file
            vr_target_resolution: For VR BAGs in resampled mode, target resolution in meters
            
        Returns:
            BathymetricGrid containing depth data and metadata
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.bag':
            return self._load_bag(path, vr_target_resolution)
        elif suffix in {'.tif', '.tiff'}:
            return self._load_geotiff(path)
        elif suffix == '.asc':
            return self._load_ascii(path)
        elif suffix == '.xyz':
            return self._load_xyz(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _load_bag(
        self, 
        path: Path,
        vr_target_resolution: Optional[float] = None,
    ) -> BathymetricGrid:
        """Load ONSWG BAG format file (supports both SR and VR BAGs)."""
        logger.info(f"Loading BAG file: {path}")
        
        # For resampled mode, use GDAL's built-in VR support directly
        if self.vr_bag_mode == 'resampled':
            return self._load_vr_bag_resampled(path, vr_target_resolution)
        
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise IOError(f"Failed to open BAG file: {path}")
        
        try:
            # Check for Variable Resolution BAG by looking at subdatasets
            subdatasets = ds.GetSubDatasets()
            
            # Look for VR refinement subdatasets
            vr_depth_sd = None
            vr_uncert_sd = None
            for sd_name, sd_desc in subdatasets:
                if 'varres_refinements' in sd_name.lower() or 'refinement' in sd_desc.lower():
                    if 'depth' in sd_desc.lower() or 'elevation' in sd_desc.lower():
                        vr_depth_sd = sd_name
                    elif 'uncertainty' in sd_desc.lower():
                        vr_uncert_sd = sd_name
            
            if vr_depth_sd and self.vr_bag_mode == 'refinements':
                logger.info(f"Detected Variable Resolution BAG, loading refinements")
                return self._load_vr_bag(path, vr_depth_sd, vr_uncert_sd, ds)
            else:
                # Single resolution BAG or base mode - use standard loading
                if vr_depth_sd and self.vr_bag_mode == 'base':
                    logger.warning(f"VR BAG detected but loading base grid only (vr_bag_mode='base')")
                return self._load_sr_bag(path, ds)
                
        finally:
            ds = None  # Close dataset
    
    def _load_sr_bag(self, path: Path, ds) -> BathymetricGrid:
        """Load Single Resolution BAG."""
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
    
    def _load_vr_bag(
        self, 
        path: Path, 
        depth_subdataset: str,
        uncert_subdataset: Optional[str],
        parent_ds,
    ) -> BathymetricGrid:
        """
        Load Variable Resolution BAG by reading refinement grids.
        
        VR BAGs store data at multiple resolutions. This method reads the
        refinement grids and mosaics them into a single grid at the finest
        available resolution.
        """
        # Get the base grid info for bounds and CRS
        base_transform = parent_ds.GetGeoTransform()
        projection = parent_ds.GetProjection()
        base_nodata = parent_ds.GetRasterBand(1).GetNoDataValue()
        
        # Try to open the refinement subdataset
        vr_ds = gdal.Open(depth_subdataset, gdal.GA_ReadOnly)
        if vr_ds is None:
            logger.warning(f"Could not open VR refinements, falling back to base grid")
            return self._load_sr_bag(path, parent_ds)
        
        try:
            # Read the refinement data
            depth = vr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            vr_transform = vr_ds.GetGeoTransform()
            nodata = vr_ds.GetRasterBand(1).GetNoDataValue()
            
            if nodata is None:
                nodata = base_nodata if base_nodata is not None else np.nan
            
            # Read uncertainty if available
            uncertainty = None
            if uncert_subdataset:
                unc_ds = gdal.Open(uncert_subdataset, gdal.GA_ReadOnly)
                if unc_ds:
                    uncertainty = unc_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                    unc_ds = None
            
            # Calculate bounds from refinement grid
            width = vr_ds.RasterXSize
            height = vr_ds.RasterYSize
            min_x = vr_transform[0]
            max_y = vr_transform[3]
            max_x = min_x + width * vr_transform[1]
            min_y = max_y + height * vr_transform[5]
            
            resolution = (abs(vr_transform[1]), abs(vr_transform[5]))
            
            logger.info(f"VR BAG loaded: {height}x{width} at {resolution[0]:.2f}m resolution")
            
            return BathymetricGrid(
                depth=depth,
                uncertainty=uncertainty,
                nodata_value=nodata if nodata is not None else np.nan,
                transform=vr_transform,
                crs=projection,
                resolution=resolution,
                bounds=(min_x, min_y, max_x, max_y),
                source_path=path,
            )
            
        finally:
            vr_ds = None
    
    def _load_vr_bag_resampled(
        self,
        path: Path,
        target_resolution: Optional[float] = None,
    ) -> BathymetricGrid:
        """
        Load VR BAG and resample to a uniform resolution grid.
        
        This uses GDAL's built-in VR BAG support with resampling.
        Useful when you need a consistent grid from VR data.
        
        Args:
            path: Path to BAG file
            target_resolution: Target resolution in meters. If None, uses finest available.
            
        Returns:
            BathymetricGrid at uniform resolution
        """
        # GDAL open options for VR BAG
        open_options = ['MODE=RESAMPLED_GRID']
        if target_resolution:
            open_options.append(f'RESX={target_resolution}')
            open_options.append(f'RESY={target_resolution}')
        
        ds = gdal.OpenEx(
            str(path), 
            gdal.OF_RASTER | gdal.OF_READONLY,
            open_options=open_options
        )
        
        if ds is None:
            raise IOError(f"Failed to open VR BAG in resampled mode: {path}")
        
        try:
            depth_band = ds.GetRasterBand(1)
            depth = depth_band.ReadAsArray().astype(np.float32)
            nodata = depth_band.GetNoDataValue()
            
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
            
            resolution = (abs(transform[1]), abs(transform[5]))
            
            logger.info(f"VR BAG resampled: {height}x{width} at {resolution[0]:.2f}m resolution")
            
            return BathymetricGrid(
                depth=depth,
                uncertainty=uncertainty,
                nodata_value=nodata if nodata is not None else np.nan,
                transform=transform,
                crs=projection,
                resolution=resolution,
                bounds=(min_x, min_y, max_x, max_y),
                source_path=path,
            )
            
        finally:
            ds = None
    
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
        Save as BAG by copying source and modifying in place.
        
        This preserves the original BAG structure (including VR refinements)
        while updating:
        - Elevation values (with corrections applied)
        - Uncertainty values (scaled by model confidence for corrected cells)
        
        Also creates sidecar GeoTIFF with classification/confidence/etc.
        
        Note: For VR BAGs loaded with resampled mode, the grid shape won't match
        the original, so we fall back to creating a new SR BAG at the resampled resolution.
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for BAG output")
        
        import h5py
        import shutil
        
        source_path = grid.source_path
        
        if source_path is None or not Path(source_path).exists():
            logger.warning("No source BAG available - creating new BAG")
            self._save_bag_new(grid, path, additional_bands)
            return
        
        # Check if grid shape matches source
        with h5py.File(str(source_path), 'r') as f:
            root = f['BAG_root']
            source_shape = root['elevation'].shape
            is_vr = 'varres_refinements' in root
        
        if grid.shape != source_shape:
            if is_vr:
                logger.warning(
                    f"Grid shape {grid.shape} doesn't match source VR BAG {source_shape}. "
                    f"This happens when loading VR BAG with resampled mode. "
                    f"Creating new SR BAG at resampled resolution."
                )
            else:
                logger.warning(
                    f"Grid shape {grid.shape} doesn't match source {source_shape}. "
                    f"Creating new BAG."
                )
            self._save_bag_new(grid, path, additional_bands)
            return
        
        # Copy source BAG to output location
        logger.info(f"Copying source BAG: {source_path} -> {path}")
        shutil.copy(str(source_path), str(path))
        
        # Extract correction info from additional_bands
        classification = additional_bands.get('classification') if additional_bands else None
        confidence = additional_bands.get('confidence') if additional_bands else None
        correction = additional_bands.get('correction') if additional_bands else None
        
        # Open copy and modify
        with h5py.File(str(path), 'r+') as f:
            root = f['BAG_root']
            
            # Check if this is a VR BAG
            is_vr = 'varres_refinements' in root
            
            if is_vr:
                logger.info("Detected VR BAG - modifying refinement grids")
                self._modify_vr_bag(root, grid, classification, confidence, correction)
            else:
                logger.info("Detected SR BAG - modifying elevation grid")
                self._modify_sr_bag(root, grid, classification, confidence, correction)
        
        logger.info(f"Saved modified BAG: {path}")
        
        # Save additional bands as sidecar GeoTIFF
        if additional_bands:
            sidecar_path = path.with_name(path.stem + '_gnn_outputs.tif')
            self._save_sidecar_geotiff(grid, sidecar_path, additional_bands)
    
    def _modify_sr_bag(
        self,
        root,
        grid: BathymetricGrid,
        classification: Optional[np.ndarray],
        confidence: Optional[np.ndarray],
        correction: Optional[np.ndarray],
    ):
        """Modify a single-resolution BAG in place."""
        elevation = root['elevation']
        original_depth = elevation[:]
        
        # Apply corrections
        cleaned_depth = original_depth.copy()
        if classification is not None and correction is not None:
            # Only correct high-confidence noise
            noise_mask = classification == 2  # CLASS_NOISE
            high_conf = confidence > 0.5 if confidence is not None else np.ones_like(noise_mask)
            apply_mask = noise_mask & high_conf
            
            if np.any(apply_mask):
                cleaned_depth[apply_mask] = original_depth[apply_mask] + correction[apply_mask]
                logger.info(f"Applied corrections to {np.sum(apply_mask):,} cells")
        
        # Write corrected depth
        elevation[...] = cleaned_depth
        
        # Update uncertainty if present
        if 'uncertainty' in root and confidence is not None:
            uncertainty = root['uncertainty']
            original_unc = uncertainty[:]
            
            # Scale uncertainty by confidence for corrected cells
            # High confidence (0.9) -> 1.1x uncertainty
            # Low confidence (0.5) -> 1.5x uncertainty
            # Not analyzed (0.0) -> unchanged
            modified_unc = original_unc.copy()
            
            corrected_mask = (classification == 2) & (confidence > 0)
            if np.any(corrected_mask):
                scale_factor = 2.0 - confidence[corrected_mask]
                modified_unc[corrected_mask] = original_unc[corrected_mask] * scale_factor
                logger.info(f"Scaled uncertainty for {np.sum(corrected_mask):,} corrected cells")
            
            uncertainty[...] = modified_unc
    
    def _modify_vr_bag(
        self,
        root,
        grid: BathymetricGrid,
        classification: Optional[np.ndarray],
        confidence: Optional[np.ndarray],
        correction: Optional[np.ndarray],
    ):
        """
        Modify a variable-resolution BAG in place.
        
        VR BAGs store high-res data in varres_refinements as a 1D array.
        We need to map our 2D corrections to the refinement indices.
        """
        # For now, modify the base grid
        # Full VR support would require mapping refinement cells
        if 'elevation' in root:
            self._modify_sr_bag(root, grid, classification, confidence, correction)
        
        # TODO: Map corrections to varres_refinements
        # This requires understanding the varres_metadata structure
        # which maps base grid cells to refinement indices
        logger.warning(
            "VR BAG refinements not yet modified - only base grid updated. "
            "Full VR support requires refinement grid mapping."
        )
    
    def _save_bag_new(
        self,
        grid: BathymetricGrid,
        path: Path,
        additional_bands: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Fall back to GeoTIFF when we can't copy/modify the source BAG.
        
        Creating a valid BAG from scratch requires proper XML metadata,
        specific HDF5 attributes, etc. that GDAL/Caris expect.
        Instead, we save as GeoTIFF which preserves all the data.
        """
        logger.warning(
            "Cannot create valid BAG without source structure. "
            "Saving as GeoTIFF instead (preserves all data and georeferencing)."
        )
        
        # Change extension to .tif
        tiff_path = path.with_suffix('.tif')
        
        # Save full GeoTIFF with all bands
        self._save_geotiff(grid, tiff_path, additional_bands)
        
        logger.info(f"Saved as GeoTIFF: {tiff_path}")
    
    def _save_sidecar_geotiff(
        self,
        grid: BathymetricGrid,
        path: Path,
        bands: Dict[str, np.ndarray],
    ):
        """Save GNN output bands as a sidecar GeoTIFF."""
        logger.info(f"Saving sidecar GeoTIFF: {path}")
        
        height, width = grid.shape
        num_bands = len(bands)
        
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
            
            band_idx = 1
            for name, data in bands.items():
                band = ds.GetRasterBand(band_idx)
                band.WriteArray(data.astype(np.float32))
                band.SetDescription(name)
                band_idx += 1
            
            ds.FlushCache()
            
        finally:
            ds = None
        
        logger.info(f"Saved sidecar GeoTIFF with {num_bands} bands: {list(bands.keys())}")
    
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

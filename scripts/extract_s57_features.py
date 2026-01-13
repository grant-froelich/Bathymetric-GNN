#!/usr/bin/env python3
"""
scripts/extract_s57_features.py

Extract feature locations from NOAA ENC data for training.

Supports two modes:
  1. Direct REST API queries to NOAA ENC Direct (no download required)
  2. Local S-57 ENC files (.000 format)

REST API endpoint (recommended, most up-to-date):
  - ENC Direct: https://encdirect.noaa.gov/arcgis/rest/services/encdirect/
  - Queries wrecks, obstructions, and rocks from official ENC data
  - Updated weekly

S-57 ENCs can be downloaded from:
  https://encdirect.noaa.gov/

Feature classes extracted:
  - WRECKS: Shipwrecks
  - UWTROC: Underwater rocks  
  - OBSTRN: Obstructions
  - SBDARE: Seabed area (substrate type)
  - SOUNDG: Soundings (individual depth points)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
from pathlib import Path
import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import urllib.request
import urllib.parse

try:
    from osgeo import ogr, osr, gdal
    gdal.UseExceptions()
    ogr.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# NOAA REST API Configuration
# =============================================================================

# ENC Direct services (primary source - updated weekly)
ENC_DIRECT_BASE = "https://encdirect.noaa.gov/arcgis/rest/services/encdirect"

ENC_SCALE_SERVICES = {
    'berthing': f"{ENC_DIRECT_BASE}/enc_berthing/MapServer",   # 1:5,000 or larger
    'harbour': f"{ENC_DIRECT_BASE}/enc_harbour/MapServer",     # 1:22,000 - 1:45,000
    'approach': f"{ENC_DIRECT_BASE}/enc_approach/MapServer",   # 1:45,001 - 1:90,000
    'coastal': f"{ENC_DIRECT_BASE}/enc_coastal/MapServer",     # 1:90,001 - 1:350,000
    'general': f"{ENC_DIRECT_BASE}/enc_general/MapServer",     # 1:350,001 - 1:1,500,000
}

# Layer name patterns in ENC Direct services
ENC_LAYER_PATTERNS = {
    'wreck_point': 'Wreck_point',
    'wreck_area': 'Wreck_area',
    'obstruction_point': 'Obstruction_point',
    'obstruction_area': 'Obstruction_area',
    'rock_point': 'Underwater_Awash_Rock_point',
    'seabed_area': 'Seabed_Area',
}

# Legacy Wrecks and Obstructions service (less up-to-date, kept for AWOIS historical data)
WRECKS_SERVICE_URL = "https://wrecks.nauticalcharts.noaa.gov/arcgis/rest/services/public_wrecks/Wrecks_And_Obstructions/MapServer"

# Layer IDs in legacy Wrecks service (for AWOIS only)
AWOIS_LAYERS = {
    'awois_wrecks': 8,
    'awois_obstructions': 14,
}


# =============================================================================
# REST API Query Functions
# =============================================================================

def query_arcgis_rest(
    service_url: str,
    layer_id: int,
    bounds: Tuple[float, float, float, float],
    out_fields: str = '*',
    max_records: int = 2000,
) -> List[Dict]:
    """
    Query ArcGIS REST service for features within bounds.
    
    Args:
        service_url: Base URL of MapServer
        layer_id: Layer ID to query
        bounds: (min_x, min_y, max_x, max_y) in WGS84
        out_fields: Fields to return ('*' for all)
        max_records: Maximum records to return
        
    Returns:
        List of feature dictionaries with geometry and attributes
    """
    query_url = f"{service_url}/{layer_id}/query"
    
    # Build geometry envelope
    geometry = json.dumps({
        'xmin': bounds[0],
        'ymin': bounds[1],
        'xmax': bounds[2],
        'ymax': bounds[3],
        'spatialReference': {'wkid': 4326}
    })
    
    params = {
        'where': '1=1',
        'geometry': geometry,
        'geometryType': 'esriGeometryEnvelope',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': out_fields,
        'returnGeometry': 'true',
        'outSR': '4326',
        'f': 'json',
        'resultRecordCount': max_records,
    }
    
    url = f"{query_url}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        logger.warning(f"Failed to query {query_url}: {e}")
        return []
    
    if 'error' in data:
        logger.warning(f"API error: {data['error']}")
        return []
    
    features = data.get('features', [])
    logger.debug(f"Retrieved {len(features)} features from layer {layer_id}")
    
    return features


def get_layer_id_by_name(service_url: str, name_pattern: str) -> Optional[int]:
    """Find layer ID by name pattern in MapServer."""
    try:
        url = f"{service_url}?f=json"
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        logger.warning(f"Failed to get service info: {e}")
        return None
    
    for layer in data.get('layers', []):
        if name_pattern.lower() in layer['name'].lower():
            return layer['id']
    
    return None


def query_wrecks_from_rest(
    bounds: Tuple[float, float, float, float],
    scales: List[str] = None,
    include_awois: bool = False,
) -> List['S57Feature']:
    """
    Query ENC Direct for wrecks within bounds.
    
    Args:
        bounds: (min_x, min_y, max_x, max_y) in WGS84
        scales: ENC scale bands to query (default: ['berthing', 'harbour', 'approach'])
        include_awois: Include historical AWOIS wrecks (from legacy service)
        
    Returns:
        List of S57Feature objects
    """
    if scales is None:
        scales = ['berthing', 'harbour', 'approach']
    
    features = []
    seen_positions = set()  # Deduplicate by position
    
    # Query ENC Direct at each scale band
    for scale in scales:
        service_url = ENC_SCALE_SERVICES.get(scale)
        if not service_url:
            logger.warning(f"Unknown scale: {scale}")
            continue
        
        # Find wreck layer ID
        layer_id = get_layer_id_by_name(service_url, 'Wreck_point')
        if layer_id is None:
            logger.debug(f"Could not find wreck layer in {scale} service")
            continue
        
        logger.info(f"Querying wrecks from ENC Direct {scale} scale (layer {layer_id})...")
        raw_features = query_arcgis_rest(service_url, layer_id, bounds)
        
        for f in raw_features:
            geom = f.get('geometry', {})
            attrs = f.get('attributes', {})
            
            x = geom.get('x')
            y = geom.get('y')
            
            if x is None or y is None:
                continue
            
            # Deduplicate by position (round to ~10m)
            pos_key = (round(x, 4), round(y, 4))
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)
            
            # Extract depth if available
            depth = attrs.get('valsou') or attrs.get('depth')
            if depth is not None:
                try:
                    depth = float(depth)
                except (ValueError, TypeError):
                    depth = None
            
            features.append(S57Feature(
                object_class='WRECKS',
                geometry_type='POINT',
                x=x,
                y=y,
                depth=depth,
                attributes={
                    'source': f'enc_direct_{scale}',
                    'catwrk': attrs.get('catwrk'),
                    'watlev': attrs.get('watlev'),
                    'quasou': attrs.get('quasou'),
                },
            ))
    
    # Optionally add AWOIS historical wrecks
    if include_awois:
        logger.info("Querying AWOIS historical wrecks...")
        try:
            raw_features = query_arcgis_rest(
                WRECKS_SERVICE_URL, AWOIS_LAYERS['awois_wrecks'], bounds
            )
            
            for f in raw_features:
                geom = f.get('geometry', {})
                attrs = f.get('attributes', {})
                
                x = geom.get('x')
                y = geom.get('y')
                
                if x is None or y is None:
                    continue
                
                pos_key = (round(x, 4), round(y, 4))
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)
                
                depth = attrs.get('depth')
                if depth is not None:
                    try:
                        depth = float(depth)
                    except (ValueError, TypeError):
                        depth = None
                
                features.append(S57Feature(
                    object_class='WRECKS',
                    geometry_type='POINT',
                    x=x,
                    y=y,
                    depth=depth,
                    attributes={
                        'source': 'awois',
                        'vesslterms': attrs.get('vesslterms'),
                        'history': attrs.get('history'),
                    },
                ))
        except Exception as e:
            logger.warning(f"Failed to query AWOIS wrecks: {e}")
    
    logger.info(f"Retrieved {len(features)} unique wrecks")
    return features


def query_obstructions_from_rest(
    bounds: Tuple[float, float, float, float],
    scales: List[str] = None,
    include_awois: bool = False,
) -> List['S57Feature']:
    """
    Query ENC Direct for obstructions within bounds.
    
    Args:
        bounds: (min_x, min_y, max_x, max_y) in WGS84
        scales: ENC scale bands to query (default: ['berthing', 'harbour', 'approach'])
        include_awois: Include historical AWOIS obstructions (from legacy service)
        
    Returns:
        List of S57Feature objects
    """
    if scales is None:
        scales = ['berthing', 'harbour', 'approach']
    
    features = []
    seen_positions = set()
    
    # Query ENC Direct at each scale band
    for scale in scales:
        service_url = ENC_SCALE_SERVICES.get(scale)
        if not service_url:
            logger.warning(f"Unknown scale: {scale}")
            continue
        
        # Find obstruction layer ID
        layer_id = get_layer_id_by_name(service_url, 'Obstruction_point')
        if layer_id is None:
            logger.debug(f"Could not find obstruction layer in {scale} service")
            continue
        
        logger.info(f"Querying obstructions from ENC Direct {scale} scale (layer {layer_id})...")
        raw_features = query_arcgis_rest(service_url, layer_id, bounds)
        
        for f in raw_features:
            geom = f.get('geometry', {})
            attrs = f.get('attributes', {})
            
            x = geom.get('x')
            y = geom.get('y')
            
            if x is None or y is None:
                continue
            
            pos_key = (round(x, 4), round(y, 4))
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)
            
            depth = attrs.get('valsou') or attrs.get('depth')
            if depth is not None:
                try:
                    depth = float(depth)
                except (ValueError, TypeError):
                    depth = None
            
            features.append(S57Feature(
                object_class='OBSTRN',
                geometry_type='POINT',
                x=x,
                y=y,
                depth=depth,
                attributes={
                    'source': f'enc_direct_{scale}',
                    'catobs': attrs.get('catobs'),
                    'watlev': attrs.get('watlev'),
                    'quasou': attrs.get('quasou'),
                },
            ))
    
    # Optionally add AWOIS historical obstructions
    if include_awois:
        logger.info("Querying AWOIS historical obstructions...")
        try:
            raw_features = query_arcgis_rest(
                WRECKS_SERVICE_URL, AWOIS_LAYERS['awois_obstructions'], bounds
            )
            
            for f in raw_features:
                geom = f.get('geometry', {})
                attrs = f.get('attributes', {})
                
                x = geom.get('x')
                y = geom.get('y')
                
                if x is None or y is None:
                    continue
                
                pos_key = (round(x, 4), round(y, 4))
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)
                
                depth = attrs.get('depth')
                if depth is not None:
                    try:
                        depth = float(depth)
                    except (ValueError, TypeError):
                        depth = None
                
                features.append(S57Feature(
                    object_class='OBSTRN',
                    geometry_type='POINT',
                    x=x,
                    y=y,
                    depth=depth,
                    attributes={
                        'source': 'awois',
                        'history': attrs.get('history'),
                    },
                ))
        except Exception as e:
            logger.warning(f"Failed to query AWOIS obstructions: {e}")
    
    logger.info(f"Retrieved {len(features)} unique obstructions")
    return features


def query_rocks_from_rest(
    bounds: Tuple[float, float, float, float],
    scale: str = 'harbour',
) -> List['S57Feature']:
    """
    Query ENC Direct for underwater rocks.
    
    Args:
        bounds: (min_x, min_y, max_x, max_y) in WGS84
        scale: ENC scale band ('berthing', 'harbour', 'approach', 'coastal', 'general')
    """
    features = []
    
    service_url = ENC_SCALE_SERVICES.get(scale)
    if not service_url:
        logger.warning(f"Unknown scale: {scale}")
        return features
    
    # Find rock layer ID
    layer_id = get_layer_id_by_name(service_url, 'Underwater_Awash_Rock')
    if layer_id is None:
        logger.warning(f"Could not find rock layer in {scale} service")
        return features
    
    logger.info(f"Querying rocks from {scale} scale (layer {layer_id})...")
    raw_features = query_arcgis_rest(service_url, layer_id, bounds)
    
    for f in raw_features:
        geom = f.get('geometry', {})
        attrs = f.get('attributes', {})
        
        x = geom.get('x')
        y = geom.get('y')
        
        if x is None or y is None:
            continue
        
        depth = attrs.get('valsou')
        if depth is not None:
            try:
                depth = float(depth)
            except (ValueError, TypeError):
                depth = None
        
        features.append(S57Feature(
            object_class='UWTROC',
            geometry_type='POINT',
            x=x,
            y=y,
            depth=depth,
            attributes={
                'source': f'enc_{scale}',
                'watlev': attrs.get('watlev'),
                'natsur': attrs.get('natsur'),
            },
        ))
    
    logger.info(f"Retrieved {len(features)} rocks from {scale} scale")
    return features


def query_all_features_from_rest(
    bounds: Tuple[float, float, float, float],
    scales: List[str] = None,
    include_awois: bool = False,
) -> List['S57Feature']:
    """
    Query all feature types from NOAA ENC Direct REST API.
    
    Args:
        bounds: (min_x, min_y, max_x, max_y) in WGS84
        scales: ENC scale bands to query (default: ['berthing', 'harbour', 'approach'])
        include_awois: Include historical AWOIS data (from legacy wrecks service)
        
    Returns:
        Combined list of all features
    """
    if scales is None:
        scales = ['berthing', 'harbour', 'approach']
    
    all_features = []
    
    # Wrecks from ENC Direct
    all_features.extend(query_wrecks_from_rest(bounds, scales, include_awois))
    
    # Obstructions from ENC Direct
    all_features.extend(query_obstructions_from_rest(bounds, scales, include_awois))
    
    # Rocks from ENC Direct
    for scale in scales:
        all_features.extend(query_rocks_from_rest(bounds, scale))
    
    logger.info(f"Total features from ENC Direct: {len(all_features)}")
    return all_features


# =============================================================================
# Data Classes and Local File Support
# =============================================================================

# S-57 object classes relevant for bathymetric feature training
FEATURE_CLASSES = {
    'WRECKS': {
        'description': 'Wrecks',
        'label': 1,  # Feature class
        'default_radius': 50,  # meters
    },
    'UWTROC': {
        'description': 'Underwater rocks',
        'label': 1,
        'default_radius': 25,
    },
    'OBSTRN': {
        'description': 'Obstructions',
        'label': 1,
        'default_radius': 30,
    },
    'SBDARE': {
        'description': 'Seabed area',
        'label': None,  # Used for context, not direct labeling
        'default_radius': 0,
    },
    'SOUNDG': {
        'description': 'Soundings',
        'label': None,  # Validation points
        'default_radius': 0,
    },
}

# S-57 attribute codes
ATTRIBUTE_CODES = {
    'CATWRK': 'Category of wreck',
    'VALSOU': 'Value of sounding',
    'WATLEV': 'Water level effect',
    'EXPSOU': 'Exposition of sounding',
    'QUASOU': 'Quality of sounding',
    'NATSUR': 'Nature of surface',
    'NATQUA': 'Nature of surface qualifying terms',
}


@dataclass
class S57Feature:
    """Represents a feature extracted from S-57 ENC or REST API."""
    object_class: str
    geometry_type: str
    x: float
    y: float
    depth: Optional[float] = None
    attributes: Optional[Dict] = None
    wkt: Optional[str] = None  # Full geometry as WKT
    
    def to_dict(self):
        return asdict(self)


def list_s57_layers(enc_path: Path) -> List[str]:
    """List all layers in an S-57 ENC file."""
    ds = ogr.Open(str(enc_path))
    if ds is None:
        raise IOError(f"Could not open S-57 file: {enc_path}")
    
    layers = []
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayerByIndex(i)
        layers.append(layer.GetName())
    
    ds = None
    return layers


def extract_features_from_s57(
    enc_path: Path,
    object_classes: List[str] = None,
    bounds: Tuple[float, float, float, float] = None,
) -> List[S57Feature]:
    """
    Extract features from S-57 ENC file.
    
    Args:
        enc_path: Path to .000 ENC file
        object_classes: List of S-57 object classes to extract (e.g., ['WRECKS', 'UWTROC'])
                       If None, extracts all known feature classes
        bounds: Optional (min_x, min_y, max_x, max_y) to filter spatially
        
    Returns:
        List of S57Feature objects
    """
    if not HAS_GDAL:
        raise ImportError("GDAL/OGR required for S-57 support")
    
    if object_classes is None:
        object_classes = list(FEATURE_CLASSES.keys())
    
    ds = ogr.Open(str(enc_path))
    if ds is None:
        raise IOError(f"Could not open S-57 file: {enc_path}")
    
    features = []
    
    for obj_class in object_classes:
        # S-57 layer names are the object class names
        layer = ds.GetLayerByName(obj_class)
        if layer is None:
            logger.debug(f"Layer {obj_class} not found in {enc_path}")
            continue
        
        # Apply spatial filter if bounds provided
        if bounds:
            layer.SetSpatialFilterRect(bounds[0], bounds[1], bounds[2], bounds[3])
        
        layer.ResetReading()
        
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            
            # Get centroid for point location
            geom_type = geom.GetGeometryName()
            
            if geom_type == 'POINT':
                x, y = geom.GetX(), geom.GetY()
            elif geom_type == 'POINT25D':
                x, y = geom.GetX(), geom.GetY()
            elif geom_type == 'MULTIPOINT' or geom_type == 'MULTIPOINT25D':
                # For soundings, extract each point
                for i in range(geom.GetGeometryCount()):
                    pt = geom.GetGeometryRef(i)
                    x, y = pt.GetX(), pt.GetY()
                    z = pt.GetZ() if pt.GetCoordinateDimension() == 3 else None
                    
                    attrs = _extract_attributes(feature)
                    
                    features.append(S57Feature(
                        object_class=obj_class,
                        geometry_type='POINT',
                        x=x,
                        y=y,
                        depth=z,
                        attributes=attrs,
                    ))
                continue
            else:
                # For polygons/lines, use centroid
                centroid = geom.Centroid()
                x, y = centroid.GetX(), centroid.GetY()
            
            # Extract depth if available (VALSOU attribute or Z coordinate)
            depth = None
            if geom.GetCoordinateDimension() == 3:
                depth = geom.GetZ()
            
            valsou = feature.GetField('VALSOU')
            if valsou is not None:
                depth = float(valsou)
            
            # Extract relevant attributes
            attrs = _extract_attributes(feature)
            
            features.append(S57Feature(
                object_class=obj_class,
                geometry_type=geom_type,
                x=x,
                y=y,
                depth=depth,
                attributes=attrs,
                wkt=geom.ExportToWkt() if geom_type not in ['POINT', 'POINT25D'] else None,
            ))
    
    ds = None
    
    logger.info(f"Extracted {len(features)} features from {enc_path}")
    return features


def _extract_attributes(feature) -> Dict:
    """Extract relevant attributes from S-57 feature."""
    attrs = {}
    
    for attr_code in ATTRIBUTE_CODES.keys():
        val = feature.GetField(attr_code)
        if val is not None:
            attrs[attr_code] = val
    
    # Also get OBJNAM (object name) if present
    objnam = feature.GetField('OBJNAM')
    if objnam:
        attrs['OBJNAM'] = objnam
    
    return attrs if attrs else None


def features_to_geojson(features: List[S57Feature], output_path: Path):
    """Save features as GeoJSON for visualization."""
    geojson = {
        'type': 'FeatureCollection',
        'features': []
    }
    
    for f in features:
        gj_feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [f.x, f.y]
            },
            'properties': {
                'object_class': f.object_class,
                'depth': f.depth,
                **(f.attributes or {})
            }
        }
        geojson['features'].append(gj_feature)
    
    with open(output_path, 'w') as fp:
        json.dump(geojson, fp, indent=2)
    
    logger.info(f"Saved {len(features)} features to {output_path}")


def create_feature_labels_from_s57(
    enc_path: Path,
    survey_path: Path,
    output_path: Path,
    feature_radius: Dict[str, float] = None,
):
    """
    Create feature labels for a survey using S-57 ENC data.
    
    Args:
        enc_path: Path to S-57 ENC file (.000)
        survey_path: Path to bathymetric survey (BAG/GeoTIFF)
        output_path: Output path for labeled GeoTIFF
        feature_radius: Override default radius per feature class (in meters)
    """
    from osgeo import gdal
    
    # Load survey to get extent and resolution
    ds = gdal.OpenEx(str(survey_path), gdal.OF_RASTER, open_options=['MODE=RESAMPLED_GRID'])
    if ds is None:
        ds = gdal.Open(str(survey_path))
    
    if ds is None:
        raise IOError(f"Could not open survey: {survey_path}")
    
    depth = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    crs = ds.GetProjection()
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None
    
    # Calculate bounds
    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + shape[1] * gt[1]
    min_y = max_y + shape[0] * gt[5]
    bounds = (min_x, min_y, max_x, max_y)
    
    resolution = abs(gt[1])
    
    logger.info(f"Survey bounds: {bounds}")
    logger.info(f"Survey resolution: {resolution}m")
    
    # Extract features within survey bounds
    features = extract_features_from_s57(
        enc_path,
        object_classes=['WRECKS', 'UWTROC', 'OBSTRN'],
        bounds=bounds,
    )
    
    logger.info(f"Found {len(features)} features within survey bounds")
    
    # Initialize labels
    nodata = 1.0e6
    valid = (depth != nodata) & np.isfinite(depth)
    labels = np.zeros(shape, dtype=np.int32)
    labels[~valid] = -1
    
    # Apply feature labels
    feature_count = 0
    
    for f in features:
        obj_class = f.object_class
        
        # Get radius for this feature class
        if feature_radius and obj_class in feature_radius:
            radius_m = feature_radius[obj_class]
        else:
            radius_m = FEATURE_CLASSES.get(obj_class, {}).get('default_radius', 30)
        
        # Convert radius to pixels
        radius_px = int(np.ceil(radius_m / resolution))
        
        # Convert geographic coordinates to pixel coordinates
        col = int((f.x - gt[0]) / gt[1])
        row = int((f.y - gt[3]) / gt[5])
        
        # Check bounds
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            continue
        
        # Create circular mask
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((xx - col)**2 + (yy - row)**2)
        feature_mask = dist <= radius_px
        
        # Apply label
        label_value = FEATURE_CLASSES.get(obj_class, {}).get('label', 1)
        if label_value is not None:
            labels[feature_mask & valid] = label_value
            feature_count += 1
    
    logger.info(f"Applied {feature_count} feature labels")
    logger.info(f"Feature cells: {np.sum(labels == 1):,}")
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        str(output_path),
        shape[1], shape[0], 2,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    
    out_ds.SetGeoTransform(gt)
    if crs:
        out_ds.SetProjection(crs)
    
    out_ds.GetRasterBand(1).WriteArray(labels.astype(np.float32))
    out_ds.GetRasterBand(1).SetDescription('labels')
    out_ds.GetRasterBand(1).SetNoDataValue(-1)
    
    out_ds.GetRasterBand(2).WriteArray(depth)
    out_ds.GetRasterBand(2).SetDescription('depth')
    
    out_ds.FlushCache()
    out_ds = None
    
    logger.info(f"Saved: {output_path}")


def get_survey_bounds(survey_path: Path) -> Tuple[float, float, float, float]:
    """Extract geographic bounds from survey file."""
    from osgeo import gdal
    
    gdal.UseExceptions()
    
    # Try VR BAG resampled mode first
    ds = gdal.OpenEx(str(survey_path), gdal.OF_RASTER, open_options=['MODE=RESAMPLED_GRID'])
    if ds is None:
        ds = gdal.Open(str(survey_path))
    
    if ds is None:
        raise IOError(f"Could not open survey: {survey_path}")
    
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None
    
    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + width * gt[1]
    min_y = max_y + height * gt[5]
    
    return (min_x, min_y, max_x, max_y)


def create_feature_labels(
    features: List[S57Feature],
    survey_path: Path,
    output_path: Path,
    feature_radius: Dict[str, float] = None,
):
    """
    Create feature labels for a survey using extracted features.
    
    Args:
        features: List of S57Feature objects
        survey_path: Path to bathymetric survey (BAG/GeoTIFF)
        output_path: Output path for labeled GeoTIFF
        feature_radius: Override default radius per feature class (in meters)
    """
    from osgeo import gdal
    
    if not features:
        logger.warning("No features to label")
        return
    
    # Load survey
    gdal.UseExceptions()
    ds = gdal.OpenEx(str(survey_path), gdal.OF_RASTER, open_options=['MODE=RESAMPLED_GRID'])
    if ds is None:
        ds = gdal.Open(str(survey_path))
    
    if ds is None:
        raise IOError(f"Could not open survey: {survey_path}")
    
    depth = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    crs = ds.GetProjection()
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None
    
    resolution = abs(gt[1])
    
    logger.info(f"Survey shape: {shape}, resolution: {resolution}m")
    
    # Initialize labels
    nodata = 1.0e6
    valid = (depth != nodata) & np.isfinite(depth)
    labels = np.zeros(shape, dtype=np.int32)
    labels[~valid] = -1
    
    # Apply feature labels
    feature_count = 0
    
    for f in features:
        obj_class = f.object_class
        
        # Get radius for this feature class
        if feature_radius and obj_class in feature_radius:
            radius_m = feature_radius[obj_class]
        else:
            radius_m = FEATURE_CLASSES.get(obj_class, {}).get('default_radius', 30)
        
        # Convert radius to pixels
        radius_px = int(np.ceil(radius_m / resolution))
        
        # Convert geographic coordinates to pixel coordinates
        col = int((f.x - gt[0]) / gt[1])
        row = int((f.y - gt[3]) / gt[5])
        
        # Check bounds
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            continue
        
        # Create circular mask
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((xx - col)**2 + (yy - row)**2)
        feature_mask = dist <= radius_px
        
        # Apply label
        label_value = FEATURE_CLASSES.get(obj_class, {}).get('label', 1)
        if label_value is not None:
            labels[feature_mask & valid] = label_value
            feature_count += 1
    
    logger.info(f"Applied {feature_count} feature labels")
    logger.info(f"Feature cells: {np.sum(labels == 1):,}")
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        str(output_path),
        shape[1], shape[0], 2,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    
    out_ds.SetGeoTransform(gt)
    if crs:
        out_ds.SetProjection(crs)
    
    out_ds.GetRasterBand(1).WriteArray(labels.astype(np.float32))
    out_ds.GetRasterBand(1).SetDescription('labels')
    out_ds.GetRasterBand(1).SetNoDataValue(-1)
    
    out_ds.GetRasterBand(2).WriteArray(depth)
    out_ds.GetRasterBand(2).SetDescription('depth')
    
    out_ds.FlushCache()
    out_ds = None
    
    logger.info(f"Saved: {output_path}")
    
    return features


def summarize_enc(enc_path: Path):
    """Print summary of S-57 ENC contents."""
    print(f"\n{'='*60}")
    print(f"S-57 ENC Summary: {enc_path.name}")
    print(f"{'='*60}\n")
    
    layers = list_s57_layers(enc_path)
    print(f"Total layers: {len(layers)}\n")
    
    # Count features per relevant layer
    print("Feature counts:")
    print("-" * 40)
    
    ds = ogr.Open(str(enc_path))
    
    for obj_class in FEATURE_CLASSES.keys():
        layer = ds.GetLayerByName(obj_class)
        if layer:
            count = layer.GetFeatureCount()
            desc = FEATURE_CLASSES[obj_class]['description']
            print(f"  {obj_class:8} ({desc:20}): {count:,}")
    
    ds = None
    
    print("\nAll layers:")
    print("-" * 40)
    for layer in sorted(layers):
        print(f"  {layer}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from NOAA ENC data (REST API or local files)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query NOAA REST API for features within survey bounds (RECOMMENDED)
  python extract_s57_features.py --survey survey.bag --labels labels.tif
  
  # Query REST API with custom bounds
  python extract_s57_features.py --bounds -122.5 37.5 -122.0 38.0 --output features.geojson
  
  # Use local ENC file instead of REST API
  python extract_s57_features.py --enc US5AK1AM.000 --survey survey.bag --labels labels.tif
  
  # List contents of local ENC file
  python extract_s57_features.py --enc US5AK1AM.000 --summarize

REST API endpoints (no download required):
  - Wrecks: https://wrecks.nauticalcharts.noaa.gov/arcgis/rest/services/public_wrecks/
  - ENC: https://encdirect.noaa.gov/arcgis/rest/services/encdirect/

Download local ENCs from: https://encdirect.noaa.gov/
        """
    )
    
    # Input sources (mutually exclusive for features)
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument('--enc', type=Path, help='Local S-57 ENC file (.000)')
    input_group.add_argument('--enc-dir', type=Path, help='Directory containing local ENC files')
    input_group.add_argument('--bounds', type=float, nargs=4, 
                             metavar=('MIN_X', 'MIN_Y', 'MAX_X', 'MAX_Y'),
                             help='Query bounds in WGS84 (for REST API mode)')
    input_group.add_argument('--survey', type=Path, 
                             help='Bathymetric survey - bounds extracted automatically')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', type=Path, help='Output GeoJSON for extracted features')
    output_group.add_argument('--labels', type=Path, help='Output GeoTIFF with feature labels')
    output_group.add_argument('--summarize', action='store_true', 
                              help='Print summary of ENC contents (local files only)')
    
    # Feature options
    feature_group = parser.add_argument_group('Feature Options')
    feature_group.add_argument('--wreck-radius', type=float, default=50, 
                               help='Radius around wrecks (meters)')
    feature_group.add_argument('--rock-radius', type=float, default=25, 
                               help='Radius around rocks (meters)')
    feature_group.add_argument('--obstruction-radius', type=float, default=30, 
                               help='Radius around obstructions (meters)')
    feature_group.add_argument('--awois', action='store_true',
                               help='Include historical AWOIS data (from legacy wrecks service)')
    feature_group.add_argument('--scales', nargs='+', 
                               default=['berthing', 'harbour', 'approach'],
                               choices=['berthing', 'harbour', 'approach', 'coastal', 'general'],
                               help='ENC scale bands to query (default: berthing, harbour, approach)')
    
    args = parser.parse_args()
    
    # Determine bounds
    bounds = None
    if args.bounds:
        bounds = tuple(args.bounds)
    elif args.survey:
        bounds = get_survey_bounds(args.survey)
        logger.info(f"Survey bounds: {bounds}")
    
    # Collect features
    all_features = []
    
    # Mode 1: Local ENC files
    enc_files = []
    if args.enc:
        enc_files.append(args.enc)
    if args.enc_dir:
        enc_files.extend(args.enc_dir.glob('*.000'))
    
    if enc_files:
        if not HAS_GDAL:
            logger.error("GDAL/OGR is required for local S-57 files")
            return 1
        
        for enc_path in enc_files:
            if args.summarize:
                summarize_enc(enc_path)
            
            features = extract_features_from_s57(enc_path, bounds=bounds)
            all_features.extend(features)
            
            # Print summary
            by_class = {}
            for f in features:
                by_class[f.object_class] = by_class.get(f.object_class, 0) + 1
            
            for obj_class, count in sorted(by_class.items()):
                logger.info(f"  {obj_class}: {count}")
    
    # Mode 2: REST API (if no local files and bounds available)
    elif bounds:
        logger.info("Querying NOAA ENC Direct REST API...")
        all_features = query_all_features_from_rest(
            bounds,
            scales=args.scales,
            include_awois=args.awois,
        )
    
    else:
        parser.error("Must specify --enc, --enc-dir, --bounds, or --survey")
    
    # Print summary
    if all_features:
        by_class = {}
        for f in all_features:
            by_class[f.object_class] = by_class.get(f.object_class, 0) + 1
        
        print(f"\nFeature Summary:")
        print("-" * 40)
        for obj_class, count in sorted(by_class.items()):
            desc = FEATURE_CLASSES.get(obj_class, {}).get('description', obj_class)
            print(f"  {obj_class:8} ({desc:20}): {count:,}")
        print(f"  {'TOTAL':8} {'':20}  {len(all_features):,}")
    
    # Save GeoJSON
    if args.output:
        features_to_geojson(all_features, args.output)
    
    # Create labels for survey
    if args.labels and args.survey:
        feature_radius = {
            'WRECKS': args.wreck_radius,
            'UWTROC': args.rock_radius,
            'OBSTRN': args.obstruction_radius,
        }
        
        create_feature_labels(
            all_features,
            args.survey,
            args.labels,
            feature_radius,
        )
    
    return 0


if __name__ == '__main__':
    exit(main())

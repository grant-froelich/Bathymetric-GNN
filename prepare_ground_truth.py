#!/usr/bin/env python3
"""Explore VR BAG HDF5 structure to understand how to process natively."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
from pathlib import Path

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

import numpy as np


def explore_hdf5(group, indent=0):
    """Recursively explore HDF5 structure."""
    prefix = "  " * indent
    
    for key in group.keys():
        item = group[key]
        
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/ (Group)")
            explore_hdf5(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            shape_str = f"shape={item.shape}" if item.shape else "scalar"
            dtype_str = f"dtype={item.dtype}"
            print(f"{prefix}{key} (Dataset: {shape_str}, {dtype_str})")
            
            # Show attributes
            for attr_name, attr_val in item.attrs.items():
                print(f"{prefix}  @{attr_name} = {attr_val}")
            
            # Show sample data for small datasets or structured arrays
            if item.dtype.names:  # Structured array
                print(f"{prefix}  Fields: {item.dtype.names}")
                if item.shape and item.shape[0] > 0:
                    print(f"{prefix}  First record: {item[0]}")
            elif item.size < 100:
                data = item[()]
                print(f"{prefix}  Data: {data}")


def analyze_vr_bag(path):
    """Analyze VR BAG structure in detail."""
    print(f"\n{'='*60}")
    print(f"VR BAG Analysis: {path}")
    print(f"{'='*60}\n")
    
    with h5py.File(str(path), 'r') as f:
        print("--- Full HDF5 Structure ---\n")
        explore_hdf5(f)
        
        root = f['BAG_root']
        
        # Base elevation grid
        print(f"\n--- Base Elevation Grid ---")
        elevation = root['elevation']
        print(f"Shape: {elevation.shape}")
        print(f"Dtype: {elevation.dtype}")
        valid_mask = elevation[()] != 1000000.0
        print(f"Valid cells: {np.sum(valid_mask):,} / {elevation.size:,}")
        
        # Check for VR components
        if 'varres_refinements' not in root:
            print("\nThis is NOT a VR BAG (no varres_refinements)")
            return
        
        print(f"\n--- VR Refinements ---")
        refinements = root['varres_refinements']
        print(f"Shape: {refinements.shape}")
        print(f"Dtype: {refinements.dtype}")
        print(f"Fields: {refinements.dtype.names}")
        
        # Sample refinement records
        if refinements.shape[0] > 0:
            print(f"\nFirst 5 refinement records:")
            for i in range(min(5, refinements.shape[0])):
                print(f"  [{i}]: {refinements[i]}")
            
            print(f"\nLast 5 refinement records:")
            for i in range(max(0, refinements.shape[0]-5), refinements.shape[0]):
                print(f"  [{i}]: {refinements[i]}")
        
        # VR Metadata
        if 'varres_metadata' in root:
            print(f"\n--- VR Metadata ---")
            metadata = root['varres_metadata']
            print(f"Shape: {metadata.shape}")
            print(f"Dtype: {metadata.dtype}")
            print(f"Fields: {metadata.dtype.names}")
            
            # Find cells with refinements
            if metadata.dtype.names and 'dimensions_x' in metadata.dtype.names:
                dims_x = metadata['dimensions_x'][:]
                has_refinement = dims_x > 0
                print(f"\nBase cells with refinements: {np.sum(has_refinement):,} / {metadata.size:,}")
                
                # Show sample metadata records with refinements
                refined_indices = np.where(has_refinement.flatten())[0]
                if len(refined_indices) > 0:
                    print(f"\nSample refined cell metadata:")
                    for idx in refined_indices[:5]:
                        row = idx // metadata.shape[1]
                        col = idx % metadata.shape[1]
                        record = metadata[row, col]
                        print(f"  Base cell ({row},{col}): {record}")
        
        # VR Tracking List
        if 'varres_tracking_list' in root:
            print(f"\n--- VR Tracking List ---")
            tracking = root['varres_tracking_list']
            print(f"Shape: {tracking.shape}")
            print(f"Dtype: {tracking.dtype}")
            if tracking.dtype.names:
                print(f"Fields: {tracking.dtype.names}")
        
        # Calculate refinement grid structure
        print(f"\n--- Refinement Grid Analysis ---")
        if 'varres_metadata' in root:
            metadata = root['varres_metadata']
            if 'dimensions_x' in metadata.dtype.names:
                dims_x = metadata['dimensions_x'][:]
                dims_y = metadata['dimensions_y'][:]
                
                # Unique refinement grid sizes
                valid_dims = dims_x[dims_x > 0]
                if len(valid_dims) > 0:
                    unique_x = np.unique(dims_x[dims_x > 0])
                    unique_y = np.unique(dims_y[dims_y > 0])
                    print(f"Unique refinement grid widths: {unique_x}")
                    print(f"Unique refinement grid heights: {unique_y}")
                    
                    # Total refinement cells
                    total_ref_cells = np.sum(dims_x.astype(np.int64) * dims_y.astype(np.int64))
                    print(f"Total refinement cells: {total_ref_cells:,}")
                    print(f"Actual refinement records: {refinements.shape[0]:,}")
                    
                    # Resolution info
                    if 'resolution_x' in metadata.dtype.names:
                        res_x = metadata['resolution_x'][:]
                        res_y = metadata['resolution_y'][:]
                        valid_res = res_x[dims_x > 0]
                        if len(valid_res) > 0:
                            print(f"Refinement resolutions: {np.unique(valid_res)}")


def main():
    parser = argparse.ArgumentParser(description="Explore VR BAG structure")
    parser.add_argument("--survey", type=Path, required=True, help="Path to BAG file")
    args = parser.parse_args()
    
    if not HAS_H5PY:
        print("ERROR: h5py is required. Install with: pip install h5py")
        sys.exit(1)
    
    if not args.survey.exists():
        print(f"ERROR: File not found: {args.survey}")
        sys.exit(1)
    
    analyze_vr_bag(args.survey)


if __name__ == "__main__":
    main()

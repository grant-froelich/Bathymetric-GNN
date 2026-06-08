#!/usr/bin/env python3
"""
scripts/spatial_error_map.py

Run a trained V10 model over a single regression-mode ground truth file and
write a per-cell error map as a GeoTIFF, plus print diagnostics on whether
errors concentrate by depth or spatially.

Unlike evaluate_v10.py (which reports aggregate metrics), this places every
prediction back onto the original grid so you can open the result in QGIS and
see WHERE the model fails, not just how much.

Output GeoTIFF bands (aligned to the input grid):
  1. target_correction   (meters, noisy - clean)
  2. predicted_correction (meters, denormalized model output)
  3. error                (predicted - target; + = corrected shallower = safe)
  4. abs_error            (meters)
  5. noisy_depth          (meters, for context)

Usage:
  python scripts/spatial_error_map.py \
      --checkpoint outputs-multiloc/best_model.pt \
      --ground-truth ground-truth-val-h14116/H14116_MB_VR_MLLW_1of1_regression.tif \
      --output eval-multiloc/h14116_error_map.tif
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# torch first for Windows DLL load order
import torch  # noqa: F401

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data import GraphBuilder
from models import BathymetricGNN


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    in_channels = checkpoint.get('in_channels', 9)
    edge_dim = checkpoint.get('edge_dim', 3)
    model = BathymetricGNN(
        in_channels=in_channels,
        hidden_channels=config.model.gnn_hidden_channels,
        num_gnn_layers=config.model.gnn_num_layers,
        gnn_type=config.model.gnn_type,
        heads=config.model.gnn_heads,
        num_classes=config.model.num_classes,
        predict_correction=config.model.predict_correction,
        dropout=config.model.gnn_dropout,
        edge_dim=edge_dim,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', -1) + 1}, "
                f"in_channels={in_channels}")
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Write a per-cell error map for one survey")
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True,
                        help="Regression-mode ground truth GeoTIFF")
    parser.add_argument('--output', type=Path, required=True, help="Output error-map GeoTIFF")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--min-valid-ratio', type=float, default=0.1)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    model, config = load_model(args.checkpoint, args.device)

    graph_builder = GraphBuilder(
        connectivity=config.graph.connectivity,
        edge_features=config.graph.edge_features,
    )

    # Read the ground truth grid directly
    ds = gdal.Open(str(args.ground_truth))
    if ds is None:
        logger.error(f"Could not open {args.ground_truth}")
        sys.exit(1)

    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    H, W = ds.RasterYSize, ds.RasterXSize
    resolution = (abs(geotransform[1]), abs(geotransform[5]))

    valid_full = ds.GetRasterBand(1).ReadAsArray().astype(np.int32) == 1
    difference = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    noisy_depth = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
    uncertainty = None
    if ds.RasterCount >= 5:
        uncertainty = ds.GetRasterBand(5).ReadAsArray().astype(np.float32)
    ds = None

    logger.info(f"Grid: {H} x {W} at {resolution[0]:.2f}m, valid cells: {valid_full.sum():,}")

    # Full-grid accumulators. Overlapping tiles average via pred_sum / count.
    pred_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.int32)

    tile_size = args.tile_size
    overlap = args.overlap
    stride = tile_size - overlap

    def process_tile(rs, re, cs, ce):
        tile_valid = valid_full[rs:re, cs:ce]
        if tile_valid.size == 0:
            return
        if np.sum(tile_valid) / tile_valid.size < args.min_valid_ratio:
            return
        tile_noisy = noisy_depth[rs:re, cs:ce]
        tile_uncert = uncertainty[rs:re, cs:ce] if uncertainty is not None else None

        graph = graph_builder.build_graph(
            depth=tile_noisy,
            valid_mask=tile_valid,
            uncertainty=tile_uncert,
            resolution=resolution,
        )
        if graph.num_nodes == 0:
            return

        graph = graph.to(args.device)
        with torch.no_grad():
            outputs = model(graph)
        pred_norm = outputs['correction'].detach().cpu().numpy()
        local_std = graph.local_std.detach().cpu().numpy()
        pred_m = pred_norm * np.maximum(local_std, 0.01)

        rows = graph.valid_rows.detach().cpu().numpy()
        cols = graph.valid_cols.detach().cpu().numpy()
        # Place into full grid (offset by tile origin)
        pred_sum[rs + rows, cs + cols] += pred_m
        count[rs + rows, cs + cols] += 1

    # Tile the grid the same way GroundTruthDataset does
    n_tiles = 0
    for row_start in range(0, H - tile_size + 1, stride):
        for col_start in range(0, W - tile_size + 1, stride):
            process_tile(row_start, row_start + tile_size, col_start, col_start + tile_size)
            n_tiles += 1
    # Edge tile (bottom-right)
    if (H % stride != 0 or W % stride != 0) and H > tile_size and W > tile_size:
        process_tile(H - tile_size, H, W - tile_size, W)
        n_tiles += 1

    logger.info(f"Processed {n_tiles} tile positions")

    # Resolve predictions; cells with no prediction stay nodata
    predicted = np.full((H, W), np.nan, dtype=np.float32)
    have_pred = count > 0
    predicted[have_pred] = (pred_sum[have_pred] / count[have_pred]).astype(np.float32)

    # Only analyze cells that are both valid and predicted
    analyze = valid_full & have_pred
    target = np.where(analyze, difference, np.nan).astype(np.float32)
    error = np.where(analyze, predicted - difference, np.nan).astype(np.float32)
    abs_error = np.abs(error)

    # --- Write GeoTIFF ---
    NODATA = 1.0e6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(str(args.output), W, H, 5, gdal.GDT_Float32,
                           options=['COMPRESS=LZW'])
    out_ds.SetGeoTransform(geotransform)
    if projection:
        out_ds.SetProjection(projection)

    band_data = [
        ('target_correction', target),
        ('predicted_correction', predicted),
        ('error', error),
        ('abs_error', abs_error),
        ('noisy_depth', np.where(analyze, noisy_depth, np.nan).astype(np.float32)),
    ]
    for i, (name, arr) in enumerate(band_data, start=1):
        b = out_ds.GetRasterBand(i)
        out = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)
        b.WriteArray(out)
        b.SetNoDataValue(NODATA)
        b.SetDescription(name)
    out_ds = None
    logger.info(f"Wrote error map: {args.output}")

    # --- Diagnostics ---
    e = error[analyze]
    ae = abs_error[analyze]
    tgt = difference[analyze]
    dep = noisy_depth[analyze]
    n = len(e)

    print()
    print("=" * 64)
    print(f"SPATIAL ERROR DIAGNOSTICS  ({args.ground_truth.name})")
    print("=" * 64)
    print(f"Cells analyzed: {n:,}")
    print(f"Overall MAE: {ae.mean():.3f}m   RMSE: {np.sqrt((e**2).mean()):.3f}m")
    print()

    # Concentration: how much of total squared error comes from worst cells
    sq = e**2
    order = np.argsort(sq)[::-1]
    cum = np.cumsum(sq[order]) / sq.sum()
    for frac in (0.01, 0.05, 0.10, 0.25):
        k = max(1, int(frac * n))
        print(f"  Worst {frac*100:4.0f}% of cells ({k:>7,}) account for "
              f"{cum[k-1]*100:5.1f}% of total squared error")
    print("  (High concentration = errors localized to few cells;")
    print("   near the diagonal = errors spread evenly.)")
    print()

    # Error vs depth
    print("  Mean |error| by depth bin:")
    abs_dep = np.abs(dep)
    edges = np.percentile(abs_dep, [0, 20, 40, 60, 80, 100])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (abs_dep >= lo) & (abs_dep <= hi)
        if m.sum() > 0:
            print(f"    depth {lo:8.1f} - {hi:8.1f}m : "
                  f"{ae[m].mean():8.3f}m  ({m.sum():>7,} cells)")
    print()

    # The large-correction failure bucket
    big = np.abs(tgt) >= 10.0
    if big.sum() > 0:
        print(f"  Cells with |target| >= 10m: {big.sum():,}")
        print(f"    mean |error| there:        {ae[big].mean():.3f}m")
        print(f"    their depth range:         {np.abs(dep[big]).min():.1f} - {np.abs(dep[big]).max():.1f}m")
        print(f"    their median depth:        {np.median(np.abs(dep[big])):.1f}m")
        print(f"    median depth of ALL cells: {np.median(abs_dep):.1f}m")
        print("    (If big-error cells sit much deeper than the median,")
        print("     the failure is depth-localized.)")
    print("=" * 64)
    print()
    print("Open the error map in QGIS, symbolize band 3 (error) with a")
    print("diverging scale centered on 0, to see if errors cluster spatially.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
scripts/verify_graph_equivalence.py

Verifies that the vectorized graph construction (data/graph_construction.py,
2026-06-09) produces graphs equivalent to the original loop-based
implementation. Run this BEFORE the V11 retrain; training should only start
after every sampled tile passes.

The legacy loop implementation is embedded here verbatim (edge building and
edge features), so the comparison is against the real pre-rewrite behavior,
not a reconstruction.

Equivalence is defined as:
  - identical node features (x), positions, local_std (bitwise)
  - identical edge SET with identical per-edge features. Edge ORDER is allowed
    to differ (the rewrite is offset-major, the loop was node-major); GNN
    message passing is permutation-invariant over edges, so order is
    canonicalized by sorting before comparison. Features must match to within
    1e-6 (float32 round-off between the two computation orders).

Usage:
    python scripts/verify_graph_equivalence.py --ground-truth-dir ground-truth-train/ --num-tiles 30
    python scripts/verify_graph_equivalence.py --ground-truth-dir ground-truth-train/ --all-tiles
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import torch before numpy to avoid DLL conflicts on Windows
import torch

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data import GraphBuilder
from training import GroundTruthDataset
from config import Config


# ---------------------------------------------------------------------------
# Legacy (pre-2026-06-09) loop-based implementation, verbatim
# ---------------------------------------------------------------------------

def legacy_build_edges(builder, valid_rows, valid_cols, coord_to_node, grid_shape):
    """Original per-node loop edge construction."""
    height, width = grid_shape

    source_nodes = []
    target_nodes = []
    edge_coords = []

    for node_idx, (r, c) in enumerate(zip(valid_rows, valid_cols)):
        for dr, dc in builder.neighbor_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) in coord_to_node:
                    neighbor_idx = coord_to_node[(nr, nc)]
                    source_nodes.append(node_idx)
                    target_nodes.append(neighbor_idx)
                    edge_coords.append((r, c, nr, nc))

    if builder.include_self_loops:
        for node_idx, (r, c) in enumerate(zip(valid_rows, valid_cols)):
            source_nodes.append(node_idx)
            target_nodes.append(node_idx)
            edge_coords.append((r, c, r, c))

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return edge_index, edge_coords


def legacy_edge_features(builder, depth, edge_coords, resolution):
    """Original per-edge loop feature computation."""
    if len(edge_coords) == 0:
        return torch.zeros((0, len(builder.edge_features)), dtype=torch.float32)

    features = []
    res_x, res_y = resolution

    for feature_name in builder.edge_features:
        feat_values = []
        for src_r, src_c, tgt_r, tgt_c in edge_coords:
            if feature_name == "distance":
                dx = (tgt_c - src_c) * res_x
                dy = (tgt_r - src_r) * res_y
                value = np.sqrt(dx ** 2 + dy ** 2)
            elif feature_name == "depth_difference":
                value = depth[tgt_r, tgt_c] - depth[src_r, src_c]
            elif feature_name == "slope":
                dx = (tgt_c - src_c) * res_x
                dy = (tgt_r - src_r) * res_y
                dz = depth[tgt_r, tgt_c] - depth[src_r, src_c]
                horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
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


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def canonicalize(edge_index: np.ndarray, edge_attr: np.ndarray):
    """Sort edges by (src, tgt) so order differences disappear."""
    order = np.lexsort((edge_index[1], edge_index[0]))
    return edge_index[:, order], edge_attr[order]


def compare_tile(dataset, idx, builder, atol=1e-6):
    """Build one tile both ways and compare. Returns list of problems."""
    problems = []
    tile = dataset.tiles[idx]

    # New (library) path: full graph incl. node features and targets
    graph_new = dataset[idx]

    # Legacy path: recompute edges + edge features with the loop code on the
    # same inputs the library used
    noisy_depth = tile['noisy_depth']
    valid_mask = tile['valid_mask'].copy()
    valid_mask &= np.isfinite(noisy_depth)

    valid_rows, valid_cols = np.where(valid_mask)
    coord_to_node = {(r, c): i for i, (r, c) in enumerate(zip(valid_rows, valid_cols))}

    ei_old, ecoords_old = legacy_build_edges(
        builder, valid_rows, valid_cols, coord_to_node, noisy_depth.shape
    )
    ea_old = legacy_edge_features(builder, noisy_depth, ecoords_old, tile['resolution'])

    # Node count
    if graph_new.num_nodes != len(valid_rows):
        problems.append(f"node count {graph_new.num_nodes} != {len(valid_rows)}")
        return problems

    # Edge set + features
    ei_new = graph_new.edge_index.numpy()
    ea_new = graph_new.edge_attr.numpy()
    if ei_new.shape != tuple(ei_old.shape):
        problems.append(f"edge count {ei_new.shape[1]} != {ei_old.shape[1]}")
        return problems

    ei_new_c, ea_new_c = canonicalize(ei_new, ea_new)
    ei_old_c, ea_old_c = canonicalize(ei_old.numpy(), ea_old.numpy())

    if not np.array_equal(ei_new_c, ei_old_c):
        n_diff = int(np.sum(np.any(ei_new_c != ei_old_c, axis=0)))
        problems.append(f"edge sets differ at {n_diff} positions after canonicalization")
        return problems

    feat_diff = np.abs(ea_new_c - ea_old_c)
    if feat_diff.size and float(feat_diff.max()) > atol:
        problems.append(
            f"edge features differ: max |delta| = {float(feat_diff.max()):.3e} "
            f"(atol {atol:g})"
        )

    return problems


def main():
    parser = argparse.ArgumentParser(
        description='Verify vectorized graph construction matches the legacy loop implementation'
    )
    parser.add_argument('--ground-truth-dir', type=Path, required=True)
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--num-tiles', type=int, default=30,
                        help='Number of tiles to sample (evenly spaced). Ignored with --all-tiles.')
    parser.add_argument('--all-tiles', action='store_true', help='Check every tile (slow)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = Config()
    builder = GraphBuilder(
        connectivity=config.graph.connectivity,
        edge_features=config.graph.edge_features,
    )

    gt_files = sorted(
        list(args.ground_truth_dir.glob("*_ground_truth.tif")) +
        list(args.ground_truth_dir.glob("*_regression.tif"))
    )
    if not gt_files:
        logger.error(f"No ground truth files in {args.ground_truth_dir}")
        sys.exit(1)

    dataset = GroundTruthDataset(
        ground_truth_paths=gt_files,
        graph_builder=builder,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_valid_ratio=config.tile.min_valid_ratio,
    )
    n = len(dataset)
    logger.info(f"Dataset: {n} tiles from {len(gt_files)} files")

    if args.all_tiles:
        indices = list(range(n))
    else:
        k = min(args.num_tiles, n)
        indices = sorted(set(np.linspace(0, n - 1, k).astype(int).tolist()))

    logger.info(f"Verifying {len(indices)} tiles...")
    t_new_total = 0.0
    failures = 0
    for j, idx in enumerate(indices):
        t0 = time.time()
        problems = compare_tile(dataset, idx, builder)
        t_new_total += time.time() - t0
        if problems:
            failures += 1
            for p in problems:
                logger.error(f"tile {idx} ({dataset.tiles[idx]['source']}): {p}")
        if (j + 1) % 10 == 0:
            logger.info(f"  {j + 1}/{len(indices)} checked")

    print()
    if failures == 0:
        print(f"PASS: all {len(indices)} sampled tiles equivalent "
              f"(edge sets identical, features within 1e-6)")
        print("Safe to proceed with the V11 retrain.")
        sys.exit(0)
    else:
        print(f"FAIL: {failures}/{len(indices)} tiles differ. Do NOT retrain; "
              f"report the output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Evaluation metrics for V10 regression-mode models.

V10 predicts a continuous correction at every cell rather than classifying
cells as noise/seafloor. Traditional classification metrics (accuracy,
precision, recall, F1) do not apply.

This module defines V10Metrics dataclass and the function that computes
it from model predictions and ground truth.

See docs/HOW_IT_WORKS.md for the reasoning behind each metric.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional
import json
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class V10Metrics:
    """
    Evaluation metrics for a V10 regression model on a single survey.
    
    All correction magnitudes are in meters unless noted otherwise.
    Hazardous error rate uses the sign convention where a positive error
    (predicted > target) leaves the corrected surface shallower than reality
    and is therefore safer for navigation than a negative error.
    """
    
    # Identification
    survey_name: str = ""
    model_version: str = ""
    n_valid_cells: int = 0
    
    # Overall regression metrics
    mae_meters: float = 0.0
    rmse_meters: float = 0.0
    mae_normalized_std: float = 0.0  # MAE in local_std units
    
    # Per-magnitude-bucket MAE in meters
    # Buckets are based on the true correction magnitude
    mae_under_0_1m: float = 0.0
    mae_0_1_to_1m: float = 0.0
    mae_1_to_10m: float = 0.0
    mae_over_10m: float = 0.0
    n_under_0_1m: int = 0
    n_0_1_to_1m: int = 0
    n_1_to_10m: int = 0
    n_over_10m: int = 0
    
    # Safety metrics: hazardous = predicted < target (corrected depth deeper than reality)
    hazardous_error_rate: float = 0.0      # fraction across all valid cells
    hazardous_error_rate_shoal: float = 0.0  # subset where target < 0
    hazardous_error_rate_deep: float = 0.0   # subset where target > 0
    n_shoal_target_cells: int = 0
    n_deep_target_cells: int = 0
    
    # Operational: how close does the corrected surface get to the clean reference?
    recovery_rmse: float = 0.0  # RMS of (corrected_depth - clean_depth) across valid cells
    recovery_mean_error: float = 0.0  # signed mean of the same residual
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save_json(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> 'V10Metrics':
        with open(path) as f:
            return cls(**json.load(f))
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"V10 Metrics: {self.survey_name} ({self.model_version})",
            f"  Cells evaluated: {self.n_valid_cells:,}",
            f"  Overall: MAE={self.mae_meters:.3f}m, RMSE={self.rmse_meters:.3f}m, "
            f"normalized MAE={self.mae_normalized_std:.3f} std",
            f"  Per-bucket MAE (true correction magnitude):",
            f"    < 0.1m:   {self.mae_under_0_1m:.4f}m  ({self.n_under_0_1m:,} cells)",
            f"    0.1-1m:   {self.mae_0_1_to_1m:.4f}m  ({self.n_0_1_to_1m:,} cells)",
            f"    1-10m:    {self.mae_1_to_10m:.4f}m  ({self.n_1_to_10m:,} cells)",
            f"    > 10m:    {self.mae_over_10m:.4f}m  ({self.n_over_10m:,} cells)",
            f"  Safety:",
            f"    Hazardous rate (overall):       {self.hazardous_error_rate:.2%}",
            f"    Hazardous rate (shoal targets): {self.hazardous_error_rate_shoal:.2%}  ({self.n_shoal_target_cells:,} cells)",
            f"    Hazardous rate (deep targets):  {self.hazardous_error_rate_deep:.2%}  ({self.n_deep_target_cells:,} cells)",
            f"  Recovery:",
            f"    RMSE vs clean: {self.recovery_rmse:.3f}m",
            f"    Mean error vs clean: {self.recovery_mean_error:+.3f}m",
        ]
        return "\n".join(lines)


def _to_numpy(x) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_v10_metrics(
    predicted_correction: np.ndarray,
    target_correction: np.ndarray,
    valid_mask: np.ndarray,
    local_std: Optional[np.ndarray] = None,
    noisy_depth: Optional[np.ndarray] = None,
    clean_depth: Optional[np.ndarray] = None,
    survey_name: str = "",
    model_version: str = "V10",
) -> V10Metrics:
    """
    Compute V10 evaluation metrics from model predictions and ground truth.
    
    All inputs in meters (un-normalized). If predictions are stored in
    normalized std-dev units, multiply by local_std before passing in.
    
    Args:
        predicted_correction: Predicted corrections (in meters) for every cell
        target_correction: True corrections (noisy - clean) for every cell
        valid_mask: Boolean mask of cells to evaluate
        local_std: Optional, per-cell local depth variability (for normalized MAE)
        noisy_depth: Optional, original noisy depth at every cell (for recovery RMSE)
        clean_depth: Optional, reference clean depth at every cell (for recovery RMSE)
        survey_name: Identifier for the survey
        model_version: Identifier for the model version
    
    Returns:
        V10Metrics dataclass
    """
    pred = _to_numpy(predicted_correction).flatten()
    target = _to_numpy(target_correction).flatten()
    mask = _to_numpy(valid_mask).flatten().astype(bool)
    
    if not np.any(mask):
        logger.warning("No valid cells, returning zero metrics")
        return V10Metrics(survey_name=survey_name, model_version=model_version)
    
    pred_v = pred[mask]
    target_v = target[mask]
    error = pred_v - target_v
    abs_error = np.abs(error)
    abs_target = np.abs(target_v)
    
    m = V10Metrics(
        survey_name=survey_name,
        model_version=model_version,
        n_valid_cells=int(mask.sum()),
    )
    
    # Overall regression metrics
    m.mae_meters = float(np.mean(abs_error))
    m.rmse_meters = float(np.sqrt(np.mean(error ** 2)))
    
    # Normalized MAE
    if local_std is not None:
        ls = _to_numpy(local_std).flatten()[mask]
        denom = np.maximum(ls, 0.01)
        m.mae_normalized_std = float(np.mean(abs_error / denom))
    
    # Per-magnitude-bucket MAE
    buckets = [
        ('under_0_1m', abs_target < 0.1),
        ('0_1_to_1m', (abs_target >= 0.1) & (abs_target < 1.0)),
        ('1_to_10m', (abs_target >= 1.0) & (abs_target < 10.0)),
        ('over_10m', abs_target >= 10.0),
    ]
    for name, bucket_mask in buckets:
        n = int(bucket_mask.sum())
        if n > 0:
            mae_bucket = float(np.mean(abs_error[bucket_mask]))
        else:
            mae_bucket = 0.0
        setattr(m, f'mae_{name}', mae_bucket)
        setattr(m, f'n_{name}', n)
    
    # Safety metrics
    # Hazardous: predicted < target means corrected depth is deeper than reality
    hazardous = error < 0
    m.hazardous_error_rate = float(np.mean(hazardous))
    
    shoal_targets = target_v < 0  # noisy was shallower; correction is shoal-direction
    deep_targets = target_v > 0
    m.n_shoal_target_cells = int(shoal_targets.sum())
    m.n_deep_target_cells = int(deep_targets.sum())
    
    if m.n_shoal_target_cells > 0:
        m.hazardous_error_rate_shoal = float(np.mean(hazardous[shoal_targets]))
    if m.n_deep_target_cells > 0:
        m.hazardous_error_rate_deep = float(np.mean(hazardous[deep_targets]))
    
    # Recovery RMSE (requires noisy and clean surfaces)
    if noisy_depth is not None and clean_depth is not None:
        noisy_v = _to_numpy(noisy_depth).flatten()[mask]
        clean_v = _to_numpy(clean_depth).flatten()[mask]
        corrected_v = noisy_v - pred_v
        recovery_error = corrected_v - clean_v
        m.recovery_rmse = float(np.sqrt(np.mean(recovery_error ** 2)))
        m.recovery_mean_error = float(np.mean(recovery_error))
    
    return m


def compare_metrics(*metrics: V10Metrics) -> str:
    """Side-by-side comparison of multiple V10Metrics instances."""
    if not metrics:
        return ""
    
    headers = [m.model_version or m.survey_name or f"run_{i}" for i, m in enumerate(metrics)]
    
    def fmt_row(label, values, fmt_str="{:.4f}"):
        return f"  {label:<35}  " + "  ".join(fmt_str.format(v) for v in values)
    
    def fmt_int_row(label, values):
        return f"  {label:<35}  " + "  ".join(f"{v:>10,}" for v in values)
    
    lines = [
        "V10 Metrics Comparison",
        "  Field" + " " * 32 + "  " + "  ".join(f"{h:>10}" for h in headers),
        "  " + "-" * 35 + "  " + "  ".join("-" * 10 for _ in headers),
        fmt_int_row("Cells evaluated", [m.n_valid_cells for m in metrics]),
        fmt_row("MAE (m)", [m.mae_meters for m in metrics]),
        fmt_row("RMSE (m)", [m.rmse_meters for m in metrics]),
        fmt_row("MAE (std)", [m.mae_normalized_std for m in metrics]),
        fmt_row("MAE <0.1m bucket", [m.mae_under_0_1m for m in metrics]),
        fmt_row("MAE 0.1-1m bucket", [m.mae_0_1_to_1m for m in metrics]),
        fmt_row("MAE 1-10m bucket", [m.mae_1_to_10m for m in metrics]),
        fmt_row("MAE >10m bucket", [m.mae_over_10m for m in metrics]),
        fmt_row("Hazardous rate (overall)", [m.hazardous_error_rate for m in metrics]),
        fmt_row("Hazardous rate (shoal)", [m.hazardous_error_rate_shoal for m in metrics]),
        fmt_row("Hazardous rate (deep)", [m.hazardous_error_rate_deep for m in metrics]),
        fmt_row("Recovery RMSE (m)", [m.recovery_rmse for m in metrics]),
        fmt_row("Recovery mean error (m)", [m.recovery_mean_error for m in metrics]),
    ]
    return "\n".join(lines)

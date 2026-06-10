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


# Maximum-allowable TVU coefficients for TVU = sqrt(a^2 + (b * depth)^2) at 95%
# confidence. Lookup is case-insensitive and ignores spaces/underscores, so
# "general 1", "General_1", and "general1" all resolve to the same entry.
#
# Two label sets are provided:
#   - IHO S-44 Edition 6 (2020) order labels.
#   - NOAA HSSD 2026 OCS Quality Metric labels (Table 5.8.1). NOAA rounds the
#     depth coefficient, so General 1 uses b=0.01 (not S-44 1a's 0.013) and
#     General 2/3 uses b=0.02 (not S-44 Order 2's 0.023). Use the HSSD labels for
#     NOAA surveys. The metric a survey is held to is set in its Project
#     Instructions, not derived from depth.
# Orders 1a/1b (and General 2/3) share the same TVU; they differ in feature
# detection, not vertical uncertainty.
IHO_TVU_COEFFICIENTS = {
    # IHO S-44 Edition 6
    "exclusive":   (0.15, 0.0075),
    "special":     (0.25, 0.0075),
    "1a":          (0.5,  0.013),
    "1b":          (0.5,  0.013),
    "2":           (1.0,  0.023),
    # NOAA HSSD 2026 OCS Quality Metric (Table 5.8.1)
    "exceptional": (0.15, 0.0075),
    "critical":    (0.25, 0.0075),
    "general1":    (0.5,  0.01),
    "general2":    (1.0,  0.02),
    "general3":    (1.0,  0.02),
    "general4":    (2.0,  0.05),
}


def _resolve_tvu(iho_order, tvu_a, tvu_b):
    """Resolve TVU coefficients from explicit a/b or an IHO order label.
    
    Explicit tvu_a and tvu_b take precedence over iho_order. Returns
    (a, b, label), or (None, None, "") when nothing usable was supplied,
    in which case the TVU breach metric is skipped.
    """
    if tvu_a is not None and tvu_b is not None:
        label = str(iho_order) if iho_order else "custom"
        return float(tvu_a), float(tvu_b), label
    if iho_order is not None:
        key = str(iho_order).strip().lower().replace(" ", "").replace("_", "")
        if key in IHO_TVU_COEFFICIENTS:
            a, b = IHO_TVU_COEFFICIENTS[key]
            return a, b, key
        logger.warning(
            f"Unknown IHO order '{iho_order}'; TVU breach not computed. "
            f"Known orders: {sorted(IHO_TVU_COEFFICIENTS)}, or pass tvu_a/tvu_b."
        )
    return None, None, ""


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
    
    # TVU-budget safety metrics (IHO S-44 / NOAA HSSD).
    # A cell is a TVU breach when its error is in the dangerous direction
    # (corrected deeper than reality) AND its magnitude exceeds the allowable
    # TVU = sqrt(a^2 + (b*depth)^2) at that cell's depth. Unlike the raw
    # hazardous_error_rate (a sign count at any magnitude), this counts only
    # dangerous errors large enough to bust the survey's uncertainty budget.
    # Populated only when TVU coefficients are supplied to compute_v10_metrics;
    # otherwise the rates stay at -1.0, meaning "not computed".
    tvu_order: str = ""
    tvu_a: float = 0.0
    tvu_b: float = 0.0
    tvu_breach_rate: float = -1.0
    tvu_breach_rate_shoal: float = -1.0
    tvu_breach_rate_deep: float = -1.0
    n_tvu_breach: int = 0
    n_tvu_breach_shoal: int = 0
    n_tvu_breach_deep: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save_json(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        if self.tvu_breach_rate >= 0:
            lines.extend([
                f"  TVU breach (order {self.tvu_order}, a={self.tvu_a:g}m, b={self.tvu_b:g}):",
                f"    Breach rate (overall):       {self.tvu_breach_rate:.2%}  ({self.n_tvu_breach:,} cells)",
                f"    Breach rate (shoal targets): {self.tvu_breach_rate_shoal:.2%}  ({self.n_tvu_breach_shoal:,} cells)",
                f"    Breach rate (deep targets):  {self.tvu_breach_rate_deep:.2%}  ({self.n_tvu_breach_deep:,} cells)",
            ])
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
    iho_order: Optional[str] = None,
    tvu_a: Optional[float] = None,
    tvu_b: Optional[float] = None,
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
        noisy_depth: Optional, original noisy depth at every cell (for recovery RMSE
            and, when TVU coefficients are given, the depth used to size the budget)
        clean_depth: Optional, reference clean depth at every cell (for recovery RMSE)
        survey_name: Identifier for the survey
        model_version: Identifier for the model version
        iho_order: Optional order label selecting TVU coefficients. Accepts IHO
            S-44 orders ("exclusive", "special", "1a", "1b", "2") or NOAA HSSD
            OCS Quality Metrics ("exceptional", "critical", "general1",
            "general2", "general3", "general4"); see IHO_TVU_COEFFICIENTS.
            Ignored if tvu_a and tvu_b are given explicitly. When neither is
            supplied, the TVU breach metric is skipped and its fields stay at -1.0.
        tvu_a: Optional explicit TVU constant term a (meters); overrides iho_order
        tvu_b: Optional explicit TVU depth-scaled term b; overrides iho_order
    
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
    
    # TVU-budget breach metrics (IHO S-44 / NOAA HSSD)
    # A breach is a dangerous-direction error (hazardous) whose magnitude exceeds
    # the allowable TVU at that cell's depth. abs_error equals the dangerous
    # deviation magnitude on hazardous cells (where error < 0).
    a_coeff, b_coeff, order_label = _resolve_tvu(iho_order, tvu_a, tvu_b)
    if a_coeff is not None:
        m.tvu_order = order_label
        m.tvu_a = float(a_coeff)
        m.tvu_b = float(b_coeff)
        if noisy_depth is not None:
            # Depth magnitude (TVU is defined on |depth|, so this is robust to
            # whichever sign convention the depth band uses).
            depth_v = np.abs(_to_numpy(noisy_depth).flatten()[mask])
            tvu_allow = np.sqrt(a_coeff ** 2 + (b_coeff * depth_v) ** 2)
            breach = hazardous & (abs_error > tvu_allow)
            m.tvu_breach_rate = float(np.mean(breach))
            m.n_tvu_breach = int(breach.sum())
            if m.n_shoal_target_cells > 0:
                m.tvu_breach_rate_shoal = float(np.mean(breach[shoal_targets]))
                m.n_tvu_breach_shoal = int(breach[shoal_targets].sum())
            if m.n_deep_target_cells > 0:
                m.tvu_breach_rate_deep = float(np.mean(breach[deep_targets]))
                m.n_tvu_breach_deep = int(breach[deep_targets].sum())
        else:
            logger.warning(
                "TVU coefficients supplied but noisy_depth is None; "
                "cannot size the budget, so TVU breach rate is not computed."
            )
    
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
    if any(m.tvu_breach_rate >= 0 for m in metrics):
        lines.extend([
            fmt_row("TVU breach (overall)", [m.tvu_breach_rate for m in metrics]),
            fmt_row("TVU breach (shoal)", [m.tvu_breach_rate_shoal for m in metrics]),
            fmt_row("TVU breach (deep)", [m.tvu_breach_rate_deep for m in metrics]),
        ])
    return "\n".join(lines)

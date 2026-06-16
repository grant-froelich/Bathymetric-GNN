#!/usr/bin/env python3
"""
Direction test for the asymmetric shoal-safety loss and the V10/V11 metrics.

WHY THIS TEST EXISTS
--------------------
From the start of regression mode through V10 the depth data was negative-down
while the loss and metrics assumed positive-down, which silently INVERTED the
safety asymmetry: the 3x penalty protected the safe direction and barely
penalized the dangerous one, and the hazard/breach metrics counted the wrong
cells. Every magnitude metric (MAE, RMSE, loss curve) is blind to this, so
nothing in the suite could catch it. The only fingerprint was the sign of
recovery_mean_error, which was itself misdescribed in the docs. See
LESSONS_LEARNED Lessons 20 and 21.

This test is the one assertion that is actually sensitive to direction. It
builds tiny synthetic cells with a known-dangerous and a known-safe error of
EQUAL magnitude (so MAE and any symmetric loss treat them identically) and
asserts:
  1. the loss penalizes the dangerous direction ~3x harder (both Huber regimes),
  2. compute_v10_metrics flags the dangerous cells as hazardous and partitions
     shoal- vs deep-target cells correctly,
  3. a TVU breach requires BOTH the dangerous direction AND a magnitude over the
     budget (Lesson 19): direction alone or magnitude alone is not a breach,
  4. recovery_mean_error is negative when the model errs safe and positive when
     it errs dangerous (the sign mapping that was historically inverted),
  5. the loss and the metrics agree on what "dangerous" means.

If a future convention change or refactor re-inverts the sign, the loss ratio
flips to ~1/3, the dangerous cells stop being flagged, and this test fails in
milliseconds instead of the inversion hiding for months behind clean MAE.

Fully synthetic: no model, no GPU, no ground-truth files.

Usage:
    python scripts/test_direction.py        # standalone, exits non-zero on failure
    pytest scripts/test_direction.py        # also collectable as pytest tests

Run this before any training run after touching losses.py, metrics.py, the
depth-convention handling in the loader, or the ground-truth sign convention.
"""

import os
# Fix OpenMP conflict on Windows - must be before any other imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import torch BEFORE numpy to avoid DLL conflicts on Windows
import torch

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from training.losses import RegressionLoss
from training.metrics import compute_v10_metrics

# Tolerances
EXACT = 1e-9   # for values that are exactly representable from the construction
RATIO = 1e-6   # for the asymmetry ratio

# Convention reminder (positive-down depths, correction = noisy - clean):
#   error = predicted_correction - target_correction
#   error < 0  -> corrected depth DEEPER than truth  -> DANGEROUS (less clearance than exists)
#   error > 0  -> corrected depth SHALLOWER than truth -> SAFE (conservative)
#   target < 0 -> noisy shallower than clean -> shoal-direction target
#   target > 0 -> noisy deeper than clean   -> deep-direction target
#   recovery_error = corrected - clean = -(error): negative = shallower than truth = safe


def _loss(pred, target, delta=1.0, safe_weight=1.0, dangerous_weight=3.0):
    """Run RegressionLoss on 1-D lists; returns a python float. Uses float64 to
    keep the exact-value assertions clean."""
    loss_fn = RegressionLoss(delta=delta, safe_weight=safe_weight,
                             dangerous_weight=dangerous_weight)
    p = torch.tensor(pred, dtype=torch.float64)
    t = torch.tensor(target, dtype=torch.float64)
    return float(loss_fn(p, t))


def _metrics(pred, target, valid_mask, noisy_depth, clean_depth, iho_order="general2"):
    """Run compute_v10_metrics on 1-D lists; returns the V10Metrics dataclass."""
    return compute_v10_metrics(
        predicted_correction=np.asarray(pred, dtype=np.float64),
        target_correction=np.asarray(target, dtype=np.float64),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        noisy_depth=np.asarray(noisy_depth, dtype=np.float64),
        clean_depth=np.asarray(clean_depth, dtype=np.float64),
        iho_order=iho_order,
        survey_name="direction_test",
        model_version="test",
    )


# ---------------------------------------------------------------------------
# 1. Loss: the dangerous direction is penalized more, by exactly the weight ratio
# ---------------------------------------------------------------------------
def test_loss_penalizes_dangerous_direction_more():
    """Equal-magnitude errors: the dangerous one (error < 0) must cost more.
    Checked in both Huber regimes. This is the assertion that flips if the sign
    convention re-inverts."""
    target = 2.0  # a deep-target cell; the asymmetry is about error sign, not target sign

    # Quadratic regime: |error| = 0.5 < delta (1.0)
    dangerous_q = _loss([target - 0.5], [target])  # error = -0.5
    safe_q = _loss([target + 0.5], [target])        # error = +0.5
    assert dangerous_q > safe_q, (
        f"quadratic regime: dangerous ({dangerous_q}) should exceed safe ({safe_q}); "
        "if this fails the safety asymmetry is inverted"
    )

    # Linear regime: |error| = 2.0 > delta (1.0)
    dangerous_l = _loss([target - 2.0], [target])  # error = -2.0
    safe_l = _loss([target + 2.0], [target])        # error = +2.0
    assert dangerous_l > safe_l, (
        f"linear regime: dangerous ({dangerous_l}) should exceed safe ({safe_l})"
    )


def test_loss_asymmetry_ratio_matches_weights():
    """The dangerous/safe ratio must equal dangerous_weight / safe_weight exactly
    (the weight multiplies a sign-symmetric Huber base, so for equal |error| the
    ratio is exact in BOTH regimes). Also verifies the exact values from the
    delta=1.0 construction, as a regression guard on the Huber math."""
    safe_w, dang_w = 1.0, 3.0
    expected_ratio = dang_w / safe_w  # 3.0
    target = 2.0

    # Quadratic regime, |error| = 0.5: huber = 0.5 * 0.5^2 = 0.125
    dangerous_q = _loss([target - 0.5], [target], safe_weight=safe_w, dangerous_weight=dang_w)
    safe_q = _loss([target + 0.5], [target], safe_weight=safe_w, dangerous_weight=dang_w)
    assert abs(safe_q - 0.125) < EXACT, f"safe quadratic huber expected 0.125, got {safe_q}"
    assert abs(dangerous_q - 0.375) < EXACT, f"dangerous quadratic expected 0.375, got {dangerous_q}"
    assert abs(dangerous_q / safe_q - expected_ratio) < RATIO

    # Linear regime, |error| = 2.0: huber = 0.5*1^2 + 1*(2-1) = 1.5
    dangerous_l = _loss([target - 2.0], [target], safe_weight=safe_w, dangerous_weight=dang_w)
    safe_l = _loss([target + 2.0], [target], safe_weight=safe_w, dangerous_weight=dang_w)
    assert abs(safe_l - 1.5) < EXACT, f"safe linear huber expected 1.5, got {safe_l}"
    assert abs(dangerous_l - 4.5) < EXACT, f"dangerous linear expected 4.5, got {dangerous_l}"
    assert abs(dangerous_l / safe_l - expected_ratio) < RATIO


def test_loss_default_weights_are_protective():
    """The default asymmetry must point the protective way (dangerous heavier)."""
    loss_fn = RegressionLoss()
    assert loss_fn.dangerous_weight > loss_fn.safe_weight, (
        f"default dangerous_weight ({loss_fn.dangerous_weight}) must exceed "
        f"safe_weight ({loss_fn.safe_weight})"
    )


# ---------------------------------------------------------------------------
# 2. Metrics: hazardous flag and shoal/deep partition follow the sign convention
# ---------------------------------------------------------------------------
def test_metrics_partition_and_hazard():
    """Five cells covering deep/shoal x dangerous/safe, plus one masked-out cell.
    Cells (positive-down, depth 20 so general2 TVU = 1.077 m):
      0 deep-target,  dangerous, over budget : clean 18, noisy 20 (target +2), pred 0  -> err -2, corrected 20 (deeper than 18)
      1 deep-target,  safe,      large mag   : clean 18, noisy 20 (target +2), pred 4  -> err +2, corrected 16 (shallower)
      2 shoal-target, dangerous, over budget : clean 22, noisy 20 (target -2), pred -4 -> err -2, corrected 24 (deeper than 22)
      3 shoal-target, safe                   : clean 22, noisy 20 (target -2), pred -1 -> err +1, corrected 21 (shallower)
      4 masked out (absurd values)           : excluded; if masking breaks, totals blow up
    """
    clean = [18, 18, 22, 22, 1]
    noisy = [20, 20, 20, 20, 1000]
    target = [n - c for n, c in zip(noisy, clean)]   # noisy - clean = [2, 2, -2, -2, 999]
    pred = [0, 4, -4, -1, 999]
    mask = [1, 1, 1, 1, 0]

    m = _metrics(pred, target, mask, noisy, clean, iho_order="general2")

    assert m.n_valid_cells == 4, f"masked cell not excluded: n_valid_cells={m.n_valid_cells}"
    assert m.n_shoal_target_cells == 2, f"n_shoal_target_cells={m.n_shoal_target_cells}, expected 2"
    assert m.n_deep_target_cells == 2, f"n_deep_target_cells={m.n_deep_target_cells}, expected 2"
    # cells 0 and 2 are dangerous -> 2 of 4 hazardous
    assert abs(m.hazardous_error_rate - 0.5) < EXACT, f"hazardous_error_rate={m.hazardous_error_rate}"
    assert abs(m.hazardous_error_rate_shoal - 0.5) < EXACT  # cell 2 of {2,3}
    assert abs(m.hazardous_error_rate_deep - 0.5) < EXACT   # cell 0 of {0,1}


def test_metrics_breach_requires_direction_and_budget():
    """A TVU breach needs the dangerous direction AND a magnitude over the budget.
    general2 at depth 20 -> TVU = 1.077 m."""
    # Dangerous and over budget (|error| = 2.0 > 1.077): one breach.
    m_breach = _metrics([0], [2], [1], [20], [18])  # err -2, corrected 20 > clean 18
    assert m_breach.hazardous_error_rate == 1.0
    assert m_breach.n_tvu_breach == 1, f"expected 1 breach, got {m_breach.n_tvu_breach}"

    # Dangerous but UNDER budget (|error| = 0.5 < 1.077): hazardous, but not a breach.
    m_subbudget = _metrics([1.5], [2], [1], [20], [18])  # err -0.5
    assert m_subbudget.hazardous_error_rate == 1.0, "still counts as a sign-level hazard"
    assert m_subbudget.n_tvu_breach == 0, (
        f"sub-budget dangerous error must not breach, got {m_subbudget.n_tvu_breach} "
        "(this is the whole point of the budget-aware metric, Lesson 19)"
    )

    # SAFE but huge (|error| = 8.0 >> budget): never a breach regardless of magnitude.
    m_safe_huge = _metrics([10], [2], [1], [20], [18])  # err +8, corrected 10 (much shallower)
    assert m_safe_huge.hazardous_error_rate == 0.0
    assert m_safe_huge.n_tvu_breach == 0, (
        f"safe-direction error must never breach, got {m_safe_huge.n_tvu_breach}; "
        "if this fails the breach metric is counting the wrong direction"
    )


# ---------------------------------------------------------------------------
# 3. recovery_mean_error sign: negative = safe, positive = dangerous
# ---------------------------------------------------------------------------
def test_metrics_recovery_sign_convention():
    """recovery_error = corrected - clean = -(error). A model that errs to the
    SAFE side (shallower than truth) has NEGATIVE recovery_mean_error; one that
    errs dangerous (deeper than truth) has POSITIVE. This is the exact mapping
    that the docs had backwards before V11 (Lessons 20, 21)."""
    # All safe: pred over-corrects -> corrected shallower than clean.
    m_safe = _metrics([4, 3], [2, 2], [1, 1], [20, 20], [18, 18])
    assert m_safe.hazardous_error_rate == 0.0
    assert m_safe.recovery_mean_error < 0, (
        f"safe bias must give negative recovery_mean_error, got {m_safe.recovery_mean_error}"
    )

    # All dangerous: pred under-corrects -> corrected deeper than clean.
    m_dangerous = _metrics([0, 1], [2, 2], [1, 1], [20, 20], [18, 18])
    assert m_dangerous.hazardous_error_rate == 1.0
    assert m_dangerous.recovery_mean_error > 0, (
        f"dangerous bias must give positive recovery_mean_error, got {m_dangerous.recovery_mean_error}"
    )


# ---------------------------------------------------------------------------
# 4. Loss and metrics agree on what "dangerous" means
# ---------------------------------------------------------------------------
def test_loss_and_metrics_share_dangerous_definition():
    """From the SAME (pred, target), the loss must weight the dangerous cell more
    AND the metrics must flag it hazardous; the safe mirror must be lighter AND
    not hazardous. Catches one layer inverting relative to the other."""
    target = 2.0
    dangerous_pred = target - 1.0   # error = -1.0 (corrected deeper than truth)
    safe_pred = target + 1.0        # error = +1.0 (corrected shallower than truth)

    # Loss side
    assert _loss([dangerous_pred], [target]) > _loss([safe_pred], [target])

    # Metrics side, same predictions (depth 20, clean 18 -> noisy 20, target +2)
    m_dangerous = _metrics([dangerous_pred], [target], [1], [20], [18])
    m_safe = _metrics([safe_pred], [target], [1], [20], [18])
    assert m_dangerous.hazardous_error_rate == 1.0, "metrics must flag the loss-dangerous cell"
    assert m_safe.hazardous_error_rate == 0.0, "metrics must not flag the loss-safe cell"


# ---------------------------------------------------------------------------
# Standalone runner (test_pipeline.py idiom: print-based, non-zero exit on fail)
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("DIRECTION TEST: asymmetric safety loss + metrics")
    print("=" * 60)
    print(f"PyTorch {torch.__version__}, NumPy {np.__version__}")

    tests = [
        ("Loss penalizes dangerous direction more", test_loss_penalizes_dangerous_direction_more),
        ("Loss asymmetry ratio matches weights", test_loss_asymmetry_ratio_matches_weights),
        ("Default weights are protective", test_loss_default_weights_are_protective),
        ("Metrics partition + hazard flag", test_metrics_partition_and_hazard),
        ("Breach requires direction AND budget", test_metrics_breach_requires_direction_and_budget),
        ("recovery_mean_error sign convention", test_metrics_recovery_sign_convention),
        ("Loss and metrics agree on 'dangerous'", test_loss_and_metrics_share_dangerous_definition),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  \u2713 {name}")
            passed += 1
        except AssertionError as e:
            print(f"  \u2717 {name}")
            print(f"      {e}")
            failed += 1
        except Exception as e:
            print(f"  \u2717 {name}  (unexpected error)")
            print(f"      {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {passed} passed, {failed} failed")
    if failed:
        print("\n\u2717 Direction test FAILED. The safety asymmetry may be inverted.")
        print("  Do NOT train or ship until this passes. See LESSONS_LEARNED 20-21.")
        sys.exit(1)
    print("\n\u2713 Safety asymmetry points the right way (dangerous penalized 3x,")
    print("  hazard/breach/recovery-sign all consistent with positive-down).")


if __name__ == "__main__":
    main()

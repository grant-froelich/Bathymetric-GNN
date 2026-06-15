# Changelog

## 2026-06-15 - V11 Trained and Evaluated: Sign Fix Validated, fp32 vs bf16 Resolved

V11 is the first model trained after the depth-convention fix and the tile-coverage
fix (see the two 2026-06-09 entries). Both the fp32 and bf16 paired runs completed on
the regenerated positive-down ground truth, and the full evaluation was re-run with
correct direction semantics. This supersedes every direction-sensitive number in the
2026-06-09 bf16 entry below, all of which were measured under the inverted sign
convention.

### Training
- Regenerated ground truth, all positive-down (`--regression-mode --no-offset`): 6
  training files (E00269 4m/8m/128m/256m sub-files + H13739 VR + H14070 VR), 1,310
  tiles; 3 validation files, 282 tiles.
- GAT, 4 layers, 64 hidden channels, 182,533 parameters. Correction Huber delta 6.566
  (sampled from normalized corrections at startup).
- fp32: best val loss 0.9302 at epoch 19, early stop at epoch 34, ~56 h wall clock.
- bf16 (`--amp`): paired run on the same data and split.

### The sign fix is validated
The 2026-06-09 dashboard predicted `recovery_mean_error` would flip positive as the
success indicator for the sign fix. That prediction had the sign backwards. Under the
post-fix positive-down convention that `metrics.py` implements, `recovery_error =
corrected - clean = -(error)`, so a **negative** `recovery_mean_error` means the
corrected surface sits shallower than truth on average, which is the conservative
(safe) direction. V11 fp32 `recovery_mean_error` is negative at all three validation
surfaces (shallow -0.585, deep -6.615, Alaska -0.179 m) and the overall hazardous rate
is under 50% everywhere (28-36%), so most cells err to the safe side. The asymmetric
3x penalty is now pushing the model the right way. The indicator manifests as
`recovery_mean_error` staying negative, not flipping positive, because the
load-boundary negation also flipped the sign-to-meaning mapping (pre-fix negative meant
dangerous; post-fix negative means safe). A still-inverted model would instead show
positive `recovery_mean_error` and a hazardous rate above 50%.

### fp32 vs bf16 (corrected signs, the authoritative comparison)
Three held-out validation surfaces, HSSD General 1 (shallow) and General 2 (deep,
Alaska). Both breach fields are dangerous-direction (corrected deeper than truth)
rates, partitioned by the true correction's direction.

| Surface (order) | MAE m, fp32 -> bf16 | shoal-target breach cells, fp32 -> bf16 | deep-target breach cells, fp32 -> bf16 |
|---|---|---|---|
| Shallow (general1) | 0.88 -> 1.22 | 0 -> 0 | 315 -> 93 |
| Deep (general2) | 11.86 -> 13.54 | 114 -> 1,539 | 4,880 -> 5,703 |
| Alaska H14116 (general2) | 10.22 -> 11.74 | 515 -> 924 | 1,734 -> 2,243 |

fp32 wins MAE at all three. fp32 wins or ties the shoal-target dangerous breach at all
three: tied at shallow (neither produces one), 13.5x lower at deep, 1.8x lower at
Alaska. The only place bf16 looks better is shallow deep-target and overall breach
(315 -> 93), but shallow has zero shoal-direction danger for either precision and
bf16's shallow MAE is worse, so it does not move the verdict. Both precisions carry a
conservative mean bias (recovery_mean_error negative everywhere); bf16's is slightly
more conservative on average yet has a markedly fatter dangerous tail, which is the
coarse-mantissa signature.

### Two 2026-06-09 conclusions are overturned (both were sign-inversion artifacts)
- "MAE improved with bf16 at all three locations." Does not reproduce on the
  regenerated ground truth: fp32 has lower MAE everywhere. The earlier fp32 MAE was
  inflated by the offset/convention issues that motivated the regeneration (earlier
  fp32 shallow MAE 2.41 vs V11 fp32 0.88).
- "Shoal-target breach stayed at ~0% for both precisions; bf16 is defensible for the
  deliverable." Wrong. With correct signs the shoal-target breach is not zero, and
  bf16 raises it 13.5x at deep. bf16 is NOT defensible on the deliverable path.

### Decision: bf16 for experimentation, fp32 for the release
- bf16 (`--amp`) is the development/experimentation default. The ~5x speedup is pure
  throughput when iterating on architecture, features, thresholds, and data mixes
  judged on aggregate metrics (MAE, loss), where the coarse mantissa does not matter.
- The qualified, shipped model is trained in fp32. The go/no-go is the shoal-target
  breach number, not MAE, and bf16 fails it. Training precision is decoupled from
  inference precision, so training the release model in fp32 costs nothing per survey
  processed later.
- Mechanism: bf16's coarse mantissa rounds away the fine directional gradient the
  asymmetric 3x shoal loss places on near-zero corrections, eroding exactly the
  discipline that protects shoals. The 13.5x shoal-breach jump at deep is the
  fingerprint.

### Caveat
Still one paired run. The bar for ever promoting bf16 to the deployed path was
shoal-breach parity across a few paired runs, not MAE parity; this run does not clear
it. The stakes asymmetry plus the consistency across both deep sites makes fp32 the
release choice now. See LESSONS_LEARNED Lessons 19, 20, and 21.

---

## 2026-06-09 - Depth Convention Fix and Full Repo Scrub (V11 Prep)

### CRITICAL FIX: Inverted Sign Convention (all direction-sensitive components)
A full repo scrub found that ground-truth bands stored GDAL elevation
(negative-down) while every loss, metric, and doc assumed positive-down depth.
Under the real data, `error > 0` was the dangerous direction, so since
regression mode was introduced: the 3x "shoal safety" weighting penalized the
SAFE direction and lightly penalized the dangerous one (observable as the
consistently negative recovery_mean_error in every eval: each model settled on
the deeper-than-truth side); the hazard metrics counted safe-direction errors
as hazardous; the shoal/deep target split was swapped; V9's ShoalSafetyLoss
weighted deep spikes instead of shoals. Magnitude metrics (MAE/RMSE/recovery
RMSE) were unaffected. All V10-and-earlier direction-sensitive numbers
(hazard rates, TVU breach rates, shoal/deep labels, including the bf16
comparison tables below) are inverted and must be re-measured after retraining.

Fix (option A, normalize at the boundary):
- `BathymetricLoader` now enforces positive-down depth in memory for every
  format. `depth_convention='auto'` (default) detects elevation sources by
  negative median and negates valid cells on load; 'positive_down' /
  'negative_down' force the interpretation. The grid records
  `source_was_negative_down`.
- `prepare_ground_truth.py` therefore now writes positive-down bands, tags the
  tif (`DEPTH_CONVENTION=POSITIVE_DOWN`) and the stats JSON
  (`depth_convention`).
- `GroundTruthDataset` and `spatial_error_map.py` REFUSE files whose median
  valid depth is negative, so stale pre-fix ground truth cannot enter a run.
  **All 9 ground-truth tifs must be regenerated** (same commands, `--no-offset`
  where applicable) and the model retrained: existing checkpoints embody the
  inverted objective. The retrained model is designated **V11**.
- Legacy V9 inference scripts now emit a prominent warning: pre-fix checkpoints
  are incompatible with the fixed loader.

### Fixed: Tile coverage gap (training AND evaluation)
Tiling covered only interior stride positions plus a single bottom-right corner
tile, leaving the right and bottom edge strips (up to stride-1 px) in no tile:
4-15% of cells at typical settings, up to ~40% on unlucky grid dims, excluded
from training and from every reported evaluation. Grids smaller than tile_size
produced no tiles at all. Both `GroundTruthDataset` and `spatial_error_map.py`
now anchor a final tile to each edge and handle small grids.

### Fixed: Survey border ring written as depth 0.0 (V9 inference merge)
Blend weights reach exactly 0 at tile edges; zero-total-weight cells (the
survey's outer ring) were initialized to 0.0 and never averaged, emerging as
spurious 0 m depths (artificial shoals). `TileMerger.finalize_output` now
resets zero-weight float cells to NaN.

### Performance: vectorized graph construction
`_build_edges` and `_compute_edge_features` replaced per-node/per-edge Python
loops (with dict lookups) with vectorized numpy. Edge set and features are
identical (verified bitwise in randomized tests; edge order is offset-major,
semantically irrelevant). Speeds up evaluation, the Huber-delta startup
sampling, and future inference (training was already worker-hidden).
`scripts/verify_graph_equivalence.py` (new) embeds the legacy implementation
verbatim and must PASS on real ground truth before the V11 retrain.

### Retired: synthetic-noise training path
`--clean-surveys` and `BathymetricGraphDataset` removed (the path had crashed
at Trainer init on a missing labels key since ground-truth stats were added).
`--ground-truth-dir` is now required; `--vr-bag-mode` removed from train.py.
`data/synthetic_noise.py` remains as a standalone module (used by
test_pipeline.py).

### Smaller fixes
- Checkpoints record true `in_channels`/`edge_dim` from the model instead of a
  hardcoded `edge_dim: 3`.
- Mixed classification/regression batches now raise instead of being silently
  coerced to the first graph's mode.
- `gdal.UseExceptions()` set explicitly (silences the GDAL 4.0 FutureWarning).
- training_history.json: `train_acc`/`val_acc` renamed `train_metric`/
  `val_metric` plus `metric_name` ('accuracy' or 'mae').
- Early-stopping log now 1-indexed like all other epoch logs.
- Empty-graph feature width now matches real graphs (latent batch-collation
  trap).
- Unused `warmup_epochs` removed from config; scheduler comment corrected
  (no "step" implementation exists).
- Dead `MultiScaleGraphBuilder` removed.

### Required sequence before V11 training
1. Regenerate all 9 ground-truth tifs with the fixed pipeline (`--no-offset`).
2. `python scripts/verify_graph_equivalence.py --ground-truth-dir ground-truth-train/` must PASS.
3. Retrain (fp32 and bf16 paired runs); re-run all evals including the TVU
   comparison with now-correct direction labels.

---

## 2026-06-09 - bf16 Mixed Precision, Graph-Cache Removal, TVU-Budget Safety Metric

### Performance: bf16 Mixed-Precision Training
Added opt-in bf16 autocast, selectable with `--amp`. Autocast wraps the forward pass and loss in both the train and validation steps; backward runs outside it. bf16 (not fp16) is used so no GradScaler is needed (it shares fp32's exponent range); master weights stay fp32.
- ~5x speedup on the multi-location run: 5.51 -> 1.07 s/it, ~36 hours -> ~5.5 hours
- Default path unchanged: with `--amp` absent the run is full fp32, identical to before
- DataLoader `persistent_workers=True` and `prefetch_factor=4` when `num_workers > 0` (Windows respawns workers every epoch otherwise)

### Removed: Disk Graph Cache
The `--graph-cache-dir` path and the on-disk graph store were removed. Profiling (`nvidia-smi dmon -s u`) showed the GPU SM pegged near 100% during training, so graph construction was never the bottleneck: the DataLoader workers already hid it behind GPU compute. The cache built correctly (1286 graphs, ~16.7 GB) but produced no speedup, because the actual wall is GAT compute over the large per-tile graphs (~46K nodes / ~366K edges each), which is what bf16 addresses.

### New Safety Metric: TVU-Budget Breach Rate
Added to `metrics.py` and `scripts/evaluate_v10.py`. Counts a cell as a breach only when its dangerous-direction error (corrected deeper than truth) exceeds the allowable TVU = sqrt(a^2 + (b*depth)^2) at that cell's depth. Reported as `tvu_breach_rate`, `tvu_breach_rate_shoal`, `tvu_breach_rate_deep`, alongside the existing raw hazard rates (not replacing them).
- Selectable via `--iho-order`: IHO S-44 (exclusive/special/1a/1b/2) or NOAA HSSD OCS Quality Metric (exceptional/critical/general1/general2/general3/general4), or explicit `--tvu-a`/`--tvu-b`
- HSSD coefficients from HSSD 2026 Table 5.8.1; NOAA rounds the S-44 depth term (General 1 b=0.01, General 2/3 b=0.02)

### bf16 vs fp32 Validation (budget-aware)
Per-location, HSSD General 1 (shallow) and General 2 (deep, Alaska). Both breach fields are dangerous-direction rates, partitioned by the true correction's direction:

| Survey | subset | raw hazard (fp32 -> bf16) | TVU breach (fp32 -> bf16) |
|--------|--------|---------------------------|---------------------------|
| Shallow E00269 | shoal-target | 0.00% -> 0.00% | 0.00% -> 0.00% |
| Shallow E00269 | deep-target | 5.07% -> 31.10% | 0.07% -> 0.62% |
| Deep E00269 | shoal-target | 7.82% -> 11.03% | 0.00% -> 0.01% |
| Deep E00269 | deep-target | 28.96% -> 44.53% | 0.03% -> 0.12% |
| Alaska H14116 | shoal-target | 5.08% -> 15.27% | 0.00% -> 0.00% |
| Alaska H14116 | deep-target | 17.38% -> 48.82% | 0.05% -> 0.04% |

MAE improved with bf16 at all three locations (shallow 2.41 -> 0.89, deep 27.37 -> 22.13, Alaska 19.00 -> 17.28 m). Alaska recovery RMSE was the lone worse aggregate (71.68 -> 89.01 m), driven by a few large errors that largely stay within the deep-water budget (its breach rate did not rise).

### Interpretation
> **CORRECTION (2026-06-09 scrub):** every direction label in this entry
> (hazardous, shoal-target, deep-target, TVU breach) was measured under the
> inverted sign convention described in the entry above. The magnitudes and
> speed results stand; the direction-sensitive conclusions must be re-measured
> after the V11 retrain.

The raw `hazardous_error_rate` over-reported by one to two orders of magnitude because most dangerous-direction flips are smaller than the allowed TVU at depth. The shoal-critical subset (shoal-target dangerous breach) stayed at ~0% for both precisions; bf16's only measurable safety cost is a sub-0.7% rise in deep-target dangerous breaches. bf16 is defensible for the deliverable on this evidence. See LESSONS_LEARNED Lesson 19.

> **SUPERSEDED (2026-06-15):** the "shoal breach ~0 for both, bf16 defensible"
> conclusion was an artifact of the inverted sign convention. Under the corrected
> V11 measurement (top of this changelog), the shoal-target breach is not zero and
> bf16 raises it 13.5x at deep; bf16's apparent MAE advantage also did not reproduce.
> The current verdict is fp32 for the release, bf16 for experimentation only.

### Caveats
- One paired bf16/fp32 run; the two checkpoints selected different best epochs off a noisy validation signal, so single-run variance is not fully excluded. Two or three paired runs are needed to confirm the shoal-target breach stays at zero before bf16 is made permanent on the deliverable path.
- The General 1 / General 2 assignments are regime-based assumptions; the applicable OCS Quality Metric per survey is set in the Project Instructions.

---

## 2026-06-08 - First Cross-Geography Generalization Result (Multi-Location V10)

### Training Run
First V10 training on corrected multi-location data with a genuinely held-out geography.
- Training: E00269 (4 SR sub-files, Pacific Islands) + H13739 (VR, Pacific Islands) + H14070 (VR, Pacific NW)
- Validation: E00269 2of6 (shallow) + E00269 5of6 (deep) + H14116 (VR, Alaska, unseen geography)
- Resolution feature active (9 input channels), corrected VR ground truth, Huber delta 5.90
- Best val loss 1.51 at epoch 12, early stopping at epoch 26
- Runtime ~36 hours (VR surveys increased per-epoch cost to 56-78 min/epoch)

### Per-Location Evaluation (best_model.pt, evaluated separately by location)

| Metric | E00269 shallow (8m) | E00269 deep (128m) | H14116 Alaska (unseen) |
|--------|---------------------|--------------------|-----------------------|
| Overall MAE | 2.41m | 27.37m | 19.00m |
| Normalized MAE | 1.77 std | 1.20 std | 0.92 std |
| MAE <0.1m bucket | 1.70m | 27.15m | 7.43m |
| MAE 1-10m bucket | 4.27m | 24.22m | 16.03m |
| MAE >10m bucket | 24.07m | 38.18m | 84.48m |
| Shoal hazard rate | 0.00% | 7.82% | 5.08% |
| Recovery RMSE | 4.12m | 47.83m | 71.68m |

### Spatial Error Analysis on H14116 (unseen Alaska)
A per-cell error map (`scripts/spatial_error_map.py`) showed the H14116 failure is highly localized, not pervasive:
- The worst 1% of cells account for 87.6% of total squared error
- The worst 5% account for 96.5%; the remaining 95% of cells contribute only 3.5%
- Error rises only modestly with depth (14m mean in shallower bins to 25m in deepest), so depth is a contributor, not the cause
- The 15,848 cells with true corrections >=10m have mean error 86m and dominate the aggregate metrics
- QGIS visualization confirmed errors are scattered individual cells and tiny clusters, not a contiguous failure region

### Interpretation
The model generalizes to unseen Alaska for the common case. The bulk of the survey (clean seafloor and small corrections, ~95% of cells) is handled reasonably, and shoal protection held at a 5% hazard rate on geography never seen in training. The model's weakness is isolated large-magnitude corrections (10m+, mostly deeper water), which are scattered individual fliers/noise spikes where it mispredicts magnitude badly. Because RMSE and MAE are dominated by this small high-error population, the headline numbers look worse than the model's typical behavior.

This is the first concrete evidence of geographic generalization in the project. It also localizes the remaining weakness: the model needs more examples of large-magnitude deep-water noise to predict those specific cells, which is a data-scarcity issue rather than an architecture or geography problem.

### Caveats
- Validation sets are small (H14116 is 18 tiles / ~196K analyzed cells from one survey); this is a first signal, not a confident measurement
- No two runs to date differ by exactly one variable, so causal attribution between runs is limited
- The normalized MAE comparison across regimes carries a depth bias (deep water has a larger local_std denominator); absolute per-bucket numbers are more directly comparable
- Hazardous error rate tracks frequency, not magnitude; magnitude of hazardous errors on unseen geography is not yet quantified

### New Tool
- `scripts/spatial_error_map.py`: runs a model over one survey and writes a per-cell error GeoTIFF (target, predicted, error, abs_error, depth) plus prints error-concentration and error-vs-depth diagnostics. Distinguishes localized from pervasive failure.

---

## 2026-06-03 - Critical Fix: VR Ground Truth Warp Used Wrong Surface Interpretation

### The Bug
When clean and noisy surveys had different resolutions (which triggers the warp-to-align path in `prepare_ground_truth.py`), the noisy surface was loaded twice with inconsistent results:
1. First correctly via `load_survey` using `MODE=RESAMPLED_GRID` (the resampled refinement surface)
2. Then that result was discarded, and `warp_noisy_to_clean` re-opened the raw BAG with `gdal.Warp(tmp, str(noisy_path), ...)` which did NOT pass `MODE=RESAMPLED_GRID`

`gdal.Warp` on the raw VR BAG path falls back to GDAL's default VR interpretation (e.g. the low-resolution base grid), which is a fundamentally different surface than the resampled refinements. Differencing this against the clean resampled surface produced systematic offsets and one-sided direction splits.

### How It Was Found
Comparison against CARIS-derived difference exports (CSAR) revealed the BAG pipeline produced wildly wrong values for VR survey pairs:

| Survey | BAG pipeline median diff | CSAR median diff | BAG direction split | CSAR direction split |
|--------|--------------------------|------------------|---------------------|----------------------|
| H13739 | -0.34m (mean abs 21.25m) | -0.04m (mean abs 6.59m) | 51/49 | 51.8/48.2 |
| H14070 | 78.17m (mean abs 102.62m) | -0.01m (mean abs 1.17m) | 1.1/98.9 | 51.7/48.3 |
| H14116 | 43.62m (mean abs 61.03m) | -0.02m (mean abs 3.67m) | 0.3/99.7 | 51.1/48.9 |

Severity tracked how much the clean and noisy VR refinement structures differed. H13739 (similar structures) was mildly wrong: correct direction, magnitudes inflated 3x. H14070 and H14116 (very different structures) were severely wrong: 78m and 43m phantom offsets, 99% one-sided.

A separate diagnostic (`check_resampled_surface.py`) confirmed the resampled noisy surface alone matches CARIS perfectly (correlation -0.9999, differing only by sign convention: GDAL reads depth negative-down, CARIS positive-down). This proved the resampling itself was correct and isolated the bug to the warp re-opening the raw BAG.

### The Fix
Replaced `warp_noisy_to_clean(noisy_path, clean_grid)` with `warp_grid_to_reference(noisy_grid, clean_grid)`, which:
- Operates on the already-loaded, correctly-resampled `noisy_grid` in-memory array
- Writes it to a temporary GeoTIFF with its own geotransform
- Warps that GeoTIFF onto the clean grid

Both surfaces now stay in the resampled interpretation through the difference.

### Verification After Fix
All three VR surveys reprocessed and now match CSAR truth:

| Survey | Fixed median diff | Fixed mean abs | Fixed direction split |
|--------|-------------------|----------------|----------------------|
| H13739 | -0.13m | 7.87m | 52.4/47.6 |
| H14070 | -0.05m | 2.14m | 52.6/47.4 |
| H14116 | -0.01m | 6.07m | 50.2/49.8 |

Magnitudes run slightly above CSAR (GDAL vs CARIS resampling/gridding differ), but offsets are gone and direction splits are symmetric.

### Note on Sign Convention
GDAL reads these BAG depths as negative-down; CARIS exports positive-down. Because `prepare_ground_truth.py` loads both clean and noisy through the same GDAL path, the sign convention cancels in the subtraction. No sign flip is needed for the difference to be correct.

### Impact on Prior Results
- The earlier V10 training run that included H13739 used the corrupted (inflated) H13739 targets. That run's results are invalid as a measure of whether H13739 helped.
- All E00269 SR results are unaffected: SR BAGs do not trigger the warp path and were always loaded consistently.
- Three VR surveys (H13739 Pacific Islands, H14070 Pacific NW, H14116 Alaska) are now usable, providing the geographic diversity beyond E00269 that the project needed.

### New Diagnostic Tool
- `scripts/check_resampled_surface.py`: compares a resampled BAG surface against a CARIS XYZ export at matching locations. Useful for validating that a loaded surface matches the authoritative hydrographic software.

---

## 2026-05-28 - Per-Cell Resolution Feature (log_footprint)

### Added Resolution Conditioning Feature
- Added `log_footprint` node feature to `data/graph_construction.py`
- Encodes each cell's resolution (footprint) as log2 of the footprint in meters
- For SR surveys: constant per file (log2(4)=2, log2(8)=3, log2(128)=7, log2(256)=8)
- For VR surveys: will vary per cell once native resolution is preserved through loading (future work)
- Rationale: lets a single model condition its correction behavior on scale rather than needing separate models per resolution regime
- log2 chosen so equal resolution ratios map to equal feature distances (4->8 is the same step as 128->256)
- `FOOTPRINT_FLOOR = 0.1` guards against log of zero on malformed resolution values
- Input channels increased from 8 to 9; existing checkpoints not loadable (retrain required)

### Controlled Comparison Result (E00269, same train/val split)
Same data and split as the prior baseline; only the resolution feature was added.

Best validation loss improved from 1.64 to 1.31.

Shallow water (2of6, 8m) showed large improvement:
| Metric | Baseline | With feature |
|--------|----------|--------------|
| Overall MAE | 1.99m | 0.87m |
| MAE <0.1m bucket | 1.39m | 0.52m |
| MAE 0.1-1m bucket | 1.86m | 0.79m |
| MAE 1-10m bucket | 3.53m | 1.74m |
| Recovery RMSE | 3.40m | 1.56m |
| Recovery mean error | -2.16m | -0.94m |
| Hazardous (shoal) | 0.00% | 0.00% |

Deep water (5of6, 128m) showed marginal improvement only:
| Metric | Baseline | With feature |
|--------|----------|--------------|
| Overall MAE | 28.09m | 26.59m |
| Recovery RMSE | 46.81m | 40.05m |
| Recovery mean error | -23.76m | -23.53m |

Deep water per-bucket MAE remained flat (~24m across all magnitude buckets), indicating the model still outputs a default magnitude rather than discriminating. This points to deep water being a data limitation (one 128m survey) rather than something the feature alone can fix.

### Interpretation
- The feature helped where the hypothesis predicted: the shallow regime with enough signal to learn from
- Shallow MAE roughly halved with no cost to shoal safety (still 0.00% shoal hazard)
- Deep water still needs more training examples, not architecture changes
- Supports staying with a single conditioned model rather than splitting by regime
- One side effect to watch: deep-direction hazard rate in shallow water rose (7.32% to 31.28%); shoal protection unaffected

---

## 2026-05-21 - Huber Delta Computation Fixed for Normalized Corrections

### Bug Identified in V10 Multi-File Training
- First V10 training run on 5 E00269 sub-files reported Huber delta of 281.7m
- Training proceeded but with erratic validation loss (range 5.98 to 18.71 across 9 epochs)
- Root cause: `_compute_training_stats` was computing delta from raw correction magnitudes in meters
- Model trains on normalized corrections (divided by local_std, clipped to +/-50 std-devs)
- A delta of 281 put the Huber loss in pure linear mode for the entire run, effectively becoming MAE
- The quadratic gradient signal that Huber provides for small errors was completely absent

### Fix
- Updated `_compute_training_stats` to sample 50 random tiles from the dataset
- Builds graphs for sampled tiles and collects normalized correction targets
- Computes 95th percentile from the same normalized values the model sees during training
- Adds 99-second startup cost but eliminates unit-mismatch risk
- Typical post-fix delta on E00269 data: 3-10 std-devs

### Validation Strategy Improvement
- Previous runs used single-file validation (sub-file 3of6 alone), which only validated shallow water performance
- New approach uses `--val-surveys` flag to point at a separate folder with multiple files
- Current recommended split: train on 1of6 + 3of6 + 4of6 + 6of6; validate on 2of6 (shallow) + 5of6 (deep)
- Validation now spans both shallow and deep regimes, giving a more honest signal about generalization

### Documentation Updates
- New "Huber Loss and the Delta Parameter" section in HOW_IT_WORKS.md
- Lesson 15 added to LESSONS_LEARNED.md
- Troubleshooting entries added to QUICK_REFERENCE.md

---

## 2026-05-20 - V10 First Training Run Operational

### First V10 Training Run on Regression-Mode Data
- Trained V10 on E00269 sub-file 1of6 (regression-mode ground truth)
- Parameters: 5 epochs, batch size 2, tile size 256, 91 tiles from 1 survey
- Training loss decreased steadily: 0.79 -> 0.74 -> 0.71 -> 0.70 -> 0.69
- MAE (normalized std-dev units) stabilized at ~0.83
- No NaN values, no divergence, gradient flow healthy
- Pipeline confirmed working end-to-end in regression mode

### E00269 Sub-Files Processed (Regression Mode)
| Sub-file | Resolution | Valid Cells | Mean Correction | Max Correction | Offset Removed |
|----------|------------|-------------|-----------------|----------------|----------------|
| 1of6 | 4m SR | 918,831 | 0.30m | 16.49m | -0.51m |
| 2of6 | 8m SR | 356,601 | 0.53m | 88.00m | -0.20m |
| 3of6 | 8m SR | 3,214,339 | 0.50m | 605.64m | -0.41m |
| 4of6 | 128m SR | 11,390,629 | 83.84m | 4934.40m | +5.91m |
| 5of6 | 128m SR | 1,118,639 | 6.77m | 775.70m | +0.42m |
| 6of6 | 256m SR | 59,793 | 17.97m | 552.44m | -0.15m |

- Correction magnitudes scale with resolution as expected; local_std normalization handles cross-regime training
- Variable offsets across sub-files were initially flagged as datum issues; confirmed all are MLLW; offsets are real signal in survey datum vs MLLW separation
- 50/50 shoal/deep split is normal for CUBE re-grid differences (not a quality warning, contrary to earlier interpretation)

### V10 Architecture Shift: Regression Mode
- Added `--regression-mode` flag to `prepare_ground_truth.py`
- Output bands changed in regression mode: band 1 = valid_mask (1=valid, 0=invalid), band 2 = correction target (meters, continuous, no threshold applied)
- Output file naming: `{survey}_regression.tif` (vs `{survey}_ground_truth.tif`) so both modes can coexist
- Added `RegressionLoss` class to `losses.py` with asymmetric Huber penalty
  - Shoal safety baked into loss direction: predictions leaving the corrected surface deeper than reality penalized 3x
  - Sign math: `corrected = noisy - predicted_correction`; error < 0 means corrected depth > true depth = navigation hazard
  - Applies to every valid cell, not just cells flagged as noise
- `BathymetricGNNLoss.forward()` dispatches on `targets['mode']`; existing classification path unchanged
- `GroundTruthDataset` auto-detects mode from band 1 description; tiles carry mode flag
- Training loop tracks MAE in regression mode, accuracy in classification mode
- Backwards compatible: existing classification workflow works exactly as before

### Code Changes
- `scripts/prepare_ground_truth.py`: +312/-77 lines
  - `--adaptive-threshold` flag (Otsu's method on log-scaled absolute differences)
  - `--no-offset` flag (skip median offset removal when both surfaces share a datum)
  - `--regression-mode` flag (skip thresholding, emit continuous correction targets)
  - GDAL warp for resolution mismatch between VR BAGs with different refinement structures
  - Nodata fix: read actual `grid.nodata_value` plus `abs(depth) < 1e5` safety check (catches both +/-1e6 sentinels)
- `training/losses.py`: Added `RegressionLoss` class, dispatched forward by mode, restored `compute_correction_delta` and `correction_delta` parameter from V8
- `training/trainer.py`:
  - Mode-aware dataset (auto-detect from band 1 description)
  - V9 local_std normalization preserved for both modes
  - `_compute_training_stats` now handles both modes
  - MAE metric for regression mode; history tracking handles both
  - `CORRECTION_NORM_FLOOR` and `CORRECTION_NORM_CAP` constants made explicit
- `scripts/train.py`: Picks up both `_ground_truth.tif` and `_regression.tif` files

---

## 2026-05-19 - V10 Regression Mode Design

### Architectural Rationale
- Identified structural problem with classification approach: forces binary decision on continuous signal
- Cells near threshold boundary get inconsistent labels for nearly identical real-world cases
- Correction head only trains on cells labeled noise, never learns to predict near-zero corrections
- Threshold value determines what model learns, but threshold is arbitrary even when adaptive
- Mixed depth regimes (shallow vs deep water) need different threshold values, creating inconsistent labeling across the training set

### Key Insight
- The difference between the clean and dirty CUBE surfaces IS the training signal at every cell
- A cell with a 0.01m difference and a cell with a 30m difference are both informative
- Regression preserves the full continuous signal; classification discards it
- Navigation safety use case is better served by predicting magnitudes than by binary classification
- Hydrographers want to know "how wrong is this cell" not "is this cell noise"

### Decision
- Pursue regression as parallel implementation (V10) while keeping V9 classification working
- Build on existing model architecture (still outputs correction head); change only what's needed in dataset, loss, and training loop
- Validate on E00269 first since data is available locally
- Defer model architecture simplification (removing unused classification head) until V10 proven

---

## 2026-05-18 - Ground Truth Preparation for New Survey Pairs

### H13739 Processed (Pacific Islands VR)
- Clean and noisy BAGs both at 16m resolution after VR resampling (different refinement structures: 16.078m vs 16.001m finest)
- Required GDAL warp to align grids before differencing
- Mean correction 32.7m, max 1304m (deep water Pacific)
- Both surfaces in MLLW; `--no-offset` flag used
- 61% noise cells / 39% seafloor at 6.73m adaptive threshold
- Noise concentrated along sparse trackline coverage, as expected for deep water surveys

### E00269 1of6 Processed (Pacific Islands SR, Northern Mariana Islands)
- 4m resolution single-resolution BAG
- Initially produced high noise percentages (67-99%) with fixed 0.15m threshold
- Investigation revealed variable systematic offsets (-0.51m to +5.91m) across sub-files
- Clean BAGs explicitly labeled MLLW; dirty BAGs in survey datum
- 4m sub-file with offset removal enabled: 75% noise at 0.117m adaptive threshold, seafloor mean diff = 0.000m (offset removal worked correctly)

### Diagnostic Script Developed (check_pair.py)
- Quick pre-check for clean/noisy BAG pair quality
- Loads both surfaces, computes difference, runs noise statistics
- Initially used too-rigid pass/fail logic from Seward assumptions; lessons led to insight that classification thresholds don't generalize across depth regimes
- Script abandoned in favor of running `prepare_ground_truth.py` directly with appropriate flags

### Adaptive Threshold Development
- Otsu's method applied to log-scaled absolute differences
- Adapts naturally across depth regimes (0.12m in shallow Pacific, 6.73m in deep Pacific)
- Replaces the requirement to pick a fixed threshold per survey
- Implementation in `compute_adaptive_threshold()` in `prepare_ground_truth.py`

### Datum Diagnostic Insights
- Variable median offsets across sub-files of the same survey indicates datum mismatch, not simple processing offset
- Survey datum to MLLW separation varies spatially with geoid model and tidal zoning
- Different sub-files cover different areas, so each gets a different offset
- This isn't a bug in the data; it's why the offset removal flag is needed

### CUBE Re-Grid Behavior Documented
- When the clean and noisy point clouds differ (noise cleaning removes outliers), CUBE re-gridding produces pervasive cell-to-cell differences, not just isolated spikes
- Every cell that had any outlier sounding in its weighted contribution changes
- Cell-to-cell differences are the training signal, not an artifact to filter out
- The Seward training data worked differently because manual edits were applied directly to the grid surface after CUBE ran, leaving unchanged cells identical
- 50/50 shoal/deep split is normal for re-grid differences; not a quality warning

---

## 2026-03-04 - Data Acquisition Plan for Geographic Diversity

### Survey Identification
- Identified 22 surveys across 8 regions for training data expansion
- Regions: Gulf Coast (3), SE Atlantic (2), Mid-Atlantic (3), Northeast (1), Great Lakes (2), Pacific NW (2), Alaska (6), Pacific Islands (3)
- Alaska surveys include diverse acquisition types: set line spacing in shallow flat water, Bering Sea/North Slope trackline, standard multibeam
- E00269 (Northern Mariana Islands) available locally for immediate processing

### Archive Request
- Clean BAGs downloadable directly from NCEI for 21 of 22 surveys
- Processed (pre-cleaning) data requested from NCEI archive to produce noisy BAGs
- Expected delivery: days to weeks
- Processing plan: one survey per region first to detect unusable pairs early

### Expected Outcome
- At 30-50% attrition, expect 11-15 usable pairs
- Combined with 4 existing Seward pairs: 15-19 total from 8+ environments
- Primary mitigation for persistent overfitting observed in V5-V9

---

## 2026-03-02 - Training Data Diversity Investigation

### Data Pair Evaluation
- Tested 3 new survey pairs for training data diversity:
  - H13532: Florida river survey (SR BAG, 1m) - 4 noise cells / 456K valid (0.00%)
  - H14190: Coastal Alaska (VR BAG, 1m) - 149 noise cells / 30M valid (0.00%)
  - F00889: Norfolk river survey (SR BAG, 0.5m) - 1 noise cell / 23M valid (0.00%)
- All three pairs produced near-zero noise because differences don't propagate to gridded surfaces
- `prepare_ground_truth.py` confirmed working for both SR and VR BAGs (auto-detection via BathymetricLoader)
- Decision: Do not include these pairs in training; find pairs with grid-visible noise instead

### Lesson Documented
- Added data quality verification step to workflow: visually inspect difference layer in QGIS before running ground truth preparation
- Training data with near-zero noise would shift class balance from 75/25 to 97/3, risking majority-class collapse

---

## 2026-02-27 - V7/V8/V9 Training Runs & Correction Normalization

### V7: Boundary-Aware Feature Computation
- **Root cause identified (V6 failure):** `scipy.ndimage.uniform_filter` with `mode='nearest'` bled nodata values (1e6) into local statistics at survey boundaries, creating artificial feature spikes
- **Fix:** Replaced with masked local statistics using only valid neighbors; nodata filled with local mean before gradient/curvature computation
- **Results:** Noise detection jumped from 3.3% to 34.8%; peak val accuracy ~72% with meaningful noise detection (confirming genuine classification, not majority-class collapse); 11,790 auto-corrections applied; mean confidence 0.825. Note: 34.8% detection against the validation set's 18.8% actual noise rate indicates significant over-prediction (false positives), likely from Seward-specific pattern memorization.
- **Visual validation:** QGIS confirmed noise classifications follow actual noise spatial patterns, not survey boundaries

### V8: Dynamic Huber Delta (No Effect)
- Added data-derived Huber delta computation from correction target distribution
- 95th percentile of raw corrections (0.652m) was below min_delta floor (1.0)
- Training dynamics identical to V7; no inference run needed
- Documented adaptive delta Options 2/3 in `losses.py` for future implementation

### V9: Local Standard Deviation Correction Normalization
- **Problem:** V7 predicted 0.4m corrections where 70m was needed; Huber loss gradient plateau above delta means model cannot distinguish large from small corrections
- **Solution:** Normalize correction targets by per-node local_std from graph construction
  - `graph_construction.py`: `_compute_node_features` returns `(features, local_std)` tuple; `build_graph` stores `data.local_std`
  - `trainer.py`: Both dataset classes divide correction targets by `max(local_std, 0.01m)`, clamp to +/-50 std devs
  - `inference_native.py`: Denormalize by multiplying predicted corrections by local_std
- Constants: `CORRECTION_NORM_FLOOR = 0.01`, `CORRECTION_NORM_CAP = 50.0`
- **Initial extreme value problem:** Max normalized correction of 2,692 from noise spikes in flat areas; solved by capping at +/-50 std devs
- **V9 training results:** Best val loss 1.813 (epoch 5), early stopping at epoch 19, classification identical to V7 (34.8% noise, 0.825 confidence), but corrections up to 32m larger than V7

### Code Changes
- `data/graph_construction.py`: Added local_std output from node feature computation
- `training/trainer.py`: Added correction normalization in both dataset classes and stats computation
- `scripts/inference_native.py`: Added denormalization step in `NativeVRProcessor.process_grid`
- `training/losses.py`: Added documentation for adaptive Huber delta options (unchanged functionally)

---

## 2026-02-27 - V5/V6 Training Iterations

### V5: Class Weight Bug Discovery
- Training without class weights resulted in model predicting seafloor for all cells
- Val accuracy stabilized at ~67% with no learning progression (majority-class-like collapse)
- Inference produced 0% noise detection with 0.967 confidence

### V6: Auto Class Weight Implementation
- `trainer.py` now scans training tiles to count class distributions automatically
- Computes inverse-frequency weights with smoothing: `weight = total / (n_classes * count + smooth)`
- Broke the all-seafloor pattern, noise detection at 3.3%
- However, visual validation revealed boundary artifact problem (see V7 above)

---

## 2026-02-12 - Unified Native BAG Processing & Documentation Updates

### Major Changes

#### Unified Native BAG Inference
- `inference_native.py` now handles BOTH VR and SR BAGs automatically
- Added `detect_bag_type()` function to auto-detect BAG type
- Added `SRBagHandler` and `SRBagWriter` classes for SR BAG native processing
- Single script for all native BAG processing simplifies user workflow

#### Auto-Detection in BathymetricLoader
- `BathymetricLoader` now auto-detects SR vs VR BAGs before loading
- SR BAGs are loaded directly, ignoring `--vr-bag-mode` setting
- Eliminates "No supergrids available" errors when loading SR BAGs with default settings

### Bug Fixes

#### Critical: Correction Sign Error in Native Inference Scripts
- Fixed correction application direction in `scripts/inference_native.py`
- **Bug**: Script was using `+= correction` which would double noise instead of removing it
- **Fix**: Changed to `-= correction` to match `models/pipeline.py` behavior
- The model predicts `correction = noisy_depth - clean_depth`, so recovery requires `clean = noisy - correction`

#### SR BAG Metadata Parsing
- Fixed `SRBagHandler` failing on SR BAGs with array-based metadata
- Metadata stored as numpy array of bytes now correctly converted via `tobytes()`

### Documentation Updates
- Simplified "Architecture" section in README to plain language "How It Works"
- Clarified that ALL classification classes (0, 1, 2) contribute to training
- Updated training strategy to emphasize real data pairs over synthetic noise
- Clarified `inference_native.py` handles both VR and SR BAGs
- Added "Practical Usage" section to README with workflow position
- Added "Documentation" section with links to HOW_IT_WORKS.md and TRAINING_PLAN.md
- Expanded "Current State" in TRAINING_PLAN.md with milestone table
- Added "Training Terminology" section clarifying epochs vs iterations vs surveys
- Added "Ground Truth Acquisition Options" and "Why Train on Diverse Data" sections
- Added "Protecting Uncharted Features" section to HOW_IT_WORKS.md
- Added "Processing Time" and "Operational Confidence Thresholds" sections
- Fixed node feature description (was "roughness", now "local statistics, gradients, curvature")
- Fixed edge feature description (was "slope direction", now "slope angle")
- Clarified uncertainty scaling only applies to native processing scripts
- Clarified feature class training is Phase 3 (planned), not currently active

## 2025-12-10 - Native VR BAG Support & Pipeline Fixes

### Major Features

#### Native VR BAG Processing
- Added `data/vr_bag.py` module for handling Variable Resolution BAGs without resampling
- New `scripts/inference_native.py` for native VR inference that preserves multi-resolution structure
- `VRBagHandler` class for reading VR BAG refinement grids
- `VRBagWriter` class for copy-and-modify workflow
- `SidecarBuilder` class for generating GeoTIFF outputs from native VR processing
- Processes each refinement grid (3x3 to 50x50 cells) individually through the GNN
- Much higher confidence scores (73.5% vs 4%) compared to resampled approach

#### Training Plan & Scripts
- Added `docs/TRAINING_PLAN.md` with comprehensive training methodology
- New `scripts/prepare_ground_truth.py` for creating labels from clean/noisy pairs
- New `scripts/evaluate_model.py` for measuring model performance against ground truth
- New `scripts/analyze_noise_patterns.py` for understanding real noise characteristics

#### VR BAG Loading Improvements
- Added `--vr-bag-mode` argument to control VR BAG loading:
  - `resampled` (default): Uses GDAL's MODE=RESAMPLED_GRID for uniform output
  - `refinements`: Direct refinement subdataset access
  - `base`: Base grid only (not recommended)
- Fixed VR BAGs loading as tiny grids (was 512x512, now 25369x25369 at 1m)

### Bug Fixes

#### Tile Merging Fix
- Fixed `TileMerger` NaN initialization bug in `data/tiling.py`
- Weighted averaging was failing because NaN + anything = NaN
- Now properly initializes first writes to 0.0 before accumulating

#### Unprocessed Valid Data Preservation
- Valid cells in sparse tiles (below min_valid_ratio) now preserved in output
- Previously these cells were lost, causing coverage gaps
- Now marked as class 0 (seafloor) with confidence 0 (not analyzed)

#### PyTorch 2.6+ Compatibility
- Fixed `torch.load()` requiring `weights_only=False` for checkpoint loading
- Updated `config.py` to handle tuple/list conversion for YAML serialization

#### RTX 50 Series (Blackwell) GPU Support
- Auto-detect unsupported GPU and fall back to CPU gracefully
- Added to all scripts: inference.py, train.py, test_pipeline.py, pipeline.py, trainer.py

#### DLL Conflict Resolution (Windows)
- Added `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` to handle OpenMP conflicts
- Fixed import order: torch must be imported before numpy on Windows

### New Files
- `data/vr_bag.py` - Native VR BAG handler
- `scripts/inference_native.py` - Native VR inference script
- `scripts/diagnose_tiles.py` - Diagnostic tool for tile validity issues
- `scripts/explore_vr_bag.py` - VR BAG structure explorer

### Modified Files
- `data/__init__.py` - Added VR BAG exports
- `data/loaders.py` - VR BAG mode support, BAG writer improvements
- `data/tiling.py` - Fixed tile merging bug
- `config/config.py` - YAML serialization fixes
- `models/pipeline.py` - VR mode support, valid_mask band, correction band output
- `training/trainer.py` - VR mode support, GPU compatibility
- `scripts/inference.py` - Added --vr-bag-mode, --min-valid-ratio arguments
- `scripts/train.py` - Added --vr-bag-mode argument
- `scripts/test_pipeline.py` - GPU compatibility fixes

### Output Format Changes
- GeoTIFF output now includes 6 bands:
  1. Depth (cleaned)
  2. Uncertainty (from original)
  3. Classification (0=seafloor, 1=feature, 2=noise)
  4. Confidence (0-1, where 0=not analyzed)
  5. Correction (suggested depth adjustment)
  6. Valid_mask (1=valid, 0=nodata)

### Uncertainty Scaling
- Corrected cells have uncertainty scaled by model confidence:
  - High confidence (0.9) -> uncertainty x 1.1
  - Low confidence (0.5) -> uncertainty x 1.5
  - Formula: `scale_factor = 2.0 - confidence`

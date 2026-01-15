# Bathymetric GNN Training Plan

A step-by-step guide for training a production-quality bathymetric noise detection model.

## Current State

| Metric | Value |
|--------|-------|
| Training data | Synthetic noise only |
| Training surveys | 5 clean VR BAGs |
| Epochs completed | 50 |
| Synthetic accuracy | 63.2% |
| Real-world accuracy | Unknown |
| Mean confidence | 73.5% (native VR processing) |

**Next milestone:** Establish baseline metrics with real ground truth data.

---

## Overview

| Phase | Goal | BAGs Needed | Scripts |
|-------|------|-------------|---------|
| 1 | Establish ground truth | 6-10 min, 16-30 target (pairs) | `prepare_ground_truth.py`, `evaluate_model.py` |
| 2 | Add feature examples | 5-10 feature-area BAGs | `extract_s57_features.py` |
| 3 | Train on real data | (uses Phase 1-2 labels) | `train.py` (needs extension) |
| 4 | Deploy and refine | Unlimited | `inference_native.py` |

**Minimum to start:** 11-20 BAGs (3-5 ground truth pairs + 5-10 feature surveys)  
**Target for v1 model:** 21-40 BAGs (8-15 ground truth pairs + 5-10 feature surveys)

---

## Processing Mode

| Mode | Script | BAG Types |
|------|--------|-----------|
| **Native BAG** | `inference_native.py` | VR and SR (auto-detected) |
| Resampled | `inference.py` | Any (loses VR structure) |

**Always use native BAG processing for training and production inference.**

The native script automatically detects BAG type:
- **VR BAGs:** Iterates through refinement grids, preserving multi-resolution structure
- **SR BAGs:** Processes the full grid directly at native resolution

---

## Phase 1: Establish Ground Truth

**Duration:** 2-4 weeks (ongoing)  
**Goal:** Create labeled data from real clean/noisy survey pairs

### Data Requirements

Ground truth pairs are the highest-quality training data available. They directly show "this is noise, this is not" without assumptions. The limiting factor is availability - someone must manually clean a survey or you need repeat surveys of the same area.

| Category | Pairs | BAGs | Assessment |
|----------|-------|------|------------|
| Too few | 1-2 | 2-4 | Metrics unreliable, high overfitting risk |
| **Minimum to start** | 3-5 | 6-10 | Can establish baseline, limited diversity |
| **Target for v1** | 8-15 | 16-30 | Good coverage, statistically meaningful |
| Diminishing returns | 20+ | 40+ | Still useful, incremental gains decrease |

**Start with 3-5 pairs to prove the workflow, then accumulate toward 8-15 for production training.**

### Diversity Matters

More important than raw count is coverage across conditions:

| Dimension | Examples | Why It Matters |
|-----------|----------|----------------|
| Depth | Shallow (<20m), mid (20-100m), deep (>100m) | Noise characteristics vary with depth |
| Seafloor | Flat mud, sand waves, rocky, slopes | Model must learn what's "normal" for each |
| Noise severity | Light speckle, moderate clusters, heavy | Confidence calibration across conditions |
| Equipment | Different sonars, configurations | Generalization to new surveys |
| Geography | Different regions, water conditions | Avoid overfitting to one area |

**Ideal mix for 8-15 pairs:**
- 2-3 shallow coastal
- 3-4 mid-depth continental shelf
- 2-3 deep water
- At least 2 with rocky/complex terrain
- At least 2 with heavy noise contamination

### Step 1.1: Organize Survey Pairs

```
data/raw/clean/
├── survey_001_clean.bag
├── survey_002_clean.bag
├── survey_003_clean.bag
...

data/raw/noisy/
├── survey_001_noisy.bag
├── survey_002_noisy.bag
├── survey_003_noisy.bag
...
```

### Step 1.2: Generate Ground Truth Labels

Run for each survey pair:

```cmd
python scripts/prepare_ground_truth.py ^
    --clean data/raw/clean/survey_001_clean.bag ^
    --noisy data/raw/noisy/survey_001_noisy.bag ^
    --output-dir data/processed/labels ^
    --noise-threshold 0.15

:: Repeat for all pairs...
```

**Output per survey:**
- `survey_001_ground_truth.tif` - 4-band GeoTIFF (labels, difference, noisy_depth, clean_depth)
- `survey_001_ground_truth_stats.json` - Noise distribution statistics

### Step 1.3: Evaluate Current Model Against Ground Truth

Run inference and evaluation for each noisy survey:

```cmd
:: Run inference
python scripts/inference_native.py ^
    --input data/raw/noisy/survey_001_noisy.bag ^
    --model outputs/final_model.pt ^
    --output outputs/predictions/survey_001_predicted.bag ^
    --min-valid-ratio 0.01

:: Evaluate against ground truth
python scripts/evaluate_model.py ^
    --ground-truth data/processed/labels/survey_001_ground_truth.tif ^
    --predictions outputs/predictions/survey_001_predicted_gnn_outputs.tif ^
    --output outputs/metrics/survey_001_eval.json

:: Repeat for all pairs...
```

### Step 1.4: Review Baseline Metrics

After processing pairs, collect metrics:
- Per-class precision/recall/F1 (seafloor, noise, feature)
- Confusion matrices
- Confidence calibration curves

---

## Phase 2: Feature Class Training

**Duration:** 2-3 weeks  
**Goal:** Add examples of real seafloor features (wrecks, rocks, obstructions)

### Data Required

| Type | Count | Description |
|------|-------|-------------|
| Feature-area VR BAGs | 5-10 | Surveys in areas with known features |
| **Additional BAGs** | **5-10** | These are NEW surveys, not pairs |

**Note:** These do NOT need to be clean/noisy pairs. Any survey in an area with known charted features works.

**Recommended survey locations:**
- 2-3 near harbors (high wreck/obstruction density)
- 2-3 in rocky coastal areas (underwater rocks)
- 2-4 in areas with charted wrecks

### Step 2.1: Extract Features from NOAA ENC Direct

The script queries ENC Direct REST API automatically - no download required.

| Source | URL | Update Frequency |
|--------|-----|------------------|
| ENC Direct | `encdirect.noaa.gov` | Weekly |

| Feature Type | S-57 Code | Default Radius |
|--------------|-----------|----------------|
| Shipwrecks | WRECKS | 50m |
| Obstructions | OBSTRN | 30m |
| Underwater rocks | UWTROC | 25m |

### Step 2.2: Generate Feature Labels

Run for each of the 5-10 feature-area surveys:

```cmd
:: Generate labels and GeoJSON for visualization
python scripts/extract_s57_features.py ^
    --survey data/raw/features/harbor_001.bag ^
    --labels data/processed/labels/harbor_001_features.tif ^
    --output data/processed/labels/harbor_001_features.geojson

python scripts/extract_s57_features.py ^
    --survey data/raw/features/rocky_coast_001.bag ^
    --labels data/processed/labels/rocky_coast_001_features.tif ^
    --output data/processed/labels/rocky_coast_001_features.geojson

:: Repeat for all 5-10 feature-area surveys...
```

### Step 2.3: Verify Feature Coverage

Open the GeoJSON files in QGIS to verify:
- Features are correctly located within survey bounds
- Radii are appropriate for survey resolution
- No obvious charted features are missing

---

## Phase 3: Train on Real Data

**Duration:** 2 weeks  
**Goal:** Train model using real ground truth and feature labels

### Data Required

| Type | Count | Source |
|------|-------|--------|
| Ground truth labels | 3-5 | Phase 1 |
| Feature labels | 5-10 | Phase 2 |
| **Total labeled surveys** | **8-15** | |

### Step 3.1: Organize Train/Validation Split

```
data/processed/
├── train/                              # 70-80% of surveys
│   ├── survey_001_ground_truth.tif
│   ├── survey_002_ground_truth.tif
│   ├── survey_003_ground_truth.tif
│   ├── harbor_001_features.tif
│   ├── harbor_002_features.tif
│   ├── rocky_coast_001_features.tif
│   ├── rocky_coast_002_features.tif
│   └── wreck_area_001_features.tif
│
└── validation/                         # 20-30% of surveys (held out)
    ├── survey_004_ground_truth.tif     # At least 1 ground truth
    ├── survey_005_ground_truth.tif
    └── harbor_003_features.tif         # At least 1 feature survey
```

**Recommended split:**
- Training: 6-12 surveys
- Validation: 2-3 surveys (include both ground truth and feature types)

### Step 3.2: Train Model

> **Note:** The current `train.py` uses synthetic noise only. To train on real labels, the script needs to be extended to accept label files. This is a development task.

#### Development Tasks for `train.py` Extension

When extending `train.py` for real labels, also refactor BAG handling:

| Task | Description |
|------|-------------|
| Add `--labels` argument | Accept ground truth and feature label files |
| Native BAG iteration | Replace GDAL resampling with native VR/SR iteration from `vr_bag.py` |
| Consolidate BAG code | Move BAG-specific code from `loaders.py` to `bag.py` (rename `vr_bag.py`) |
| Keep `loaders.py` | Retain for GeoTIFF/ASC formats only |

This refactor improves training quality (native resolution) and simplifies the codebase.

Current training (synthetic noise only):

```cmd
python scripts/train.py ^
    --clean-surveys data/raw/clean ^
    --output-dir outputs/models/v2 ^
    --epochs 100 ^
    --batch-size 4
```

Future training (with real labels - requires script update):

```cmd
:: TODO: Extend train.py to support --labels argument
python scripts/train.py ^
    --clean-surveys data/raw/clean ^
    --labels data/processed/train ^
    --val-labels data/processed/validation ^
    --output-dir outputs/models/v2 ^
    --epochs 100
```

### Step 3.3: Evaluate on Validation Set

Run inference and evaluation on each held-out validation survey:

```cmd
python scripts/inference_native.py ^
    --input data/raw/noisy/survey_004_noisy.bag ^
    --model outputs/models/v2/best_model.pt ^
    --output outputs/predictions/survey_004_predicted.bag

python scripts/evaluate_model.py ^
    --ground-truth data/processed/validation/survey_004_ground_truth.tif ^
    --predictions outputs/predictions/survey_004_predicted_gnn_outputs.tif ^
    --output outputs/metrics/survey_004_val_eval.json
```

---

## Phase 4: Deployment & Refinement

**Duration:** Ongoing  
**Goal:** Deploy model and continuously improve from feedback

### Step 4.1: Production Inference

```cmd
python scripts/inference_native.py ^
    --input new_survey.bag ^
    --model outputs/models/v2/best_model.pt ^
    --output cleaned_survey.bag ^
    --min-valid-ratio 0.01
```

**Outputs:**
- `cleaned_survey.bag` - Corrected BAG (same format as input: VR or SR)
- `cleaned_survey_gnn_outputs.tif` - Sidecar with classification, confidence, corrections

### Step 4.2: Review Low-Confidence Regions

Identify uncertain regions for human review:
- Regions with confidence between 0.4-0.7
- Cluster size > 100 cells

Use the sidecar GeoTIFF confidence band to identify areas needing review.

### Step 4.3: Incorporate Feedback

1. Human reviewers correct model outputs in uncertain regions
2. Save corrections as new ground truth labels
3. Add to training set
4. Retrain model periodically (monthly or after accumulating corrections)

---

## Future: Architecture Evaluation

**When to evaluate:** After establishing real-data baseline metrics in Phase 3.

The current model uses GAT (Graph Attention Network). This is reasonable but may not be optimal for grid-based bathymetric data.

### Current Architecture

| Component | Implementation | Notes |
|-----------|----------------|-------|
| GNN type | GAT with 4 attention heads | Learns neighbor importance |
| Edge features | Distance, depth difference, gradient | Used by attention mechanism |
| Local features | MLP on node attributes | Processes each node independently first |

### Why GAT May Not Be Optimal

| Strength | Weakness |
|----------|----------|
| Learns to weight neighbors differently | Designed for irregular graphs, overkill for regular grids |
| Supports edge features | Attention learned from features alone, not spatial position |
| Can learn "ignore noisy neighbors" | More parameters than needed |

### Alternatives to Evaluate

| Architecture | Potential Benefit | When to Try |
|--------------|-------------------|-------------|
| **Hybrid CNN + GNN** | CNN excels at local patterns, GNN adds context | If GAT plateaus below target |
| **EdgeConv** | Explicitly uses edge geometry | If attention weights show no meaningful patterns |
| **Simpler GCN** | Faster, fewer parameters | If GAT is too slow for production |

### Hybrid CNN + GNN Concept

```
Input Grid
    ↓
CNN layers (detect local noise patterns: spikes, texture)
    ↓
GNN layers (aggregate spatial context: isolated vs connected)
    ↓
Classification head
```

This separates two tasks:
1. **Local pattern detection** (CNN strength): "Does this neighborhood look noisy?"
2. **Contextual reasoning** (GNN strength): "Is this spike isolated or part of a formation?"

### Decision Criteria

Revisit architecture if:
- Real-label training plateaus below 85% noise precision
- Attention weights show no meaningful patterns (all neighbors weighted equally)
- Inference is too slow for production (>1 min per survey)

**Do not change architecture until real-data baseline is established.** The current 63% accuracy on synthetic data is not a reliable indicator of architecture performance.

---

## Quick Reference

### Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `prepare_ground_truth.py` | Generate labels from clean/noisy pairs | Ready |
| `evaluate_model.py` | Compare predictions to ground truth | Ready |
| `extract_s57_features.py` | Extract features from ENC Direct | Ready |
| `train.py` | Train model (synthetic noise) | Needs extension for real labels |
| `inference_native.py` | Run inference preserving VR structure | Ready |
| `analyze_noise_patterns.py` | Characterize noise (optional, for research) | Ready |

### Command Sequence

```cmd
:: PHASE 1: Ground Truth (run for each of 3-5 pairs)
python scripts/prepare_ground_truth.py --clean X_clean.bag --noisy X_noisy.bag --output-dir data/processed/labels
python scripts/inference_native.py --input X_noisy.bag --model outputs/final_model.pt --output outputs/predictions/X.bag
python scripts/evaluate_model.py --ground-truth X_ground_truth.tif --predictions X_gnn_outputs.tif --output X_eval.json

:: PHASE 2: Feature Labels (run for each of 5-10 feature surveys)
python scripts/extract_s57_features.py --survey X.bag --labels X_features.tif --output X_features.geojson

:: PHASE 3: Training
python scripts/train.py --clean-surveys data/raw/clean --output-dir outputs/models/v2 --epochs 100

:: PHASE 4: Production
python scripts/inference_native.py --input new_survey.bag --model outputs/models/v2/best_model.pt --output cleaned.bag
```

---

## Data Requirements Summary

| Phase | Minimum | Target | Notes |
|-------|---------|--------|-------|
| 1 | 6-10 BAGs (3-5 pairs) | 16-30 BAGs (8-15 pairs) | Clean/noisy pairs, diverse conditions |
| 2 | 5-10 BAGs | 5-10 BAGs | Standalone feature-area surveys |
| 3 | 0 | 0 | Uses Phase 1+2 labels |
| 4 | Unlimited | Unlimited | Production surveys, ongoing feedback |

| Milestone | Ground Truth Pairs | Feature Surveys | Total BAGs |
|-----------|-------------------|-----------------|------------|
| **Minimum to start** | 3-5 | 5-10 | 11-20 |
| **Target for v1** | 8-15 | 5-10 | 21-40 |

---

## Target Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Noise Precision | > 0.90 | Unknown |
| Noise Recall | > 0.85 | Unknown |
| Feature Precision | > 0.80 | Unknown |
| Feature Recall | > 0.70 | Unknown |
| Overall Accuracy | > 0.90 | 0.63 (synthetic) |
| Mean Confidence (correct) | > 0.85 | 0.735 |
| Mean Confidence (incorrect) | < 0.50 | Unknown |

---

## Timeline

| Week | Phase | Deliverable | Data |
|------|-------|-------------|------|
| 1-2 | 1 | Baseline metrics (proof of concept) | 3-5 ground truth pairs |
| 3-4 | 2 | Feature labels | 5-10 feature surveys |
| 5 | 3 | Train/val split, extend train.py | Organize existing |
| 6-7 | 3 | Model v2 (minimum viable) | Train on available data |
| 8-12 | 1 | Accumulate more ground truth | Target 8-15 pairs total |
| 13+ | 3-4 | Model v2 (production quality), deploy | Retrain with full dataset |

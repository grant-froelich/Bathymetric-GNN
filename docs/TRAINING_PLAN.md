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
| 1 | Establish ground truth | 3-5 clean/noisy pairs (6-10 BAGs) | `prepare_ground_truth.py`, `evaluate_model.py` |
| 2 | Analyze noise patterns | (uses Phase 1 outputs) | `analyze_noise_patterns.py` |
| 3 | Add feature examples | 5-10 feature-area BAGs | `extract_s57_features.py` |
| 4 | Train on real data | 10-15 total BAGs | `train.py` (needs extension) |
| 5 | Deploy and refine | Unlimited | `inference_vr_native.py` |

**Total data requirement:** 15-20 VR BAG files minimum

---

## Phase 1: Establish Ground Truth

**Duration:** 1-2 weeks  
**Goal:** Create labeled validation data from real clean/noisy survey pairs

### Data Required

| Type | Count | Description |
|------|-------|-------------|
| Clean VR BAGs | 3-5 | Manually cleaned reference surveys |
| Noisy VR BAGs | 3-5 | Corresponding uncleaned versions |
| **Total BAGs** | **6-10** | Matched pairs of the same survey area |

**Selection criteria for survey pairs:**
- Include variety of depths (shallow, mid, deep)
- Include variety of seafloor types (flat, sloped, rocky)
- Include variety of noise levels (light, moderate, heavy)

### Step 1.1: Organize Survey Pairs

```
data/raw/clean/
├── survey_001_clean.bag
├── survey_002_clean.bag
├── survey_003_clean.bag
├── survey_004_clean.bag
└── survey_005_clean.bag

data/raw/noisy/
├── survey_001_noisy.bag
├── survey_002_noisy.bag
├── survey_003_noisy.bag
├── survey_004_noisy.bag
└── survey_005_noisy.bag
```

### Step 1.2: Generate Ground Truth Labels

Run for each of the 3-5 survey pairs:

```cmd
python scripts/prepare_ground_truth.py ^
    --clean data/raw/clean/survey_001_clean.bag ^
    --noisy data/raw/noisy/survey_001_noisy.bag ^
    --output-dir data/processed/labels ^
    --noise-threshold 0.15

python scripts/prepare_ground_truth.py ^
    --clean data/raw/clean/survey_002_clean.bag ^
    --noisy data/raw/noisy/survey_002_noisy.bag ^
    --output-dir data/processed/labels

:: Repeat for surveys 003, 004, 005...
```

**Output per survey:**
- `survey_001_ground_truth.tif` - 4-band GeoTIFF (labels, difference, noisy_depth, clean_depth)
- `survey_001_ground_truth_stats.json` - Noise distribution statistics

### Step 1.3: Evaluate Current Model Against Ground Truth

Run inference and evaluation for each of the 3-5 noisy surveys:

```cmd
:: Run inference
python scripts/inference_vr_native.py ^
    --input data/raw/noisy/survey_001_noisy.bag ^
    --model outputs/final_model.pt ^
    --output outputs/predictions/survey_001_predicted.bag ^
    --min-valid-ratio 0.01

:: Evaluate against ground truth
python scripts/evaluate_model.py ^
    --ground-truth data/processed/labels/survey_001_ground_truth.tif ^
    --predictions outputs/predictions/survey_001_predicted_gnn_outputs.tif ^
    --output outputs/metrics/survey_001_eval.json

:: Repeat for all survey pairs...
```

### Step 1.4: Review Baseline Metrics

After processing all 3-5 pairs, collect metrics:
- Per-class precision/recall/F1 (seafloor, noise, feature)
- Confusion matrices
- Confidence calibration curves

---

## Phase 2: Analyze Noise Patterns

**Duration:** 1 week  
**Goal:** Characterize real noise to improve synthetic noise generation

### Data Required

| Type | Count | Source |
|------|-------|--------|
| Ground truth files | 3-5 | Phase 1 outputs |
| **No additional BAGs needed** | | |

### Step 2.1: Run Noise Analysis

Analyze all ground truth files together:

```cmd
python scripts/analyze_noise_patterns.py ^
    --input data/processed/labels/survey_001_ground_truth.tif ^
           data/processed/labels/survey_002_ground_truth.tif ^
           data/processed/labels/survey_003_ground_truth.tif ^
           data/processed/labels/survey_004_ground_truth.tif ^
           data/processed/labels/survey_005_ground_truth.tif ^
    --output outputs/analysis/noise_patterns.json
```

**Analysis output includes:**
- Noise magnitude distribution (mean, std, percentiles)
- Noise direction bias (shoaling vs deepening)
- Depth-dependent noise rates
- Spatial clustering characteristics
- Swath position effects (outer beam noise rates)
- Roughness context (noise in smooth vs rough terrain)

### Step 2.2: Update Synthetic Noise Generator

Based on analysis, update `data/synthetic_noise.py`:

```python
# Example parameter updates based on real noise analysis:

# Swath profile (if outer beams show 2x noise rate)
swath_noise_profile = [2.0, 1.0, 1.0, 2.0]

# Clustering (if 70% of noise is clustered vs isolated)
cluster_probability = 0.70

# Depth-dependent magnitude
def get_noise_magnitude(depth):
    if depth < 20:
        return 0.15
    elif depth < 100:
        return 0.25
    else:
        return 0.40
```

---

## Phase 3: Feature Class Training

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

### Step 3.1: Extract Features from NOAA ENC Direct

The script queries ENC Direct REST API automatically - no download required.

| Source | URL | Update Frequency |
|--------|-----|------------------|
| ENC Direct | `encdirect.noaa.gov` | Weekly |

| Feature Type | S-57 Code | Default Radius |
|--------------|-----------|----------------|
| Shipwrecks | WRECKS | 50m |
| Obstructions | OBSTRN | 30m |
| Underwater rocks | UWTROC | 25m |

### Step 3.2: Generate Feature Labels

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

### Step 3.3: Verify Feature Coverage

Open the GeoJSON files in QGIS to verify:
- Features are correctly located within survey bounds
- Radii are appropriate for survey resolution
- No obvious charted features are missing

---

## Phase 4: Train on Real Data

**Duration:** 2 weeks  
**Goal:** Train model using real ground truth and feature labels

### Data Required

| Type | Count | Source |
|------|-------|--------|
| Ground truth labels | 3-5 | Phase 1 |
| Feature labels | 5-10 | Phase 3 |
| **Total labeled surveys** | **8-15** | |

### Step 4.1: Organize Train/Validation Split

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

### Step 4.2: Train Model

> **Note:** The current `train.py` uses synthetic noise only. To train on real labels, the script needs to be extended to accept label files. This is a development task.

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

### Step 4.3: Evaluate on Validation Set

Run inference and evaluation on each held-out validation survey:

```cmd
python scripts/inference_vr_native.py ^
    --input data/raw/noisy/survey_004_noisy.bag ^
    --model outputs/models/v2/best_model.pt ^
    --output outputs/predictions/survey_004_predicted.bag

python scripts/evaluate_model.py ^
    --ground-truth data/processed/validation/survey_004_ground_truth.tif ^
    --predictions outputs/predictions/survey_004_predicted_gnn_outputs.tif ^
    --output outputs/metrics/survey_004_val_eval.json
```

---

## Phase 5: Deployment & Refinement

**Duration:** Ongoing  
**Goal:** Deploy model and continuously improve from feedback

### Step 5.1: Production Inference

```cmd
python scripts/inference_vr_native.py ^
    --input new_survey.bag ^
    --model outputs/models/v2/best_model.pt ^
    --output cleaned_survey.bag ^
    --min-valid-ratio 0.01
```

**Outputs:**
- `cleaned_survey.bag` - Corrected VR BAG
- `cleaned_survey_gnn_outputs.tif` - Sidecar with classification, confidence, corrections

### Step 5.2: Review Low-Confidence Regions

> **Note:** `export_for_review.py` script needs to be created.

Identify uncertain regions for human review:
- Regions with confidence between 0.4-0.7
- Cluster size > 100 cells

### Step 5.3: Incorporate Feedback

1. Human reviewers correct model outputs in uncertain regions
2. Save corrections as new ground truth labels
3. Add to training set
4. Retrain model periodically (monthly or after accumulating corrections)

---

## Quick Reference

### Existing Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `prepare_ground_truth.py` | Generate labels from clean/noisy pairs | Ready |
| `evaluate_model.py` | Compare predictions to ground truth | Ready |
| `analyze_noise_patterns.py` | Characterize real noise | Ready |
| `extract_s57_features.py` | Extract features from ENC Direct | Ready |
| `train.py` | Train model (synthetic noise) | Needs extension for real labels |
| `inference_vr_native.py` | Run inference preserving VR structure | Ready |
| `export_for_review.py` | Export uncertain regions | Not yet created |

### Command Sequence

```cmd
:: PHASE 1: Ground Truth (run for each of 3-5 pairs)
python scripts/prepare_ground_truth.py --clean X_clean.bag --noisy X_noisy.bag --output-dir data/processed/labels
python scripts/inference_vr_native.py --input X_noisy.bag --model outputs/final_model.pt --output outputs/predictions/X.bag
python scripts/evaluate_model.py --ground-truth X_ground_truth.tif --predictions X_gnn_outputs.tif --output X_eval.json

:: PHASE 2: Noise Analysis (run once with all ground truth files)
python scripts/analyze_noise_patterns.py --input data/processed/labels/*_ground_truth.tif --output outputs/analysis/noise_patterns.json

:: PHASE 3: Feature Labels (run for each of 5-10 feature surveys)
python scripts/extract_s57_features.py --survey X.bag --labels X_features.tif --output X_features.geojson

:: PHASE 4: Training
python scripts/train.py --clean-surveys data/raw/clean --output-dir outputs/models/v2 --epochs 100

:: PHASE 5: Production
python scripts/inference_vr_native.py --input new_survey.bag --model outputs/models/v2/best_model.pt --output cleaned.bag
```

---

## Data Requirements Summary

| Phase | New BAGs Required | Running Total | Notes |
|-------|-------------------|---------------|-------|
| 1 | 6-10 (3-5 pairs) | 6-10 | Clean/noisy pairs required |
| 2 | 0 | 6-10 | Uses Phase 1 outputs |
| 3 | 5-10 | 11-20 | Standalone feature-area surveys |
| 4 | 0 | 11-20 | Uses Phases 1+3 labels |
| 5 | Unlimited | - | Production surveys |

**Minimum to start training:** 11-20 VR BAG files

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

| Week | Phase | Deliverable | BAGs Needed |
|------|-------|-------------|-------------|
| 1-2 | 1 | Baseline metrics | 6-10 (3-5 pairs) |
| 3 | 2 | Noise analysis report | 0 (uses Phase 1) |
| 4-5 | 3 | Feature labels | 5-10 feature surveys |
| 6 | 4 | Train/val split, extend train.py | 0 (organize existing) |
| 7-8 | 4 | Model v2 trained on real data | 0 |
| 9+ | 5 | Production deployment | Unlimited |

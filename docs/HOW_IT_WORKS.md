# How It Works: GNNs for Bathymetric Noise Detection

This document explains the theory behind Graph Neural Networks and how this tool applies them to bathymetric data cleaning.

> **Sign Convention:** This document uses the standard bathymetric convention where **depths are positive down**. A depth of 10m means 10 meters below the water surface. Larger values = deeper water.

## The Problem

Bathymetric surveys measure seafloor depth using acoustic sonar. The raw data contains noise from various sources:

| Noise Source | Appearance | Challenge |
|--------------|------------|-----------|
| Water column returns | Spikes above seafloor | Can look like real features |
| Multipath reflections | Systematic offsets | Depth-dependent patterns |
| Refraction errors | Smooth distortions | Hard to distinguish from slopes |
| System noise | Random speckle | Mixed with real texture |

**The core challenge:** Some noise looks like real seafloor features, and some real features look like noise. A spike on a flat seafloor is almost certainly noise, but the same spike in a rocky area might be a real rock.

```
Flat seafloor + spike = probably noise
Rocky area + spike = probably real feature
```

**Context matters.** This is why we use Graph Neural Networks.

### Shoal Spikes Are Critical

While noise can create spikes in both directions (shoaler or deeper), **shoal spikes are the primary concern** for navigation safety:

| Spike Direction | Depth Change | Safety Impact |
|-----------------|--------------|---------------|
| **Shoal** (shallower) | 10m → 5m | **Critical** - vessel could run aground on uncharted shallow |
| Deep (deeper) | 10m → 15m | Lower risk - vessel has more clearance than expected |

A false shoal spike that gets removed is a missed hazard. A real shoal feature that gets classified as noise is dangerous. This asymmetry means the model should be **conservative about removing shoal spikes** - when uncertain, preserve them for human review.

### Protecting Uncharted Features

A common concern: how do we prevent the model from cleaning away a real but uncharted feature (a new wreck, rock outcrop, or obstruction)?

**Multiple layers of protection:**

| Protection Layer | How It Works | Status |
|------------------|--------------|--------|
| **Confidence thresholds** | Low-confidence classifications are flagged for human review, not auto-corrected | ✓ Active |
| **Sidecar GeoTIFF** | All changes are documented with classification and confidence, enabling review | ✓ Active |
| **Conservative defaults** | Auto-correct threshold (0.85) means only high-confidence noise is touched | ✓ Active |
| **Feature class training** | Model learns to recognize feature-like patterns from ENC examples | Phase 3 (planned) |

**Current state:** With only synthetic noise training, the model classifies points as seafloor (0) or noise (2). Feature class (1) training is planned for Phase 3 when ENC feature labels are integrated.

**The key insight:** A properly trained model learns what features *look like*, not just where charted features *are*. When trained on diverse seafloor types including rocky terrain and known features, the model learns patterns like:

- Isolated spike on flat bottom → likely noise
- Spike connected to similar-depth neighbors → likely real feature
- Spike with low uncertainty in variable terrain → likely real feature

**For truly novel features** (something the model has never seen), the confidence score will typically be lower because the model is uncertain. These get flagged for human review rather than auto-corrected.

---

## Why Graphs?

Traditional approaches treat each depth measurement independently:

```
Point-by-point filtering:
┌─────────────────────────────────────┐
│  For each point:                    │
│    - Look at local statistics       │
│    - Apply threshold                │
│    - Classify as noise or not       │
└─────────────────────────────────────┘
```

This misses spatial context. A point's classification should depend on its neighbors.

**Graphs explicitly encode spatial relationships:**

```
Grid data:                    Graph representation:
┌───┬───┬───┬───┐            
│ A │ B │ C │ D │             A ── B ── C ── D
├───┼───┼───┼───┤             │    │    │    │
│ E │ F │ G │ H │   ──────►   E ── F ── G ── H
├───┼───┼───┼───┤             │    │    │    │
│ I │ J │ K │ L │             I ── J ── K ── L
└───┴───┴───┴───┘            
```

Each grid cell becomes a **node**. Connections between neighbors become **edges**.

---

## Graph Neural Network Basics

### Components of a Graph

| Component | What It Represents | In Bathymetric Data |
|-----------|-------------------|---------------------|
| **Nodes** | Individual data points | Grid cells with depth values |
| **Edges** | Connections between points | Spatial adjacency (neighbors) |
| **Node features** | Properties of each point | Depth, local statistics, gradients, curvature |
| **Edge features** | Properties of connections | Distance, depth difference, slope angle |

### Message Passing

GNNs work through **message passing**: each node collects information from its neighbors to update its own representation.

```
Step 1: Each node has initial features
        ┌───┐     ┌───┐     ┌───┐
        │ A │ ─── │ B │ ─── │ C │
        └───┘     └───┘     └───┘
        depth     depth     depth
        = 10m     = 15m     = 10m
                  (spike!)

Step 2: Node B collects "messages" from neighbors A and C
        
        A says: "I'm at 10m depth, pretty flat here"
        C says: "I'm at 10m depth, same as A"
        
Step 3: B updates its representation using neighbor info
        
        B now knows: "My neighbors are both at 10m,
                      but I'm at 15m - that's a 5m spike
                      which looks anomalous"
```

This happens for **all nodes simultaneously**, then repeats for multiple **layers**. Each layer expands the receptive field:

```
Layer 1: Each node knows about immediate neighbors (1-hop)
Layer 2: Each node knows about neighbors-of-neighbors (2-hop)
Layer 3: Each node knows about 3-hop neighborhood
...
```

After several layers, each node's representation encodes information about its broader spatial context.

### Attention Mechanism (GAT)

This tool uses **Graph Attention Networks (GAT)**. The key idea: not all neighbors are equally important.

```
Standard message passing:
    Node B = average(A, C)           # Equal weights

Attention-based:
    Node B = 0.8 × A + 0.2 × C       # Learned weights
```

The attention weights are learned during training. The network might learn:
- "Pay more attention to neighbors at similar depths"
- "Ignore neighbors that also look noisy"
- "Weight upslope neighbors differently than downslope"

### How Attention Weights Are Learned

Attention weights aren't hand-coded rules - they emerge from training on diverse examples. The network learns **which neighbors matter** by seeing many examples of noise vs. real features across different conditions.

#### The Learning Process

```
For each training example:
1. Network makes predictions using current attention weights
2. Compare predictions to ground truth labels
3. Calculate error (loss)
4. Adjust attention weights to reduce error
5. Repeat thousands of times across diverse surveys
```

The key insight: **attention weights that work across diverse conditions are the ones that capture real patterns**, not artifacts of a specific survey type.

#### Why Training Diversity Matters

Each dimension of diversity teaches the network different aspects of "what matters":

| Training Diversity | What Attention Learns |
|--------------------|----------------------|
| **Depth ranges** | Shallow water has different noise characteristics than deep water. Attention learns depth-appropriate neighbor weighting. |
| **Seafloor types** | On flat mud, any spike is suspicious. On rocky terrain, spikes are normal. Attention learns to assess spikes relative to local texture. |
| **Noise severity** | Light noise looks different from heavy contamination. Attention learns to recognize noise patterns at different intensities. |
| **Equipment types** | Different sonars produce different noise signatures. Attention learns equipment-invariant features. |
| **Geographic regions** | Water column properties vary by region. Attention learns to generalize across oceanographic conditions. |

#### Concrete Example: Learning from Depth Diversity

Consider how depth diversity affects what attention learns:

```
Shallow survey (5-20m):
┌────────────────────────────────────────────────────────────┐
│ - Water column noise common (fish, bubbles, turbulence)    │
│ - Spikes often isolated, clearly noise                     │
│ - Network learns: "isolated shallow spikes = high noise    │
│   probability, weight similar-depth neighbors heavily"     │
└────────────────────────────────────────────────────────────┘

Deep survey (100-500m):
┌────────────────────────────────────────────────────────────┐
│ - Less water column interference                           │
│ - Multipath and refraction errors more common              │
│ - Systematic patterns across swaths                        │
│ - Network learns: "check across-track neighbors for        │
│   systematic offsets, not just local spikes"               │
└────────────────────────────────────────────────────────────┘
```

If trained only on shallow surveys, the network might learn attention patterns that fail on deep water (and vice versa). Training on both teaches **depth-invariant** attention.

#### Concrete Example: Learning from Seafloor Diversity

```
Flat mud seafloor:
┌────────────────────────────────────────────────────────────┐
│ Depths: 50, 50, 50, 55, 50, 50, 50                         │
│                      ↑                                     │
│                   spike                                    │
│                                                            │
│ All neighbors are ~50m, spike stands out                   │
│ Network learns: "when neighbors are uniform, even small    │
│   deviations are suspicious - weight ALL neighbors"        │
└────────────────────────────────────────────────────────────┘

Rocky outcrop:
┌────────────────────────────────────────────────────────────┐
│ Depths: 48, 52, 47, 55, 53, 49, 51                         │
│                      ↑                                     │
│                   same value, but context differs          │
│                                                            │
│ Neighbors vary naturally, 55m fits the pattern             │
│ Network learns: "when neighbors are variable, assess       │
│   whether spike fits local texture - weight SIMILAR        │
│   neighbors more than dissimilar ones"                     │
└────────────────────────────────────────────────────────────┘
```

Training on both seafloor types teaches the network to **adapt attention based on local context**.

#### What Happens Without Diversity

| Missing Diversity | Failure Mode |
|-------------------|--------------|
| Only flat seafloors | Network flags all texture as noise, removes real rocky features |
| Only shallow water | Network misses deep-water systematic errors |
| Only light noise | Network under-confident on heavily contaminated surveys |
| Only one sonar type | Network fails on different equipment |

This is why the training plan emphasizes collecting **diverse** ground truth pairs, not just **many** pairs.

#### Multi-Head Attention

The GAT architecture uses **4 attention heads**, meaning it learns 4 different attention patterns simultaneously:

```
Head 1: Might learn "weight neighbors by depth similarity"
Head 2: Might learn "weight neighbors by uncertainty"
Head 3: Might learn "weight upslope vs downslope differently"
Head 4: Might learn "weight by distance"

Final output = combination of all 4 perspectives
```

Each head can specialize in different aspects of the problem. Diverse training data gives each head enough examples to learn meaningful patterns.

---

## How This Tool Works

### Step 1: Build the Graph

The bathymetric grid is converted to a graph:

```python
# Each valid cell becomes a node
# Neighboring cells are connected by edges

Grid (5x5):                    Graph:
┌────┬────┬────┬────┬────┐    
│ 10 │ 10 │ 11 │ 10 │ 10 │     Nodes: 25 (one per cell)
├────┼────┼────┼────┼────┤     Edges: ~80 (8-connectivity)
│ 10 │ 10 │ 15 │ 10 │ 10 │     
├────┼────┼────┼────┼────┤     Node at (2,2) has depth 15m
│ 11 │ 15 │ 25 │ 14 │ 11 │     (potential noise spike)
├────┼────┼────┼────┼────┤     
│ 10 │ 10 │ 14 │ 10 │ 10 │     
├────┼────┼────┼────┼────┤     
│ 10 │ 10 │ 11 │ 10 │ 10 │     
└────┴────┴────┴────┴────┘    
```

### Step 2: Compute Node Features

Each node gets features describing its local properties:

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| Depth | Raw depth value | Basic measurement |
| Uncertainty | Measurement uncertainty | Low uncertainty = more trustworthy |
| Local mean | Average of neighbors | Baseline for comparison |
| Local std | Standard deviation of neighbors | Roughness indicator |
| Depth difference | Difference from local mean | Spike magnitude |
| Gradient magnitude | Slope steepness | Distinguishes slopes from spikes |
| Gradient direction | Slope direction (encoded) | Directional context |

### Step 3: Compute Edge Features

Each edge gets features describing the relationship between connected nodes:

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| Distance | Spatial distance between nodes | Closer = more relevant |
| Depth difference | Depth change along edge | Slope vs spike indicator |
| Gradient alignment | How edge aligns with local slope | Consistent slope vs anomaly |

### Step 4: Run the GNN

The graph passes through the neural network:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Input: Node features (7 per node)                      │
│         Edge features (3 per edge)                      │
│         Edge connectivity                               │
│                                                         │
│              ↓                                          │
│                                                         │
│  Local Feature Extractor (MLP)                          │
│  - Processes each node's features independently         │
│  - Expands to hidden dimension (64)                     │
│                                                         │
│              ↓                                          │
│                                                         │
│  GNN Backbone (4 GAT layers)                            │
│  - Layer 1: Aggregate 1-hop neighborhood                │
│  - Layer 2: Aggregate 2-hop neighborhood                │
│  - Layer 3: Aggregate 3-hop neighborhood                │
│  - Layer 4: Final representation                        │
│                                                         │
│              ↓                                          │
│                                                         │
│  Output Heads:                                          │
│  - Classification: seafloor / feature / noise           │
│  - Confidence: 0-1 certainty score                      │
│  - Correction: suggested depth adjustment               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Step 5: Interpret Results

Each node gets three outputs:

| Output | Values | Meaning |
|--------|--------|---------|
| **Classification** | 0, 1, or 2 | 0=seafloor, 1=feature, 2=noise |
| **Confidence** | 0.0 to 1.0 | How certain the model is |
| **Correction** | Depth offset | Suggested adjustment if noise |

**Decision logic:**

```
If classification == noise AND confidence > threshold:
    Apply correction automatically
    
If classification == noise AND confidence < threshold:
    Flag for human review
    
If classification == feature:
    Preserve (do not modify)
    
If classification == seafloor:
    Preserve (do not modify)
```

---

## Why GNNs Work for This Problem

### Local Ambiguity, Global Clarity

Consider this scenario:

```
Scenario A: Spike on flat seafloor
┌────┬────┬────┬────┬────┐
│ 10 │ 10 │ 10 │ 10 │ 10 │
├────┼────┼────┼────┼────┤
│ 10 │ 10 │ 10 │ 10 │ 10 │
├────┼────┼────┼────┼────┤
│ 10 │ 10 │ 15 │ 10 │ 10 │  ← Spike is isolated
├────┼────┼────┼────┼────┤     Almost certainly NOISE
│ 10 │ 10 │ 10 │ 10 │ 10 │
├────┼────┼────┼────┼────┤
│ 10 │ 10 │ 10 │ 10 │ 10 │
└────┴────┴────┴────┴────┘

Scenario B: Spike in rocky area
┌────┬────┬────┬────┬────┐
│ 10 │ 12 │ 11 │ 13 │ 10 │
├────┼────┼────┼────┼────┤
│ 11 │ 14 │ 12 │ 11 │ 12 │
├────┼────┼────┼────┼────┤
│ 12 │ 13 │ 15 │ 14 │ 11 │  ← Spike fits the pattern
├────┼────┼────┼────┼────┤     Probably a real FEATURE
│ 10 │ 11 │ 13 │ 12 │ 10 │
├────┼────┼────┼────┼────┤
│ 10 │ 10 │ 11 │ 10 │ 10 │
└────┴────┴────┴────┴────┘
```

Looking at the center cell alone (15m), both scenarios look identical. But the context is completely different:
- Scenario A: Neighbors are all flat (10m), spike is anomalous
- Scenario B: Neighbors are variable (10 to 14m), spike fits the pattern

**The GNN learns to use this context.**

### What the Network Learns

During training, the GNN learns patterns like:

| Pattern | Learned Association |
|---------|---------------------|
| Isolated spike on flat seafloor | Noise |
| Spike connected to other spikes | Feature (rock outcrop) |
| Smooth depth change | Seafloor slope |
| Abrupt depth change breaking ridge | Noise |
| Cluster of anomalies near ship track edge | Noise (systematic) |
| Single point with high uncertainty | Noise |
| Single point with low uncertainty in rough area | Feature |

### The Training Process

The network learns these patterns from labeled examples:

```
Training data:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Clean survey (manually verified)                       │
│  + Corresponding noisy survey (before cleaning)         │
│  ────────────────────────────────────────────────       │
│  = Ground truth labels (this is noise, this is not)     │
│                                                         │
└─────────────────────────────────────────────────────────┘

Training loop:
1. Feed noisy survey through GNN
2. Compare predictions to ground truth labels
3. Compute loss (how wrong were we?)
4. Update network weights to reduce loss
5. Repeat for many surveys and epochs
```

---

## Practical Workflow

### For End Users

```
┌─────────────────────────────────────────────────────────┐
│  1. Run inference on new survey                         │
│                                                         │
│     python scripts/inference_native.py \                │
│         --input survey.bag \                            │
│         --model model.pt \                              │
│         --output cleaned.bag                            │
│                                                         │
│  2. Review outputs                                      │
│     - cleaned.bag: Corrected depths                     │
│     - cleaned_gnn_outputs.tif: Classification/confidence│
│                                                         │
│  3. Check low-confidence regions in GIS                 │
│     - Load sidecar GeoTIFF                              │
│     - Style confidence band                             │
│     - Review flagged areas                              │
│                                                         │
│  4. Provide feedback (optional)                         │
│     - Correct any errors                                │
│     - Save as new training data                         │
│     - Model improves over time                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Understanding Confidence

The confidence score indicates how certain the model is:

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| > 0.85 | High certainty | Auto-corrected, trust result |
| 0.7 - 0.85 | Moderate certainty | Spot-check recommended |
| 0.5 - 0.7 | Uncertain | Manual review recommended |
| < 0.5 | Low certainty | Definitely review |

**Well-calibrated confidence means:**
- When the model says 90% confident, it's right ~90% of the time
- When the model says 50% confident, it's right ~50% of the time

This calibration improves as the model trains on more real data.

---

## Comparison to Traditional Methods

| Aspect | Traditional Filtering | GNN Approach |
|--------|----------------------|--------------|
| Context | Local only (3x3 or 5x5 window) | Multi-hop neighborhood (learned) |
| Parameters | Hand-tuned thresholds | Learned from data |
| Adaptability | Fixed rules | Learns survey-specific patterns |
| Features | Designer chooses what matters | Network learns what matters |
| Edge cases | Fails on ambiguous cases | Provides confidence score |
| Improvement | Requires manual tuning | Improves with more training data |

---

## Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| Training data required | Needs labeled examples to learn | Start with 3-5 clean/noisy pairs |
| Computational cost | Slower than simple filters | GPU acceleration, tiled processing |
| Not magic | Can't detect noise that humans can't | Confidence scores flag uncertainty |
| Domain shift | May struggle with very different surveys | Include diverse training data |
| Feature confusion | May confuse rare features with noise | Train with feature examples |

---

## Processing Time

Typical processing times (on GPU):

| Operation | Time | Notes |
|-----------|------|-------|
| **Training** (5 survey pairs, 50 epochs) | Minutes | One-time or periodic retraining |
| **Inference** (single survey) | Seconds to minutes | Depends on survey size |
| **Full production run** | Minutes per survey | Includes sidecar GeoTIFF generation |

Processing is slower than simple threshold filters but provides context-aware detection that filters cannot achieve. CPU-only processing is supported but significantly slower.

---

## Operational Confidence Thresholds

The tool uses configurable confidence thresholds to balance automation vs. human review:

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `auto_correct_threshold` | 0.85 | Only auto-correct noise if confidence exceeds this |
| `review_threshold` | 0.60 | Flag for human review if confidence below this |

**These thresholds are policy decisions**, not technical requirements. Adjusting them changes the tradeoff:

| Setting | Effect |
|---------|--------|
| Higher auto-correct threshold (e.g., 0.95) | More conservative - fewer auto-corrections, more human review |
| Lower auto-correct threshold (e.g., 0.75) | More aggressive - more auto-corrections, less human review |
| Higher review threshold (e.g., 0.70) | More items flagged for review |
| Lower review threshold (e.g., 0.50) | Fewer items flagged for review |

The "right" thresholds depend on operational risk tolerance. A 95% confidence benchmark for shoal safety is a policy question for Coast Survey leadership, not a technical decision.

---

## Summary

**Graph Neural Networks work for bathymetric noise detection because:**

1. **Spatial context matters** - A spike's meaning depends on its surroundings
2. **Graphs encode relationships** - Neighbors are explicitly connected
3. **Message passing aggregates context** - Each node learns from its neighborhood
4. **Attention focuses on relevant neighbors** - Not all connections are equal
5. **End-to-end learning** - Network discovers useful patterns automatically

**The practical result:**

```
Input: Noisy survey with ambiguous points
       ↓
       GNN analyzes spatial context
       ↓
Output: Classification + Confidence + Correction
        (noise vs feature vs seafloor)
```

Human reviewers focus on uncertain regions. The model improves with feedback. Quality increases over time.

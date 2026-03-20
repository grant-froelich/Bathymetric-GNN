"""
Shared constants for Bathymetric GNN Processing.

These constants are used across training and inference and MUST stay in sync.
Centralizing them here prevents silent divergence between the two paths.

Saved into model checkpoints so old models can be loaded with correct values.
"""

# Floor value for local_std normalization to prevent division by near-zero
# in perfectly flat areas. 0.01m is well below any real bathymetric variability.
CORRECTION_NORM_FLOOR = 0.01

# Maximum allowed normalized correction magnitude (in local_std units).
# Corrections beyond this are clamped to prevent extreme outliers from
# dominating training. 50 std devs is well beyond any legitimate noise
# pattern while still allowing the model to learn large corrections.
CORRECTION_NORM_CAP = 50.0

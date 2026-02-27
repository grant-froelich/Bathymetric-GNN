"""
Loss functions for bathymetric GNN training.

Includes:
- Classification loss (noise vs feature vs seafloor)
- Correction loss (predicted vs actual depth correction)
- Confidence calibration loss
- Combined multi-task loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class ClassificationLoss(nn.Module):
    """
    Weighted cross-entropy loss for node classification.
    
    Handles class imbalance (typically more seafloor than noise/features).
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            logits: [num_nodes, num_classes]
            targets: [num_nodes] class labels
            
        Returns:
            Scalar loss
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


class CorrectionLoss(nn.Module):
    """
    Loss for depth correction prediction.
    
    Uses Huber loss for robustness to outliers.
    """
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute correction loss.
        
        Args:
            predicted: [num_nodes] predicted corrections
            target: [num_nodes] true corrections (noisy - clean)
            mask: Optional [num_nodes] mask for which nodes to include
            
        Returns:
            Scalar loss
        """
        if mask is not None:
            predicted = predicted[mask]
            target = target[mask]
        
        if len(predicted) == 0:
            return torch.tensor(0.0, device=predicted.device)
        
        return F.huber_loss(predicted, target, delta=self.delta)


class ConfidenceCalibrationLoss(nn.Module):
    """
    Loss to calibrate confidence predictions.
    
    Encourages the model to output high confidence when correct
    and low confidence when wrong.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        confidence: torch.Tensor,
        predicted_class: torch.Tensor,
        true_class: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence calibration loss.
        
        Args:
            confidence: [num_nodes] predicted confidence (0-1)
            predicted_class: [num_nodes] predicted class labels
            true_class: [num_nodes] true class labels
            
        Returns:
            Scalar loss
        """
        # Correctness indicator
        correct = (predicted_class == true_class).float()
        
        # We want confidence to match correctness
        # High confidence when correct, low when wrong
        return F.binary_cross_entropy(confidence, correct)


class FeaturePreservationLoss(nn.Module):
    """
    Loss to encourage preservation of real features.
    
    Penalizes classifying real features as noise more heavily
    than the reverse error.
    """
    
    def __init__(self, feature_class: int = 1, noise_class: int = 2, penalty_weight: float = 2.0):
        super().__init__()
        self.feature_class = feature_class
        self.noise_class = noise_class
        self.penalty_weight = penalty_weight
    
    def forward(
        self,
        predicted_class: torch.Tensor,
        true_class: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature preservation penalty.
        
        Args:
            predicted_class: [num_nodes] predicted class labels
            true_class: [num_nodes] true class labels
            
        Returns:
            Scalar loss
        """
        # Identify cases where real features were classified as noise
        is_real_feature = true_class == self.feature_class
        predicted_as_noise = predicted_class == self.noise_class
        
        false_noise = (is_real_feature & predicted_as_noise).float()
        
        # Return weighted penalty
        return self.penalty_weight * false_noise.mean()


class ShoalSafetyLoss(nn.Module):
    """
    Asymmetric loss for navigation safety.
    
    Penalizes shoal-direction errors more heavily than deep-direction errors.
    Removing a real shoal feature is dangerous for navigation.
    
    Sign convention (depths positive down):
    - correction < 0: noisy is SHALLOWER than clean (shoal spike)
    - correction > 0: noisy is DEEPER than clean (deep spike)
    """
    
    def __init__(
        self,
        seafloor_class: int = 0,
        noise_class: int = 2,
        shoal_penalty: float = 3.0,
        deep_penalty: float = 1.0,
    ):
        super().__init__()
        self.seafloor_class = seafloor_class
        self.noise_class = noise_class
        self.shoal_penalty = shoal_penalty
        self.deep_penalty = deep_penalty
    
    def forward(
        self,
        predicted_class: torch.Tensor,
        true_class: torch.Tensor,
        correction_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute shoal safety penalty.
        
        Args:
            predicted_class: [num_nodes] predicted class labels
            true_class: [num_nodes] true class labels
            correction_targets: [num_nodes] depth corrections (noisy - clean)
            
        Returns:
            Scalar loss
        """
        # False positives: predicted noise but actually seafloor
        is_real_seafloor = true_class == self.seafloor_class
        predicted_as_noise = predicted_class == self.noise_class
        false_positive = is_real_seafloor & predicted_as_noise
        
        if not false_positive.any():
            return torch.tensor(0.0, device=predicted_class.device)
        
        # correction < 0 means noisy is shallower (shoal spike)
        is_shoal = correction_targets < 0
        
        shoal_false_positive = (false_positive & is_shoal).float()
        deep_false_positive = (false_positive & ~is_shoal).float()
        
        # Asymmetric penalties - shoal errors are much worse
        penalty = (
            self.shoal_penalty * shoal_false_positive.sum() +
            self.deep_penalty * deep_false_positive.sum()
        ) / max(false_positive.sum().item(), 1.0)
        
        return penalty


class BathymetricGNNLoss(nn.Module):
    """
    Combined multi-task loss for bathymetric GNN training.
    
    Combines:
    - Classification loss
    - Correction loss (optional)
    - Confidence calibration loss
    - Feature preservation penalty
    - Shoal safety penalty (asymmetric for navigation safety)
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        classification_weight: float = 1.0,
        correction_weight: float = 0.5,
        confidence_weight: float = 0.2,
        feature_preservation_weight: float = 0.3,
        shoal_safety_weight: float = 0.5,
        label_smoothing: float = 0.0,
        correction_delta: float = 1.0,
    ):
        """
        Initialize combined loss.
        
        Args:
            class_weights: Weights for each class in classification loss
            classification_weight: Weight for classification loss
            correction_weight: Weight for correction loss
            confidence_weight: Weight for confidence calibration loss
            feature_preservation_weight: Weight for feature preservation penalty
            shoal_safety_weight: Weight for shoal safety penalty
            label_smoothing: Label smoothing for classification
            correction_delta: Huber loss delta for correction head. Controls
                the transition point between quadratic (MSE-like) and linear
                (MAE-like) loss. Errors below delta get proportional gradients;
                errors above delta get constant gradients. Should be set to
                cover the typical range of correction magnitudes in the
                training data (e.g. 95th percentile of |corrections|).
        """
        super().__init__()
        
        self.classification_loss = ClassificationLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        self.correction_loss = CorrectionLoss(delta=correction_delta)
        self.confidence_loss = ConfidenceCalibrationLoss()
        self.feature_preservation_loss = FeaturePreservationLoss()
        self.shoal_safety_loss = ShoalSafetyLoss()
        
        self.classification_weight = classification_weight
        self.correction_weight = correction_weight
        self.confidence_weight = confidence_weight
        self.feature_preservation_weight = feature_preservation_weight
        self.shoal_safety_weight = shoal_safety_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dictionary containing:
                - class_logits: [num_nodes, num_classes]
                - predicted_class: [num_nodes]
                - confidence: [num_nodes]
                - correction: [num_nodes] (optional)
            targets: Target values dictionary containing:
                - class_labels: [num_nodes]
                - correction_targets: [num_nodes] (optional)
                - noise_mask: [num_nodes] (optional)
                
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Classification loss
        class_loss = self.classification_loss(
            outputs['class_logits'],
            targets['class_labels'],
        )
        losses['classification'] = class_loss
        
        # Correction loss (only for noise points if mask provided)
        if 'correction' in outputs and 'correction_targets' in targets:
            noise_mask = targets.get('noise_mask', None)
            corr_loss = self.correction_loss(
                outputs['correction'],
                targets['correction_targets'],
                mask=noise_mask,
            )
            losses['correction'] = corr_loss
        else:
            losses['correction'] = torch.tensor(0.0, device=class_loss.device)
        
        # Confidence calibration loss
        conf_loss = self.confidence_loss(
            outputs['confidence'],
            outputs['predicted_class'],
            targets['class_labels'],
        )
        losses['confidence'] = conf_loss
        
        # Feature preservation penalty
        feat_loss = self.feature_preservation_loss(
            outputs['predicted_class'],
            targets['class_labels'],
        )
        losses['feature_preservation'] = feat_loss
        
        # Shoal safety penalty (asymmetric)
        if 'correction_targets' in targets:
            shoal_loss = self.shoal_safety_loss(
                outputs['predicted_class'],
                targets['class_labels'],
                targets['correction_targets'],
            )
            losses['shoal_safety'] = shoal_loss
        else:
            losses['shoal_safety'] = torch.tensor(0.0, device=class_loss.device)
        
        # Total weighted loss
        total = (
            self.classification_weight * losses['classification'] +
            self.correction_weight * losses['correction'] +
            self.confidence_weight * losses['confidence'] +
            self.feature_preservation_weight * losses['feature_preservation'] +
            self.shoal_safety_weight * losses['shoal_safety']
        )
        losses['total'] = total
        
        return losses


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = 3,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute inverse frequency class weights.
    
    Args:
        labels: [N] tensor of class labels
        num_classes: Number of classes
        smoothing: Smoothing factor to prevent extreme weights
        
    Returns:
        [num_classes] tensor of weights
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts + smoothing * counts.sum()  # Add smoothing
    
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return weights


def compute_correction_delta(
    corrections: np.ndarray,
    percentile: float = 95.0,
    min_delta: float = 1.0,
) -> float:
    """
    Compute Huber loss delta from the training data correction distribution.
    
    Sets delta to a percentile of the absolute correction magnitudes so that
    the vast majority of corrections fall within the quadratic (proportional
    gradient) regime of the Huber loss. Only the extreme tail beyond this
    percentile gets capped gradients.
    
    This is Option 1 (data-derived, compute-once) of three approaches:
    
    Option 1 (CURRENT): Compute delta once from training data at startup.
        Pros: Simple, deterministic, no risk of instability.
        Cons: Fixed for entire training run. If the model improves and
        errors shrink, the quadratic regime is wider than necessary.
    
    Option 2 (FUTURE): Adapt delta during training based on prediction error.
        Track the running distribution of the model's actual correction
        errors each epoch. Set delta to a percentile of recent errors.
        Early in training when errors are large, delta is large. As the
        model improves, delta shrinks to keep the quadratic regime focused
        on the range where the model is still learning.
        Implementation notes:
        - Compute percentile of |predicted - target| at end of each epoch
        - Use exponential moving average (momentum ~0.9) to smooth updates
        - Set a floor value (e.g. min_delta) to prevent collapse
        - Risk: if delta drops too fast, large corrections that the model
          hasn't learned yet get constant gradients again
    
    Option 3 (FUTURE): Combine Options 1 and 2.
        Use the data-derived delta (Option 1) as a ceiling. Allow the
        error-derived delta (Option 2) to decay below this ceiling with
        momentum, but never exceed it. This gives the adaptive benefits
        of Option 2 with a safety bound from Option 1.
        Implementation notes:
        - ceiling = data-derived delta (computed once)
        - running_delta = EMA of per-epoch error percentile
        - effective_delta = min(running_delta, ceiling)
        - Provides best of both: adapts as model learns, but can't
          exceed the data-justified range
    
    Args:
        corrections: Array of correction magnitudes from training data
            (absolute values of noisy - clean for noise-labeled cells)
        percentile: Percentile to use for delta (default 95th)
        min_delta: Minimum delta to prevent degenerate behavior
        
    Returns:
        Huber delta value
    """
    if len(corrections) == 0:
        return min_delta
    
    abs_corrections = np.abs(corrections)
    delta = float(np.percentile(abs_corrections, percentile))
    
    return max(delta, min_delta)

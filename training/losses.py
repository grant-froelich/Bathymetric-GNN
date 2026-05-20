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


class RegressionLoss(nn.Module):
    """
    Asymmetric Huber loss for regression-mode training.
    
    Applies to every valid cell, not just cells flagged as noise. Bakes
    shoal-safety asymmetry directly into the correction loss: predictions
    that leave the corrected surface deeper than reality are penalized
    more heavily than predictions that leave it shallower.
    
    Sign math (depths positive down, correction = noisy - clean):
        corrected_depth = noisy - predicted_correction
        error = predicted_correction - target_correction
        
        error > 0  -> corrected depth is shallower than reality (SAFE)
        error < 0  -> corrected depth is deeper than reality (DANGEROUS)
    
    The dangerous direction always means "we are saying there is more
    water than there really is," which is the case mariners need to
    be protected against.
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        safe_weight: float = 1.0,
        dangerous_weight: float = 3.0,
    ):
        super().__init__()
        self.delta = delta
        self.safe_weight = safe_weight
        self.dangerous_weight = dangerous_weight
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute asymmetric Huber loss on correction predictions.
        
        Args:
            predicted: [num_nodes] predicted corrections
            target: [num_nodes] true corrections (noisy - clean)
            valid_mask: Optional [num_nodes] mask of cells to include
        
        Returns:
            Scalar loss
        """
        if valid_mask is not None:
            predicted = predicted[valid_mask]
            target = target[valid_mask]
        
        if predicted.numel() == 0:
            return torch.tensor(0.0, device=predicted.device, requires_grad=True)
        
        error = predicted - target  # signed
        abs_error = torch.abs(error)
        
        # Huber: quadratic for |e| <= delta, linear above
        delta_t = torch.tensor(self.delta, device=abs_error.device, dtype=abs_error.dtype)
        quadratic = torch.minimum(abs_error, delta_t)
        linear = abs_error - quadratic
        base_loss = 0.5 * quadratic.pow(2) + self.delta * linear
        
        # Asymmetric weighting: error < 0 = predicted under target = dangerous
        weight = torch.where(
            error >= 0,
            torch.full_like(error, self.safe_weight),
            torch.full_like(error, self.dangerous_weight),
        )
        
        return (weight * base_loss).mean()


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
        regression_delta: float = 1.0,
        regression_safe_weight: float = 1.0,
        regression_dangerous_weight: float = 3.0,
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
            correction_delta: Huber delta for classification-mode correction loss
                             (typically derived from training data via compute_correction_delta)
            regression_delta: Huber delta for regression mode
            regression_safe_weight: Weight for safe-direction errors (regression mode)
            regression_dangerous_weight: Weight for dangerous-direction errors (regression mode)
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
        self.regression_loss = RegressionLoss(
            delta=regression_delta,
            safe_weight=regression_safe_weight,
            dangerous_weight=regression_dangerous_weight,
        )
        
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
        Compute combined loss. Dispatches based on targets['mode'].
        
        Args:
            outputs: Model outputs dictionary containing:
                - class_logits: [num_nodes, num_classes]
                - predicted_class: [num_nodes]
                - confidence: [num_nodes]
                - correction: [num_nodes] (optional)
            targets: Target values dictionary containing:
                - mode: 'classification' or 'regression' (default 'classification')
                - class_labels: [num_nodes] (classification mode)
                - correction_targets: [num_nodes]
                - noise_mask: [num_nodes] (classification mode)
                - valid_mask: [num_nodes] (regression mode)
                
        Returns:
            Dictionary with individual losses and total loss
        """
        mode = targets.get('mode', 'classification')
        
        if mode == 'regression':
            return self._forward_regression(outputs, targets)
        return self._forward_classification(outputs, targets)
    
    def _forward_regression(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Regression-mode loss: asymmetric Huber on all valid cells."""
        losses = {}
        device = outputs['correction'].device if 'correction' in outputs else None
        
        if 'correction' not in outputs or 'correction_targets' not in targets:
            raise ValueError(
                "Regression mode requires 'correction' in outputs and "
                "'correction_targets' in targets"
            )
        
        valid_mask = targets.get('valid_mask', None)
        reg_loss = self.regression_loss(
            outputs['correction'],
            targets['correction_targets'],
            valid_mask=valid_mask,
        )
        losses['regression'] = reg_loss
        
        # Inactive losses in regression mode (filled in for consistent dict shape)
        zero = torch.tensor(0.0, device=device)
        losses['classification'] = zero
        losses['correction'] = zero
        losses['confidence'] = zero
        losses['feature_preservation'] = zero
        losses['shoal_safety'] = zero
        
        losses['total'] = reg_loss
        return losses
    
    def _forward_classification(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Classification-mode loss: existing multi-task combination."""
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
        
        # Regression loss inactive in classification mode
        losses['regression'] = torch.tensor(0.0, device=class_loss.device)
        
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
    
    This is the V8 data-derived approach. Future options documented inline.
    
    Args:
        corrections: Array of correction magnitudes from training data
            (absolute values of noisy - clean for noise-labeled cells, or
            absolute values of NORMALIZED corrections in V9+)
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

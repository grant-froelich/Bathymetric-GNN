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


class BathymetricGNNLoss(nn.Module):
    """
    Combined multi-task loss for bathymetric GNN training.
    
    Combines:
    - Classification loss
    - Correction loss (optional)
    - Confidence calibration loss
    - Feature preservation penalty
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        classification_weight: float = 1.0,
        correction_weight: float = 0.5,
        confidence_weight: float = 0.2,
        feature_preservation_weight: float = 0.3,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize combined loss.
        
        Args:
            class_weights: Weights for each class in classification loss
            classification_weight: Weight for classification loss
            correction_weight: Weight for correction loss
            confidence_weight: Weight for confidence calibration loss
            feature_preservation_weight: Weight for feature preservation penalty
            label_smoothing: Label smoothing for classification
        """
        super().__init__()
        
        self.classification_loss = ClassificationLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        self.correction_loss = CorrectionLoss()
        self.confidence_loss = ConfidenceCalibrationLoss()
        self.feature_preservation_loss = FeaturePreservationLoss()
        
        self.classification_weight = classification_weight
        self.correction_weight = correction_weight
        self.confidence_weight = confidence_weight
        self.feature_preservation_weight = feature_preservation_weight
    
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
        
        # Total weighted loss
        total = (
            self.classification_weight * losses['classification'] +
            self.correction_weight * losses['correction'] +
            self.confidence_weight * losses['confidence'] +
            self.feature_preservation_weight * losses['feature_preservation']
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

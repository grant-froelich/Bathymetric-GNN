"""
Graph Neural Network architecture for bathymetric data processing.

The model performs:
1. Node classification: noise / feature / smooth seafloor
2. Confidence estimation: how certain is the classification
3. Depth correction: suggested correction for noisy points
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GCNConv,
        GATConv, 
        SAGEConv,
        GINConv,
        BatchNorm,
        global_mean_pool,
    )
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocalFeatureExtractor(nn.Module):
    """
    Extracts local features from raw node attributes.
    
    This is a simple MLP that processes each node's features independently
    before the graph convolution layers aggregate neighborhood information.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_channels, out_channels))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GNNBackbone(nn.Module):
    """
    Graph Neural Network backbone supporting multiple architectures.
    
    Supported types:
    - GCN: Graph Convolutional Network
    - GAT: Graph Attention Network
    - GraphSAGE: Sample and Aggregate
    - GIN: Graph Isomorphism Network
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        gnn_type: str = "GAT",
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required")
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Input channels for this layer
            if i == 0:
                layer_in = in_channels
            elif gnn_type == "GAT":
                layer_in = hidden_channels * heads
            else:
                layer_in = hidden_channels
            
            # Output channels
            layer_out = hidden_channels
            
            # Create appropriate conv layer
            if gnn_type == "GCN":
                conv = GCNConv(layer_in, layer_out)
            
            elif gnn_type == "GAT":
                # Last layer doesn't use multiple heads for output
                conv = GATConv(
                    layer_in,
                    layer_out,
                    heads=heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=i < num_layers - 1,  # Concat heads except last layer
                )
            
            elif gnn_type == "GraphSAGE":
                conv = SAGEConv(layer_in, layer_out)
            
            elif gnn_type == "GIN":
                mlp = nn.Sequential(
                    nn.Linear(layer_in, layer_out),
                    nn.ReLU(),
                    nn.Linear(layer_out, layer_out),
                )
                conv = GINConv(mlp)
            
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.convs.append(conv)
            
            # Batch norm after each layer
            if gnn_type == "GAT" and i < num_layers - 1:
                self.norms.append(BatchNorm(hidden_channels * heads))
            else:
                self.norms.append(BatchNorm(hidden_channels))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GNN layers.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, hidden_channels]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Apply convolution
            if self.gnn_type == "GAT" and edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            
            # Apply normalization
            x = norm(x)
            
            # Apply activation (except last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class ClassificationHead(nn.Module):
    """Output head for node classification."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for each class."""
        return self.mlp(x)


class ConfidenceHead(nn.Module):
    """Output head for confidence estimation."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns confidence score in [0, 1]."""
        return self.mlp(x).squeeze(-1)


class CorrectionHead(nn.Module):
    """Output head for depth correction prediction."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            # No activation - corrections can be positive or negative
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted depth correction."""
        return self.mlp(x).squeeze(-1)


class BathymetricGNN(nn.Module):
    """
    Complete GNN model for bathymetric data processing.
    
    Architecture:
    1. Local feature extraction (MLP)
    2. GNN backbone (message passing)
    3. Task-specific output heads:
       - Classification (noise/feature/seafloor)
       - Confidence estimation
       - Depth correction (optional)
    """
    
    # Class labels
    CLASS_SEAFLOOR = 0
    CLASS_FEATURE = 1
    CLASS_NOISE = 2
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_gnn_layers: int = 4,
        gnn_type: str = "GAT",
        heads: int = 4,
        num_classes: int = 3,
        predict_correction: bool = True,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        """
        Initialize the model.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden dimension size
            num_gnn_layers: Number of GNN layers
            gnn_type: Type of GNN ("GCN", "GAT", "GraphSAGE", "GIN")
            heads: Number of attention heads (for GAT)
            num_classes: Number of output classes
            predict_correction: Whether to predict depth corrections
            dropout: Dropout rate
            edge_dim: Number of edge features (for GAT)
        """
        super().__init__()
        
        self.predict_correction = predict_correction
        self.num_classes = num_classes
        
        # Local feature extractor
        self.feature_extractor = LocalFeatureExtractor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
        )
        
        # GNN backbone
        self.gnn = GNNBackbone(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        
        # Output heads
        self.classification_head = ClassificationHead(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels // 2,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.confidence_head = ConfidenceHead(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels // 2,
            dropout=dropout,
        )
        
        if predict_correction:
            self.correction_head = CorrectionHead(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels // 2,
                dropout=dropout,
            )
        else:
            self.correction_head = None
        
        logger.info(
            f"Created BathymetricGNN: {gnn_type} with {num_gnn_layers} layers, "
            f"{hidden_channels} hidden channels"
        )
    
    def forward(
        self,
        data: Data,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, in_channels]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim] (optional)
                
        Returns:
            Dictionary with:
                - class_logits: [num_nodes, num_classes]
                - class_probs: [num_nodes, num_classes]
                - predicted_class: [num_nodes]
                - confidence: [num_nodes]
                - correction: [num_nodes] (if predict_correction=True)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        
        # Extract local features
        x = self.feature_extractor(x)
        
        # Apply GNN
        x = self.gnn(x, edge_index, edge_attr)
        
        # Get outputs from heads
        class_logits = self.classification_head(x)
        class_probs = F.softmax(class_logits, dim=-1)
        predicted_class = torch.argmax(class_probs, dim=-1)
        confidence = self.confidence_head(x)
        
        outputs = {
            'class_logits': class_logits,
            'class_probs': class_probs,
            'predicted_class': predicted_class,
            'confidence': confidence,
        }
        
        if self.predict_correction:
            correction = self.correction_head(x)
            outputs['correction'] = correction
        
        return outputs
    
    def predict(
        self,
        data: Data,
        auto_correct_threshold: float = 0.85,
        review_threshold: float = 0.6,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with thresholding for deployment.
        
        Args:
            data: Input graph data
            auto_correct_threshold: Auto-correct if confidence > threshold
            review_threshold: Flag for review if confidence < threshold
            
        Returns:
            Dictionary with predictions and review flags
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
        
        confidence = outputs['confidence']
        predicted_class = outputs['predicted_class']
        
        # Determine action for each node
        # 0 = keep as-is, 1 = auto-correct, 2 = flag for review
        action = torch.zeros_like(predicted_class)
        
        # High confidence noise predictions -> auto-correct
        is_noise = predicted_class == self.CLASS_NOISE
        high_conf = confidence > auto_correct_threshold
        action[is_noise & high_conf] = 1
        
        # Low confidence predictions -> review
        low_conf = confidence < review_threshold
        action[low_conf] = 2
        
        outputs['action'] = action
        outputs['needs_review'] = (action == 2)
        outputs['auto_correct'] = (action == 1)
        
        return outputs


class BathymetricGNNLightning(nn.Module):
    """
    PyTorch Lightning wrapper for BathymetricGNN.
    
    Handles training, validation, and logging.
    
    Note: Requires pytorch-lightning to be installed.
    This is a placeholder showing the interface - full implementation
    would go in training/trainer.py
    """
    
    def __init__(self, model: BathymetricGNN, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        return self.model(data)
    
    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Single training step."""
        outputs = self.forward(batch)
        
        # Classification loss
        class_loss = F.cross_entropy(
            outputs['class_logits'],
            batch.y,  # Ground truth labels
        )
        
        # Correction loss (if applicable)
        if self.model.predict_correction and hasattr(batch, 'correction_target'):
            correction_loss = F.mse_loss(
                outputs['correction'],
                batch.correction_target,
            )
        else:
            correction_loss = 0.0
        
        total_loss = class_loss + 0.5 * correction_loss
        
        return total_loss

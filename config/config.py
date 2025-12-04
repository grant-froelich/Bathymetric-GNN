"""
Configuration module for Bathymetric GNN Processing.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from pathlib import Path
import json
import yaml


@dataclass
class TileConfig:
    """Configuration for tile-based processing."""
    tile_size: int = 1024                    # Tile dimensions (pixels)
    overlap: int = 128                        # Overlap between tiles (pixels)
    min_valid_ratio: float = 0.1             # Minimum valid data ratio to process tile
    

@dataclass 
class GraphConfig:
    """Configuration for graph construction."""
    connectivity: str = "8-connected"         # "4-connected" or "8-connected"
    max_edge_distance: float = 2.0           # Maximum edge distance (in grid cells)
    include_self_loops: bool = False
    edge_features: List[str] = field(default_factory=lambda: [
        "distance",
        "depth_difference", 
        "slope",
    ])


@dataclass
class ModelConfig:
    """Configuration for GNN model architecture."""
    # Feature extractor
    local_feature_channels: int = 32
    local_feature_layers: int = 3
    local_kernel_size: int = 5
    
    # GNN
    gnn_type: str = "GAT"                    # "GCN", "GAT", "GraphSAGE", "GIN"
    gnn_hidden_channels: int = 64
    gnn_num_layers: int = 4
    gnn_heads: int = 4                       # For GAT
    gnn_dropout: float = 0.1
    
    # Output heads
    num_classes: int = 3                     # noise, feature, smooth seafloor
    predict_correction: bool = True          # Also predict depth correction


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4                      # Number of tiles per batch
    epochs: int = 100
    
    # Learning rate schedule
    scheduler: str = "cosine"                # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Loss weights
    classification_weight: float = 1.0
    correction_weight: float = 0.5
    confidence_weight: float = 0.2
    
    # Class weights (noise, feature, seafloor)
    class_weights: Optional[List[float]] = None
    
    # Data augmentation
    augment_rotations: bool = True
    augment_flips: bool = True
    augment_noise_intensity: bool = True


@dataclass
class SyntheticNoiseConfig:
    """Configuration for synthetic noise generation."""
    # Noise types to generate
    enable_gaussian: bool = True
    enable_spikes: bool = True
    enable_blobs: bool = True                # Fish, kelp clusters
    enable_systematic: bool = True           # Sonar artifacts
    
    # Intensity ranges
    gaussian_std_range: Tuple[float, float] = (0.1, 0.5)
    spike_magnitude_range: Tuple[float, float] = (1.0, 5.0)
    spike_density_range: Tuple[float, float] = (0.001, 0.01)
    blob_size_range: Tuple[int, int] = (3, 15)
    blob_count_range: Tuple[int, int] = (5, 50)
    
    # Correlation with seafloor complexity
    noise_complexity_correlation: float = 0.3  # More noise in complex areas


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    # Confidence thresholds
    auto_correct_threshold: float = 0.85     # Auto-correct if confidence > this
    review_threshold: float = 0.6            # Flag for review if confidence < this
    
    # Output options
    export_classification: bool = True
    export_confidence: bool = True
    export_correction_magnitude: bool = True
    export_review_priority: bool = True


@dataclass
class Config:
    """Master configuration class."""
    # Sub-configs
    tile: TileConfig = field(default_factory=TileConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    noise: SyntheticNoiseConfig = field(default_factory=SyntheticNoiseConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Paths
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    model_path: Optional[str] = None
    
    # Hardware
    device: str = "cuda"                     # "cuda", "cpu", "mps"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    def save(self, path: Path):
        """Save configuration to YAML file."""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        return cls(
            tile=TileConfig(**data.get('tile', {})),
            graph=GraphConfig(**data.get('graph', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            noise=SyntheticNoiseConfig(**data.get('noise', {})),
            inference=InferenceConfig(**data.get('inference', {})),
            data_dir=data.get('data_dir'),
            output_dir=data.get('output_dir'),
            model_path=data.get('model_path'),
            device=data.get('device', 'cuda'),
            num_workers=data.get('num_workers', 4),
            pin_memory=data.get('pin_memory', True),
            log_level=data.get('log_level', 'INFO'),
            wandb_project=data.get('wandb_project'),
            wandb_entity=data.get('wandb_entity'),
        )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.tile.tile_size > self.tile.overlap * 2, \
            "Tile size must be larger than 2x overlap"
        assert self.model.gnn_type in ["GCN", "GAT", "GraphSAGE", "GIN"], \
            f"Unknown GNN type: {self.model.gnn_type}"
        assert self.graph.connectivity in ["4-connected", "8-connected"], \
            f"Unknown connectivity: {self.graph.connectivity}"

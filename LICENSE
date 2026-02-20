"""
Training utilities for bathymetric GNN.

Includes:
- Dataset creation from clean surveys + synthetic noise
- Dataset creation from ground truth files (real noise)
- Training loop with validation
- Checkpoint management
- Logging and metrics tracking
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

import numpy as np
from tqdm import tqdm

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from config import Config
from data import (
    BathymetricLoader,
    TileManager,
    GraphBuilder,
    SyntheticNoiseGenerator,
    NoiseAugmentor,
)
from models.gnn import BathymetricGNN
from .losses import BathymetricGNNLoss, compute_class_weights

logger = logging.getLogger(__name__)


class GroundTruthDataset(Dataset):
    """
    Dataset that loads training samples from prepared ground truth files.
    
    Ground truth files are GeoTIFFs with 4 bands:
    - Band 1: Labels (0=seafloor, 2=noise, -1=nodata)
    - Band 2: Difference (noisy - clean = correction target)
    - Band 3: Noisy depth
    - Band 4: Clean depth
    """
    
    def __init__(
        self,
        ground_truth_paths: List[Path],
        graph_builder: GraphBuilder,
        tile_size: int = 512,
        overlap: int = 64,
        min_valid_ratio: float = 0.1,
    ):
        """
        Initialize dataset from ground truth files.
        
        Args:
            ground_truth_paths: Paths to ground truth GeoTIFF files
            graph_builder: GraphBuilder for creating graphs
            tile_size: Size of tiles to extract
            overlap: Overlap between tiles
            min_valid_ratio: Minimum ratio of valid cells to include a tile
        """
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required for GroundTruthDataset")
        
        self.graph_builder = graph_builder
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_valid_ratio = min_valid_ratio
        
        # Load all ground truth files and extract tiles
        self.tiles = []
        
        logger.info(f"Loading {len(ground_truth_paths)} ground truth files...")
        
        for gt_path in ground_truth_paths:
            try:
                self._load_ground_truth(gt_path)
            except Exception as e:
                logger.warning(f"Failed to load ground truth {gt_path}: {e}")
        
        logger.info(f"Dataset contains {len(self.tiles)} tiles from {len(ground_truth_paths)} ground truth files")
    
    def _load_ground_truth(self, path: Path):
        """Load a ground truth file and extract tiles."""
        ds = gdal.Open(str(path))
        if ds is None:
            raise IOError(f"Failed to open ground truth file: {path}")
        
        # Read bands
        labels = ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
        difference = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        noisy_depth = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
        clean_depth = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
        
        # Get resolution from geotransform
        gt = ds.GetGeoTransform()
        resolution = (abs(gt[1]), abs(gt[5]))
        
        ds = None
        
        height, width = labels.shape
        stride = self.tile_size - self.overlap
        
        # Extract tiles
        for row_start in range(0, height - self.tile_size + 1, stride):
            for col_start in range(0, width - self.tile_size + 1, stride):
                row_end = row_start + self.tile_size
                col_end = col_start + self.tile_size
                
                # Extract tile data
                tile_labels = labels[row_start:row_end, col_start:col_end]
                tile_diff = difference[row_start:row_end, col_start:col_end]
                tile_noisy = noisy_depth[row_start:row_end, col_start:col_end]
                tile_clean = clean_depth[row_start:row_end, col_start:col_end]
                
                # Valid mask (labels != -1)
                valid_mask = tile_labels >= 0
                valid_ratio = np.sum(valid_mask) / valid_mask.size
                
                if valid_ratio >= self.min_valid_ratio:
                    self.tiles.append({
                        'labels': tile_labels.copy(),
                        'difference': tile_diff.copy(),
                        'noisy_depth': tile_noisy.copy(),
                        'clean_depth': tile_clean.copy(),
                        'valid_mask': valid_mask.copy(),
                        'resolution': resolution,
                        'source': path.stem,
                    })
        
        # Handle edge tiles if there's remaining data
        if height % stride != 0 or width % stride != 0:
            # Bottom-right corner tile
            if height > self.tile_size and width > self.tile_size:
                row_start = height - self.tile_size
                col_start = width - self.tile_size
                
                tile_labels = labels[row_start:, col_start:]
                tile_diff = difference[row_start:, col_start:]
                tile_noisy = noisy_depth[row_start:, col_start:]
                tile_clean = clean_depth[row_start:, col_start:]
                
                valid_mask = tile_labels >= 0
                valid_ratio = np.sum(valid_mask) / valid_mask.size
                
                if valid_ratio >= self.min_valid_ratio:
                    self.tiles.append({
                        'labels': tile_labels.copy(),
                        'difference': tile_diff.copy(),
                        'noisy_depth': tile_noisy.copy(),
                        'clean_depth': tile_clean.copy(),
                        'valid_mask': valid_mask.copy(),
                        'resolution': resolution,
                        'source': path.stem,
                    })
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Data:
        """Get a single training sample."""
        tile = self.tiles[idx]
        
        noisy_depth = tile['noisy_depth']
        valid_mask = tile['valid_mask']
        labels = tile['labels']
        difference = tile['difference']
        resolution = tile['resolution']
        
        # Build graph from noisy data
        # Note: We don't have uncertainty from ground truth, so pass None
        graph = self.graph_builder.build_graph(
            depth=noisy_depth,
            valid_mask=valid_mask,
            uncertainty=None,
            resolution=resolution,
        )
        
        # Add labels to graph
        if graph.num_nodes > 0:
            rows = graph.valid_rows.numpy()
            cols = graph.valid_cols.numpy()
            
            # Classification labels
            graph.y = torch.tensor(labels[rows, cols], dtype=torch.long)
            
            # Correction targets (difference = noisy - clean, so correction to apply is -difference)
            # Model predicts: correction = noisy - clean
            # To recover clean: clean = noisy - correction
            graph.correction_target = torch.tensor(difference[rows, cols], dtype=torch.float32)
            
            # Noise mask (where labels == 2)
            graph.noise_mask = torch.tensor(labels[rows, cols] == 2, dtype=torch.bool)
        else:
            graph.y = torch.tensor([], dtype=torch.long)
            graph.correction_target = torch.tensor([], dtype=torch.float32)
            graph.noise_mask = torch.tensor([], dtype=torch.bool)
        
        return graph


class BathymetricGraphDataset(Dataset):
    """
    Dataset that generates training samples from clean bathymetric data.
    
    For each sample:
    1. Load a tile from a clean survey
    2. Add synthetic noise
    3. Build graph representation
    4. Return (graph, labels)
    """
    
    def __init__(
        self,
        survey_paths: List[Path],
        tile_manager: TileManager,
        graph_builder: GraphBuilder,
        noise_generator: SyntheticNoiseGenerator,
        augment: bool = True,
        cache_tiles: bool = True,
        vr_bag_mode: str = 'resampled',
    ):
        """
        Initialize dataset.
        
        Args:
            survey_paths: Paths to clean survey files
            tile_manager: TileManager for extracting tiles
            graph_builder: GraphBuilder for creating graphs
            noise_generator: SyntheticNoiseGenerator for adding noise
            augment: Whether to apply augmentation
            cache_tiles: Whether to cache extracted tiles
            vr_bag_mode: How to handle VR BAGs ('refinements', 'resampled', 'base')
        """
        self.survey_paths = survey_paths
        self.tile_manager = tile_manager
        self.graph_builder = graph_builder
        self.noise_augmentor = NoiseAugmentor(noise_generator) if augment else None
        self.noise_generator = noise_generator
        
        self.loader = BathymetricLoader(vr_bag_mode=vr_bag_mode)
        
        # Extract all tiles from all surveys
        self.tiles = []
        self.tile_metadata = []  # (survey_idx, tile_row, tile_col)
        
        logger.info(f"Loading {len(survey_paths)} surveys...")
        
        for survey_idx, survey_path in enumerate(survey_paths):
            try:
                grid = self.loader.load(survey_path)
                
                for tile in self.tile_manager.iterate_tiles(grid, skip_empty=True):
                    if cache_tiles:
                        self.tiles.append({
                            'data': tile.data.copy(),
                            'valid_mask': tile.valid_mask.copy(),
                            'uncertainty': tile.uncertainty.copy() if tile.uncertainty is not None else None,
                            'resolution': grid.resolution,
                        })
                    self.tile_metadata.append((survey_idx, tile.tile_row, tile.tile_col))
                    
            except Exception as e:
                logger.warning(f"Failed to load survey {survey_path}: {e}")
        
        self.cache_tiles = cache_tiles
        self.grids = {}  # Cache loaded grids if not caching tiles
        
        logger.info(f"Dataset contains {len(self.tiles)} tiles from {len(survey_paths)} surveys")
    
    def __len__(self) -> int:
        return len(self.tile_metadata)
    
    def __getitem__(self, idx: int) -> Data:
        """Get a single training sample."""
        if self.cache_tiles:
            tile_data = self.tiles[idx]
            clean_depth = tile_data['data']
            valid_mask = tile_data['valid_mask']
            uncertainty = tile_data['uncertainty']
            resolution = tile_data['resolution']
        else:
            # Load on demand (slower but less memory)
            survey_idx, tile_row, tile_col = self.tile_metadata[idx]
            # Implementation would load specific tile...
            raise NotImplementedError("On-demand loading not yet implemented")
        
        # Add synthetic noise
        if self.noise_augmentor is not None:
            noise_result = self.noise_augmentor(clean_depth, valid_mask)
        else:
            noise_result = self.noise_generator.generate(clean_depth, valid_mask)
        
        # Build graph from noisy data
        graph = self.graph_builder.build_graph(
            depth=noise_result.noisy_depth,
            valid_mask=valid_mask,
            uncertainty=uncertainty,
            resolution=resolution,
        )
        
        # Add labels to graph
        if graph.num_nodes > 0:
            # Get labels for valid nodes
            rows = graph.valid_rows.numpy()
            cols = graph.valid_cols.numpy()
            
            # Classification labels (0=seafloor, 1=feature, 2=noise)
            # For synthetic data, we only have noise vs non-noise
            # Map to: 0=clean, 2=noise (no explicit features in synthetic data)
            class_labels = noise_result.classification[rows, cols]
            class_labels = np.where(class_labels == 1, 2, 0)  # Map 1->2 (noise class)
            graph.y = torch.tensor(class_labels, dtype=torch.long)
            
            # Correction targets (how much the depth was changed)
            corrections = noise_result.noisy_depth[rows, cols] - clean_depth[rows, cols]
            graph.correction_target = torch.tensor(corrections, dtype=torch.float32)
            
            # Noise mask for loss computation
            graph.noise_mask = torch.tensor(
                noise_result.noise_mask[rows, cols],
                dtype=torch.bool
            )
        else:
            graph.y = torch.tensor([], dtype=torch.long)
            graph.correction_target = torch.tensor([], dtype=torch.float32)
            graph.noise_mask = torch.tensor([], dtype=torch.bool)
        
        return graph


class Trainer:
    """
    Training manager for bathymetric GNN.
    """
    
    def __init__(
        self,
        config: Config,
        model: BathymetricGNN,
        train_dataset: BathymetricGraphDataset,
        val_dataset: Optional[BathymetricGraphDataset] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: GNN model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Directory for checkpoints and logs
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device with Blackwell compatibility check
        if config.device == "cuda" and torch.cuda.is_available():
            # Check if GPU is actually usable by running a simple operation
            try:
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1
                self.device = torch.device("cuda")
            except RuntimeError as e:
                if "no kernel image" in str(e):
                    logger.warning(
                        "GPU detected but not supported by PyTorch (likely Blackwell/RTX 50 series). "
                        "Falling back to CPU."
                    )
                    self.device = torch.device("cpu")
                else:
                    raise
        else:
            self.device = torch.device("cpu")
        
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = GeometricDataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
            )
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Setup scheduler
        if config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
            )
        elif config.training.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
        else:
            self.scheduler = None
        
        # Setup loss
        self.criterion = BathymetricGNNLoss(
            classification_weight=config.training.classification_weight,
            correction_weight=config.training.correction_weight,
            confidence_weight=config.training.confidence_weight,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized, device: {self.device}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            # Validation epoch
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss - self.config.training.min_delta:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self._save_checkpoint('best_model.pt')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.training.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                logger.info(
                    f"Epoch {epoch+1}/{self.config.training.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.training.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}"
                )
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Final save
        self._save_checkpoint('final_model.pt')
        
        return history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Compute loss
            targets = {
                'class_labels': batch.y,
                'correction_targets': batch.correction_target,
                'noise_mask': batch.noise_mask,
            }
            losses = self.criterion(outputs, targets)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += losses['total'].item() * batch.num_nodes
            correct = (outputs['predicted_class'] == batch.y).sum().item()
            total_correct += correct
            total_samples += batch.num_nodes
            
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'acc': correct / batch.num_nodes if batch.num_nodes > 0 else 0,
            })
        
        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
        }
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]"):
                batch = batch.to(self.device)
                
                outputs = self.model(batch)
                
                targets = {
                    'class_labels': batch.y,
                    'correction_targets': batch.correction_target,
                    'noise_mask': batch.noise_mask,
                }
                losses = self.criterion(outputs, targets)
                
                total_loss += losses['total'].item() * batch.num_nodes
                total_correct += (outputs['predicted_class'] == batch.y).sum().item()
                total_samples += batch.num_nodes
        
        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
        }
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'in_channels': self.model.feature_extractor.mlp[0].in_features,
            'edge_dim': 3,  # Default
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

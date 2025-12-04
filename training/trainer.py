"""
Training utilities for bathymetric GNN.

Includes:
- Dataset creation from clean surveys + synthetic noise
- Training loop with validation
- Checkpoint management
- Logging and metrics tracking
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

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
        """
        self.survey_paths = survey_paths
        self.tile_manager = tile_manager
        self.graph_builder = graph_builder
        self.noise_augmentor = NoiseAugmentor(noise_generator) if augment else None
        self.noise_generator = noise_generator
        
        self.loader = BathymetricLoader()
        
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
        
        # Setup device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
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

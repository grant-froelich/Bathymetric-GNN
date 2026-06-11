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
from contextlib import nullcontext
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
    gdal.UseExceptions()  # Opt in explicitly; silences the GDAL 4.0 FutureWarning
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from config import Config
from data import GraphBuilder
from models.gnn import BathymetricGNN
from .losses import BathymetricGNNLoss, compute_class_weights, compute_correction_delta

logger = logging.getLogger(__name__)

# V9 correction normalization constants
CORRECTION_NORM_FLOOR = 0.01  # minimum local_std to avoid division by near-zero
CORRECTION_NORM_CAP = 50.0    # cap normalized corrections to +/- this many std devs


class GroundTruthDataset(Dataset):
    """
    Dataset that loads training samples from prepared ground truth files.
    
    Supports two file types, auto-detected from band 1 description:
    
    Classification mode (band 1 = 'labels'):
    - Band 1: Labels (0=seafloor, 2=noise, -1=nodata)
    - Band 2: Difference (noisy - clean = correction target)
    - Band 3: Noisy depth
    - Band 4: Clean depth
    - Band 5: Uncertainty (optional)
    
    Regression mode (band 1 = 'valid_mask'):
    - Band 1: Valid mask (1=valid, 0=invalid)
    - Band 2: Correction target in meters (continuous, no thresholding)
    - Band 3: Noisy depth
    - Band 4: Clean depth
    - Band 5: Uncertainty (optional)
    
    Tiles store a 'mode' field indicating which type they came from.
    Mixing modes in one dataset is allowed but discouraged; the model
    and loss function need to handle both consistently.
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
        """Load a ground truth file and extract tiles. Auto-detects mode."""
        ds = gdal.Open(str(path))
        if ds is None:
            raise IOError(f"Failed to open ground truth file: {path}")
        
        # Detect mode from band 1 description
        band1 = ds.GetRasterBand(1)
        band1_desc = (band1.GetDescription() or '').strip().lower()
        
        if band1_desc == 'valid_mask':
            mode = 'regression'
        else:
            # Default to classification for unlabeled bands or 'labels' description
            mode = 'classification'
        
        # Read bands
        band1_data = band1.ReadAsArray()
        difference = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        noisy_depth = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
        clean_depth = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
        
        # Read uncertainty if available (band 5)
        uncertainty = None
        if ds.RasterCount >= 5:
            uncertainty = ds.GetRasterBand(5).ReadAsArray().astype(np.float32)
        
        # Get resolution from geotransform
        gt = ds.GetGeoTransform()
        resolution = (abs(gt[1]), abs(gt[5]))
        
        ds = None
        
        # Build labels and valid_mask depending on mode
        if mode == 'classification':
            labels_full = band1_data.astype(np.int32)
            valid_full = labels_full >= 0
        else:  # regression
            valid_full = band1_data.astype(np.int32) == 1
            # Placeholder labels (not used in regression training, but kept
            # for consistency in the data structure). -1 for invalid cells.
            labels_full = np.where(valid_full, 0, -1).astype(np.int32)
        
        # Depth convention guard. All losses and metrics assume positive-down
        # depth. Ground truth produced before the 2026-06-09 convention fix
        # stored GDAL elevation (negative down), which silently inverted every
        # direction-sensitive semantic (the 3x shoal-safety weighting, the
        # hazard metrics, the shoal/deep split). Refuse such files outright so
        # a stale pre-fix tif can never enter a training or evaluation run.
        if np.any(valid_full):
            median_noisy = float(np.median(noisy_depth[valid_full]))
            if median_noisy < 0:
                raise ValueError(
                    f"{path.name}: median valid depth is {median_noisy:.1f} "
                    f"(negative). This ground truth file stores elevation "
                    f"(negative-down) and predates the depth-convention fix. "
                    f"Regenerate it with the current prepare_ground_truth.py "
                    f"before training or evaluating."
                )
        
        logger.info(f"  Loaded {path.name} in {mode} mode")
        
        height, width = band1_data.shape
        stride = self.tile_size - self.overlap
        
        def _maybe_add_tile(rs, re, cs, ce):
            tile_labels = labels_full[rs:re, cs:ce]
            tile_diff = difference[rs:re, cs:ce]
            tile_noisy = noisy_depth[rs:re, cs:ce]
            tile_clean = clean_depth[rs:re, cs:ce]
            tile_valid = valid_full[rs:re, cs:ce]
            tile_uncert = uncertainty[rs:re, cs:ce] if uncertainty is not None else None
            
            valid_ratio = np.sum(tile_valid) / tile_valid.size
            if valid_ratio >= self.min_valid_ratio:
                self.tiles.append({
                    'mode': mode,
                    'labels': tile_labels.copy(),
                    'difference': tile_diff.copy(),
                    'noisy_depth': tile_noisy.copy(),
                    'clean_depth': tile_clean.copy(),
                    'uncertainty': tile_uncert.copy() if tile_uncert is not None else None,
                    'valid_mask': tile_valid.copy(),
                    'resolution': resolution,
                    'source': path.stem,
                })
        
        # Tile start positions covering the FULL grid in each dimension.
        # The regular stride positions cover the interior; if they end short of
        # the edge, one final tile is anchored to the edge (start = dim -
        # tile_size) so the right and bottom strips are always covered. The
        # previous implementation only added a single bottom-right corner tile,
        # leaving up to stride-1 pixels of every right and bottom edge in no
        # tile at all (4-15% of cells at typical settings) and therefore
        # excluded from both training and evaluation.
        def _tile_starts(dim: int) -> List[int]:
            if dim <= self.tile_size:
                return [0]
            starts = list(range(0, dim - self.tile_size + 1, stride))
            if starts[-1] + self.tile_size < dim:
                starts.append(dim - self.tile_size)
            return starts
        
        for row_start in _tile_starts(height):
            row_end = min(row_start + self.tile_size, height)
            for col_start in _tile_starts(width):
                col_end = min(col_start + self.tile_size, width)
                _maybe_add_tile(row_start, row_end, col_start, col_end)
    
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
        uncertainty = tile.get('uncertainty', None)
        mode = tile.get('mode', 'classification')
        
        # Build graph from noisy data (provides local_std for normalization)
        graph = self.graph_builder.build_graph(
            depth=noisy_depth,
            valid_mask=valid_mask,
            uncertainty=uncertainty,
            resolution=resolution,
        )
        
        # Attach training targets to the graph
        if graph.num_nodes > 0:
            rows = graph.valid_rows.numpy()
            cols = graph.valid_cols.numpy()
            
            # Raw correction in meters
            raw_correction = difference[rows, cols].astype(np.float32)
            
            # V9 normalization: divide by per-node local_std so the model learns
            # corrections in std-dev units. Floor at 0.01m to avoid division by
            # near-zero in flat areas; cap at +/-50 std-devs to bound outliers.
            if hasattr(graph, 'local_std') and graph.local_std is not None:
                local_std = graph.local_std.detach().cpu().numpy()
                denom = np.maximum(local_std, CORRECTION_NORM_FLOOR)
                normalized = raw_correction / denom
                normalized = np.clip(normalized, -CORRECTION_NORM_CAP, CORRECTION_NORM_CAP)
                graph.correction_target = torch.tensor(normalized, dtype=torch.float32)
            else:
                graph.correction_target = torch.tensor(raw_correction, dtype=torch.float32)
            
            # Mode flag so the loss function knows what to compute
            graph.mode = mode
            
            if mode == 'classification':
                # Classification labels (0=seafloor, 2=noise)
                graph.y = torch.tensor(labels[rows, cols], dtype=torch.long)
                # Noise mask used by correction-on-noise-only logic
                graph.noise_mask = torch.tensor(labels[rows, cols] == 2, dtype=torch.bool)
            else:
                # Regression mode: no classification labels
                # Provide empty placeholder for graph.y so downstream code that
                # accesses it doesn't crash; loss function should check graph.mode.
                graph.y = torch.full((graph.num_nodes,), -1, dtype=torch.long)
                # In regression mode every valid cell contributes to the correction loss
                graph.noise_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
        else:
            graph.correction_target = torch.tensor([], dtype=torch.float32)
            graph.mode = mode
            graph.y = torch.tensor([], dtype=torch.long)
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
        train_dataset: GroundTruthDataset,
        val_dataset: Optional[GroundTruthDataset] = None,
        output_dir: Optional[Path] = None,
        use_amp: bool = False,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: GNN model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Directory for checkpoints and logs
            use_amp: If True, run the forward pass and loss under bf16 autocast
                (mixed precision). Speeds up the GAT matmuls on CUDA; bf16 needs
                no GradScaler. Defaults to False (full fp32, unchanged behavior)
                so existing callers are unaffected.
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed-precision (bf16 autocast) setting
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16
        
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
        
        # Setup data loaders. When using worker processes, keep them alive across
        # epochs (Windows uses spawn, so workers are otherwise recreated every
        # epoch) and prefetch more batches per worker so the GPU waits less
        # between iterations. Gated on num_workers > 0: with 0 workers these
        # options are invalid, so the loaders fall back to default behavior.
        loader_kwargs = {}
        if config.num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 4
        
        self.train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            **loader_kwargs,
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = GeometricDataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                **loader_kwargs,
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
        
        # Setup loss with class weights and correction delta computed from training data
        class_weights, correction_delta, training_mode = self._compute_training_stats()
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            logger.info(f"Class weights: {class_weights.tolist()}")
        logger.info(f"Correction Huber delta: {correction_delta:.3f}")
        logger.info(f"Training mode: {training_mode}")
        
        self.criterion = BathymetricGNNLoss(
            class_weights=class_weights,
            classification_weight=config.training.classification_weight,
            correction_weight=config.training.correction_weight,
            confidence_weight=config.training.confidence_weight,
            correction_delta=correction_delta,
            regression_delta=correction_delta,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized, device: {self.device}")
        if self.use_amp and self.device.type == 'cuda':
            logger.info("Mixed precision enabled (bf16 autocast)")
        elif self.use_amp:
            logger.info("Mixed precision requested but device is not CUDA; running fp32")
    
    def _compute_training_stats(self) -> Tuple[Optional[torch.Tensor], float, str]:
        """
        Compute class weights and correction Huber delta from training dataset.
        
        Scans all tiles in the training dataset to:
        1. Count per-class samples for inverse-frequency class weights (classification only)
        2. Sample actual normalized corrections from the dataset to set Huber delta
        3. Detect dominant mode (classification or regression) across the dataset
        
        For the Huber delta, this method samples real graphs from the dataset
        and collects the NORMALIZED correction targets the model actually sees
        during training. Using raw correction magnitudes here would set delta
        to a value far larger than any prediction error, putting the Huber
        loss in pure linear mode and defeating its purpose.
        
        Returns:
            (class_weights, correction_delta, mode)
            class_weights is None if dataset is regression-only.
        """
        all_labels = []
        mode_counts = {'classification': 0, 'regression': 0}
        
        for tile in self.train_dataset.tiles:
            mode = tile.get('mode', 'classification')
            mode_counts[mode] += 1
            
            if mode == 'classification':
                labels = tile['labels']
                valid_mask = tile['valid_mask']
                all_labels.append(labels[valid_mask].flatten())
        
        # Determine dominant mode
        dominant_mode = max(mode_counts, key=mode_counts.get)
        
        # Class weights (classification mode only)
        class_weights = None
        if all_labels:
            labels_tensor = torch.tensor(np.concatenate(all_labels), dtype=torch.long)
            class_weights = compute_class_weights(labels_tensor, num_classes=3)
        
        # Sample normalized corrections from the dataset to compute Huber delta.
        # Building graphs is expensive, so we sample a subset of tiles.
        n_tiles = len(self.train_dataset)
        sample_size = min(50, n_tiles)
        if sample_size > 0:
            rng = np.random.default_rng(seed=42)
            sample_indices = rng.choice(n_tiles, size=sample_size, replace=False)
            
            normalized_corrections = []
            logger.info(f"Sampling {sample_size} tiles to compute Huber delta from normalized corrections...")
            for idx in sample_indices:
                try:
                    graph = self.train_dataset[int(idx)]
                    if graph.num_nodes > 0 and hasattr(graph, 'correction_target'):
                        ct = graph.correction_target.detach().cpu().numpy()
                        if ct.size > 0:
                            normalized_corrections.append(np.abs(ct))
                except Exception as e:
                    logger.warning(f"Skipping tile {idx} during delta computation: {e}")
            
            if normalized_corrections:
                corrections_arr = np.concatenate(normalized_corrections)
                correction_delta = compute_correction_delta(
                    corrections_arr, percentile=95.0, min_delta=1.0
                )
            else:
                correction_delta = 1.0
        else:
            correction_delta = 1.0
        
        return class_weights, correction_delta, dominant_mode
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'metric_name': None,  # 'accuracy' (classification) or 'mae' (regression)
        }
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            # Metric key differs by mode: 'accuracy' (classification) or 'mae' (regression)
            train_metric_key = 'accuracy' if 'accuracy' in train_metrics else 'mae'
            train_metric_label = 'Acc' if train_metric_key == 'accuracy' else 'MAE'
            history['metric_name'] = train_metric_key
            history['train_metric'].append(train_metrics.get(train_metric_key, 0.0))
            
            # Validation epoch
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()
                history['val_loss'].append(val_metrics['loss'])
                val_metric_key = 'accuracy' if 'accuracy' in val_metrics else 'mae'
                history['val_metric'].append(val_metrics.get(val_metric_key, 0.0))
                
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
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
                
                logger.info(
                    f"Epoch {epoch+1}/{self.config.training.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val {train_metric_label}: {val_metrics.get(val_metric_key, 0.0):.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.training.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train {train_metric_label}: {train_metrics.get(train_metric_key, 0.0):.4f}"
                )
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Persist per-epoch history so progress survives interruption
            self._save_history(history)
        
        # Final save
        self._save_checkpoint('final_model.pt')
        self._save_history(history)
        
        return history
    
    def _save_history(self, history):
        """Write the training history dict to JSON, overwriting each epoch."""
        import json
        history_path = self.output_dir / 'training_history.json'
        try:
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")
    def _autocast(self):
        """bf16 autocast context when AMP is on and device is CUDA, else no-op.
        
        bf16 (not fp16) so no GradScaler is required: it shares fp32's exponent
        range. Master weights stay fp32; only the matmuls inside the context run
        in bf16, while reductions and softmax stay fp32.
        """
        if self.use_amp and self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass (under bf16 autocast when AMP is enabled)
            self.optimizer.zero_grad()
            with self._autocast():
                outputs = self.model(batch)
                
                # Detect mode (PyG batches string attrs as lists, one per graph)
                batch_mode = getattr(batch, 'mode', 'classification')
                if isinstance(batch_mode, list):
                    if batch_mode and any(m != batch_mode[0] for m in batch_mode):
                        raise ValueError(
                            f"Mixed-mode batch: {set(batch_mode)}. Classification and "
                            f"regression ground truth files must not share a dataset."
                        )
                    batch_mode = batch_mode[0] if batch_mode else 'classification'
                
                # Build targets dict
                targets = {
                    'mode': batch_mode,
                    'class_labels': batch.y,
                    'correction_targets': batch.correction_target,
                    'noise_mask': batch.noise_mask,
                }
                if batch_mode == 'regression':
                    # In regression mode, every node in the graph is a valid cell
                    # (the dataset already filtered to valid cells before building the graph)
                    targets['valid_mask'] = torch.ones(batch.num_nodes, dtype=torch.bool, device=self.device)
                
                losses = self.criterion(outputs, targets)
            
            # Backward pass (outside autocast; bf16 needs no GradScaler)
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += losses['total'].item() * batch.num_nodes
            total_samples += batch.num_nodes
            
            if batch_mode == 'regression':
                # Track mean absolute correction error instead of classification accuracy
                with torch.no_grad():
                    if 'correction' in outputs:
                        mae = torch.abs(outputs['correction'] - batch.correction_target).mean().item()
                    else:
                        mae = 0.0
                total_correct += mae * batch.num_nodes  # store sum, divide later
                pbar.set_postfix({'loss': losses['total'].item(), 'mae': mae})
            else:
                correct = (outputs['predicted_class'] == batch.y).sum().item()
                total_correct += correct
                pbar.set_postfix({
                    'loss': losses['total'].item(),
                    'acc': correct / batch.num_nodes if batch.num_nodes > 0 else 0,
                })
        
        metrics = {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
        }
        if total_samples > 0:
            # In regression mode this is MAE; in classification mode it's accuracy
            metrics['accuracy' if batch_mode == 'classification' else 'mae'] = (
                total_correct / total_samples
            )
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_mode = 'classification'
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]"):
                batch = batch.to(self.device)
                
                with self._autocast():
                    outputs = self.model(batch)
                    
                    batch_mode = getattr(batch, 'mode', 'classification')
                    if isinstance(batch_mode, list):
                        if batch_mode and any(m != batch_mode[0] for m in batch_mode):
                            raise ValueError(
                                f"Mixed-mode batch: {set(batch_mode)}. Classification and "
                                f"regression ground truth files must not share a dataset."
                            )
                        batch_mode = batch_mode[0] if batch_mode else 'classification'
                    
                    targets = {
                        'mode': batch_mode,
                        'class_labels': batch.y,
                        'correction_targets': batch.correction_target,
                        'noise_mask': batch.noise_mask,
                    }
                    if batch_mode == 'regression':
                        targets['valid_mask'] = torch.ones(batch.num_nodes, dtype=torch.bool, device=self.device)
                    
                    losses = self.criterion(outputs, targets)
                
                total_loss += losses['total'].item() * batch.num_nodes
                total_samples += batch.num_nodes
                
                if batch_mode == 'regression':
                    if 'correction' in outputs:
                        mae = torch.abs(outputs['correction'] - batch.correction_target).mean().item()
                    else:
                        mae = 0.0
                    total_correct += mae * batch.num_nodes
                else:
                    total_correct += (outputs['predicted_class'] == batch.y).sum().item()
        
        metrics = {
            'loss': total_loss / total_samples if total_samples > 0 else 0,
        }
        if total_samples > 0:
            metrics['accuracy' if batch_mode == 'classification' else 'mae'] = (
                total_correct / total_samples
            )
        return metrics
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'in_channels': getattr(self.model, 'in_channels',
                                    self.model.feature_extractor.mlp[0].in_features),
            'edge_dim': getattr(self.model, 'edge_dim', 3)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

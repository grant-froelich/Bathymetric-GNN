"""
Synthetic noise generation for bathymetric training data.

Generates realistic noise patterns to add to clean reference surveys,
enabling training without large paired noisy/clean datasets.

Noise types:
- Gaussian: Environmental/sensor noise
- Spikes: Double returns, multipath
- Blobs: Fish, kelp, suspended sediment
- Systematic: Sonar artifacts, refraction errors
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import truncnorm

logger = logging.getLogger(__name__)


@dataclass
class NoiseLabel:
    """Labels for synthetic noise generation."""
    noisy_depth: np.ndarray           # Depth with noise added
    clean_depth: np.ndarray           # Original clean depth
    noise_mask: np.ndarray            # Boolean mask where noise was added
    noise_magnitude: np.ndarray       # Magnitude of noise at each point
    classification: np.ndarray        # Per-pixel class (0=clean, 1=noise)


class SyntheticNoiseGenerator:
    """
    Generates realistic synthetic noise for bathymetric data.
    
    The noise patterns are designed to mimic real acoustic artifacts
    while providing ground truth labels for training.
    """
    
    def __init__(
        self,
        # Gaussian noise
        enable_gaussian: bool = True,
        gaussian_std_range: Tuple[float, float] = (0.1, 0.5),
        
        # Spike noise
        enable_spikes: bool = True,
        spike_magnitude_range: Tuple[float, float] = (1.0, 5.0),
        spike_density_range: Tuple[float, float] = (0.001, 0.01),
        
        # Blob noise (fish, kelp)
        enable_blobs: bool = True,
        blob_size_range: Tuple[int, int] = (3, 15),
        blob_count_range: Tuple[int, int] = (5, 50),
        blob_magnitude_range: Tuple[float, float] = (0.5, 3.0),
        
        # Systematic noise (sonar artifacts)
        enable_systematic: bool = True,
        systematic_amplitude_range: Tuple[float, float] = (0.2, 1.0),
        
        # Correlation with seafloor complexity
        complexity_correlation: float = 0.3,
        
        # Random seed
        seed: Optional[int] = None,
    ):
        """
        Initialize noise generator.
        
        Args:
            enable_*: Whether to generate each noise type
            *_range: (min, max) ranges for noise parameters
            complexity_correlation: How much noise density correlates with seafloor complexity
            seed: Random seed for reproducibility
        """
        self.enable_gaussian = enable_gaussian
        self.gaussian_std_range = gaussian_std_range
        
        self.enable_spikes = enable_spikes
        self.spike_magnitude_range = spike_magnitude_range
        self.spike_density_range = spike_density_range
        
        self.enable_blobs = enable_blobs
        self.blob_size_range = blob_size_range
        self.blob_count_range = blob_count_range
        self.blob_magnitude_range = blob_magnitude_range
        
        self.enable_systematic = enable_systematic
        self.systematic_amplitude_range = systematic_amplitude_range
        
        self.complexity_correlation = complexity_correlation
        
        self.rng = np.random.default_rng(seed)
    
    def generate(
        self,
        clean_depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        intensity: float = 1.0,
    ) -> NoiseLabel:
        """
        Add synthetic noise to clean bathymetric data.
        
        Args:
            clean_depth: Clean depth array
            valid_mask: Mask of valid depth cells
            intensity: Overall noise intensity multiplier (0-2)
            
        Returns:
            NoiseLabel with noisy data and ground truth labels
        """
        if valid_mask is None:
            valid_mask = np.isfinite(clean_depth)
        
        # Initialize outputs
        noisy_depth = clean_depth.copy()
        noise_mask = np.zeros(clean_depth.shape, dtype=bool)
        noise_magnitude = np.zeros(clean_depth.shape, dtype=np.float32)
        
        # Compute local complexity for spatially-varying noise
        complexity = self._compute_complexity(clean_depth, valid_mask)
        
        # Compute depth statistics for scaling
        valid_depths = clean_depth[valid_mask]
        if len(valid_depths) == 0:
            return NoiseLabel(
                noisy_depth=noisy_depth,
                clean_depth=clean_depth,
                noise_mask=noise_mask,
                noise_magnitude=noise_magnitude,
                classification=np.zeros(clean_depth.shape, dtype=np.int64),
            )
        
        depth_std = np.std(valid_depths)
        depth_range = np.ptp(valid_depths)
        
        # Apply each noise type
        if self.enable_gaussian:
            self._add_gaussian_noise(
                noisy_depth, valid_mask, noise_mask, noise_magnitude,
                depth_std, intensity
            )
        
        if self.enable_spikes:
            self._add_spike_noise(
                noisy_depth, valid_mask, noise_mask, noise_magnitude,
                depth_range, complexity, intensity
            )
        
        if self.enable_blobs:
            self._add_blob_noise(
                noisy_depth, valid_mask, noise_mask, noise_magnitude,
                depth_range, intensity
            )
        
        if self.enable_systematic:
            self._add_systematic_noise(
                noisy_depth, valid_mask, noise_mask, noise_magnitude,
                depth_std, intensity
            )
        
        # Create classification labels
        classification = np.where(noise_mask, 1, 0).astype(np.int64)
        
        logger.debug(
            f"Generated noise: {np.sum(noise_mask)} noisy cells "
            f"({100 * np.sum(noise_mask) / np.sum(valid_mask):.1f}% of valid)"
        )
        
        return NoiseLabel(
            noisy_depth=noisy_depth,
            clean_depth=clean_depth,
            noise_mask=noise_mask,
            noise_magnitude=noise_magnitude,
            classification=classification,
        )
    
    def _compute_complexity(
        self,
        depth: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute local seafloor complexity (roughness).
        
        Used to modulate noise density - more noise in complex areas
        where it's harder to distinguish from real features.
        """
        # Local standard deviation as complexity measure
        depth_filled = np.where(valid_mask, depth, np.nanmean(depth))
        local_std = ndimage.generic_filter(
            depth_filled,
            np.std,
            size=11,
            mode='nearest'
        )
        
        # Normalize to 0-1 range
        if local_std.max() > local_std.min():
            complexity = (local_std - local_std.min()) / (local_std.max() - local_std.min())
        else:
            complexity = np.zeros_like(local_std)
        
        return complexity
    
    def _add_gaussian_noise(
        self,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        noise_mask: np.ndarray,
        noise_magnitude: np.ndarray,
        depth_std: float,
        intensity: float,
    ):
        """Add Gaussian environmental noise."""
        std_min, std_max = self.gaussian_std_range
        noise_std = self.rng.uniform(std_min, std_max) * depth_std * intensity
        
        gaussian_noise = self.rng.normal(0, noise_std, depth.shape).astype(np.float32)
        
        # Apply only to valid cells
        depth[valid_mask] += gaussian_noise[valid_mask]
        
        # Gaussian noise is everywhere, but we don't mark it as "noise to remove"
        # since it's typically below the detection threshold
        # Only mark significant deviations
        significant = np.abs(gaussian_noise) > 2 * noise_std
        noise_mask[valid_mask & significant] = True
        noise_magnitude[valid_mask] = np.maximum(
            noise_magnitude[valid_mask],
            np.abs(gaussian_noise[valid_mask])
        )
    
    def _add_spike_noise(
        self,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        noise_mask: np.ndarray,
        noise_magnitude: np.ndarray,
        depth_range: float,
        complexity: np.ndarray,
        intensity: float,
    ):
        """Add spike noise (double returns, multipath)."""
        # Base density modulated by complexity
        density_min, density_max = self.spike_density_range
        base_density = self.rng.uniform(density_min, density_max) * intensity
        
        # Spatially varying density
        density_map = base_density * (
            1 + self.complexity_correlation * (complexity - 0.5)
        )
        
        # Generate spike locations
        spike_prob = self.rng.random(depth.shape)
        spike_locations = (spike_prob < density_map) & valid_mask
        
        # Generate spike magnitudes
        mag_min, mag_max = self.spike_magnitude_range
        num_spikes = np.sum(spike_locations)
        
        if num_spikes > 0:
            # Mix of positive and negative spikes
            signs = self.rng.choice([-1, 1], size=num_spikes)
            magnitudes = self.rng.uniform(mag_min, mag_max, size=num_spikes) * depth_range * intensity
            
            spike_values = signs * magnitudes
            depth[spike_locations] += spike_values
            
            noise_mask[spike_locations] = True
            noise_magnitude[spike_locations] = np.abs(spike_values)
        
        logger.debug(f"Added {num_spikes} spike noise points")
    
    def _add_blob_noise(
        self,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        noise_mask: np.ndarray,
        noise_magnitude: np.ndarray,
        depth_range: float,
        intensity: float,
    ):
        """Add blob noise (fish, kelp, suspended matter)."""
        count_min, count_max = self.blob_count_range
        num_blobs = self.rng.integers(
            int(count_min * intensity),
            int(count_max * intensity) + 1
        )
        
        height, width = depth.shape
        
        for _ in range(num_blobs):
            # Random blob center in valid region
            valid_indices = np.argwhere(valid_mask)
            if len(valid_indices) == 0:
                continue
            
            center_idx = self.rng.integers(len(valid_indices))
            center_r, center_c = valid_indices[center_idx]
            
            # Random blob size
            size_min, size_max = self.blob_size_range
            blob_size = self.rng.integers(size_min, size_max + 1)
            
            # Create blob mask using distance from center
            r_coords, c_coords = np.ogrid[:height, :width]
            distance = np.sqrt((r_coords - center_r)**2 + (c_coords - center_c)**2)
            
            # Soft blob with Gaussian falloff
            blob_mask = distance < blob_size
            blob_weight = np.exp(-distance**2 / (2 * (blob_size/2)**2))
            
            # Random blob magnitude (usually positive - objects in water column)
            mag_min, mag_max = self.blob_magnitude_range
            blob_magnitude = self.rng.uniform(mag_min, mag_max) * depth_range * intensity
            
            # Occasionally negative (shadows)
            if self.rng.random() < 0.2:
                blob_magnitude = -blob_magnitude
            
            # Apply blob
            blob_area = blob_mask & valid_mask
            depth[blob_area] += (blob_weight * blob_magnitude)[blob_area]
            
            noise_mask[blob_area] = True
            noise_magnitude[blob_area] = np.maximum(
                noise_magnitude[blob_area],
                np.abs((blob_weight * blob_magnitude)[blob_area])
            )
        
        logger.debug(f"Added {num_blobs} blob noise regions")
    
    def _add_systematic_noise(
        self,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        noise_mask: np.ndarray,
        noise_magnitude: np.ndarray,
        depth_std: float,
        intensity: float,
    ):
        """Add systematic noise (sonar artifacts, refraction errors)."""
        height, width = depth.shape
        
        # Choose artifact type
        artifact_type = self.rng.choice(['stripe', 'wave', 'gradient'])
        
        amp_min, amp_max = self.systematic_amplitude_range
        amplitude = self.rng.uniform(amp_min, amp_max) * depth_std * intensity
        
        if artifact_type == 'stripe':
            # Horizontal or vertical stripes (along-track artifacts)
            orientation = self.rng.choice(['horizontal', 'vertical'])
            frequency = self.rng.uniform(0.01, 0.05)
            
            if orientation == 'horizontal':
                coords = np.arange(height)[:, np.newaxis] * np.ones((1, width))
            else:
                coords = np.ones((height, 1)) * np.arange(width)[np.newaxis, :]
            
            artifact = amplitude * np.sin(2 * np.pi * frequency * coords)
        
        elif artifact_type == 'wave':
            # Undulating pattern (refraction artifacts)
            freq_x = self.rng.uniform(0.005, 0.02)
            freq_y = self.rng.uniform(0.005, 0.02)
            phase = self.rng.uniform(0, 2 * np.pi)
            
            x_coords = np.arange(width)[np.newaxis, :] * np.ones((height, 1))
            y_coords = np.arange(height)[:, np.newaxis] * np.ones((1, width))
            
            artifact = amplitude * np.sin(
                2 * np.pi * (freq_x * x_coords + freq_y * y_coords) + phase
            )
        
        else:  # gradient
            # Linear gradient (calibration drift)
            direction = self.rng.choice(['x', 'y', 'diagonal'])
            
            if direction == 'x':
                artifact = amplitude * np.linspace(-1, 1, width)[np.newaxis, :] * np.ones((height, 1))
            elif direction == 'y':
                artifact = amplitude * np.linspace(-1, 1, height)[:, np.newaxis] * np.ones((1, width))
            else:
                x_grad = np.linspace(-1, 1, width)[np.newaxis, :]
                y_grad = np.linspace(-1, 1, height)[:, np.newaxis]
                artifact = amplitude * (x_grad + y_grad) / 2
        
        artifact = artifact.astype(np.float32)
        
        # Apply to valid regions
        depth[valid_mask] += artifact[valid_mask]
        
        # Mark significant systematic deviations
        significant = np.abs(artifact) > amplitude * 0.5
        noise_mask[valid_mask & significant] = True
        noise_magnitude[valid_mask] = np.maximum(
            noise_magnitude[valid_mask],
            np.abs(artifact[valid_mask])
        )
        
        logger.debug(f"Added {artifact_type} systematic noise pattern")


class NoiseAugmentor:
    """
    Applies noise augmentation during training.
    
    Randomly varies noise parameters to improve model robustness.
    """
    
    def __init__(
        self,
        base_generator: SyntheticNoiseGenerator,
        intensity_range: Tuple[float, float] = (0.5, 1.5),
        seed: Optional[int] = None,
    ):
        """
        Initialize augmentor.
        
        Args:
            base_generator: Base noise generator
            intensity_range: Range of intensity multipliers
            seed: Random seed
        """
        self.generator = base_generator
        self.intensity_range = intensity_range
        self.rng = np.random.default_rng(seed)
    
    def __call__(
        self,
        clean_depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> NoiseLabel:
        """Generate augmented noisy sample."""
        intensity = self.rng.uniform(*self.intensity_range)
        return self.generator.generate(clean_depth, valid_mask, intensity)

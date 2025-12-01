"""Long-distance optimization module for Air-Pen.

Solves pixel quantization noise and jitter when the user stands far from the webcam
by adaptively scaling filter cutoff frequencies and expanding UI hitboxes.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple
from config.settings import (
    FILTER_BETA,
    FILTER_MIN_CUTOFF,
    MAX_SCALE_FACTOR,
    MIN_SCALE_FACTOR,
    REFERENCE_PALM_SIZE,
)


class DistanceZone(str, Enum):
    """Estimated physical user distance category."""
    CLOSE = "Close (<0.8m)"
    OPTIMAL = "Optimal (0.8m - 1.8m)"
    FAR = "Far (1.8m - 2.8m)"
    EXTREME = "Extreme (>2.8m)"


@dataclass
class AdaptiveParameters:
    """Dynamically adjusted parameters for current distance."""
    zone: DistanceZone
    scale_factor: float
    filter_min_cutoff: float
    filter_beta: float
    ui_hitbox_padding: int
    brush_display_scale: float


class LongDistanceOptimizer:
    """Monitors palm scale and tunes filter and UI sensitivity dynamically."""

    def __init__(self, reference_palm_size: float = REFERENCE_PALM_SIZE):
        self.reference_palm_size = reference_palm_size

    def analyze(self, current_palm_size: float) -> AdaptiveParameters:
        """Compute distance category and dynamic parameter adaptations."""
        # Scale factor relative to nominal close-distance reference
        raw_scale = current_palm_size / max(1.0, self.reference_palm_size)
        clamped_scale = max(MIN_SCALE_FACTOR, min(MAX_SCALE_FACTOR, raw_scale))

        # Categorize distance zone
        if clamped_scale > 1.25:
            zone = DistanceZone.CLOSE
        elif clamped_scale >= 0.70:
            zone = DistanceZone.OPTIMAL
        elif clamped_scale >= 0.45:
            zone = DistanceZone.FAR
        else:
            zone = DistanceZone.EXTREME

        # Adaptive filter cutoff:
        # At long distances (small scale), high-frequency jitter increases due to pixel quantization.
        # We lower min_cutoff to strengthen low-pass filtering on stationary/slow points.
        adaptive_min_cutoff = FILTER_MIN_CUTOFF * (0.4 + 0.6 * clamped_scale)
        adaptive_min_cutoff = max(0.35, min(2.5, adaptive_min_cutoff))

        # We also slightly reduce beta at extreme distances to avoid snapping to sensor noise
        adaptive_beta = FILTER_BETA * (0.5 + 0.5 * clamped_scale)

        # UI Hitbox padding: when standing far away, expand clickable target margins
        # (inverse relationship to scale)
        hitbox_padding = int(max(0.0, (1.0 - clamped_scale) * 28.0))

        # Subtle brush display scaling to keep ink width visually consistent
        brush_display_scale = max(0.8, min(1.2, math.sqrt(clamped_scale)))

        return AdaptiveParameters(
            zone=zone,
            scale_factor=clamped_scale,
            filter_min_cutoff=adaptive_min_cutoff,
            filter_beta=adaptive_beta,
            ui_hitbox_padding=hitbox_padding,
            brush_display_scale=brush_display_scale
        )

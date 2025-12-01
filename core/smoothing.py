"""One-Euro Filter (1€ Filter) for Sub-Millimeter Jitter Elimination.

Casiez, G., Roussel, N. and Vogel, D. (2012).
1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Human-Computer Interaction.
ACM Conference on Human Factors in Computing Systems (CHI '12).
"""

import math
import time
from typing import Optional, Tuple


class LowPassFilter:
    """First-order low-pass exponential smoothing filter."""

    def __init__(self, alpha: float = 0.5):
        self._alpha = alpha
        self._hat_x_prev: Optional[float] = None

    def __call__(self, x: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self._alpha = alpha

        if self._hat_x_prev is None:
            hat_x = x
        else:
            hat_x = self._alpha * x + (1.0 - self._alpha) * self._hat_x_prev

        self._hat_x_prev = hat_x
        return hat_x

    def reset(self) -> None:
        self._hat_x_prev = None


class OneEuroFilter:
    """1-Dimensional One-Euro Filter.

    Adapts the cutoff frequency based on input signal speed:
    - Low speed: Lower cutoff -> Heavy smoothing -> No jitter/tremor
    - High speed: Higher cutoff -> Light smoothing -> Minimal latency / no lag
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._t_prev: Optional[float] = None
        self._x_prev: Optional[float] = None

    def _compute_alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.0
        return 1.0 / (1.0 + tau / te) if (tau + te) > 0 else 1.0

    def __call__(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.perf_counter()

        if self._t_prev is None or self._x_prev is None:
            self._t_prev = timestamp
            self._x_prev = x
            return x

        dt = timestamp - self._t_prev
        if dt <= 0.0:
            dt = 1e-5  # Prevent division by zero

        rate = 1.0 / dt

        # Estimate instantaneous velocity and filter it
        dx = (x - self._x_prev) * rate
        edx = self._dx_filter(dx, self._compute_alpha(rate, self.d_cutoff))

        # Dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter the position with the dynamic cutoff
        filtered_x = self._x_filter(x, self._compute_alpha(rate, cutoff))

        self._t_prev = timestamp
        self._x_prev = filtered_x
        return filtered_x

    def reset(self) -> None:
        self._x_filter.reset()
        self._dx_filter.reset()
        self._t_prev = None
        self._x_prev = None


class OneEuroFilter2D:
    """2D Point filter for smoothing coordinates (x, y) simultaneously."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def set_parameters(self, min_cutoff: float, beta: float) -> None:
        """Dynamically update parameters (e.g., for long-distance tuning)."""
        self.fx.min_cutoff = min_cutoff
        self.fx.beta = beta
        self.fy.min_cutoff = min_cutoff
        self.fy.beta = beta

    def __call__(
        self,
        point: Tuple[float, float],
        timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        if timestamp is None:
            timestamp = time.perf_counter()

        x, y = point
        return (
            self.fx(float(x), timestamp),
            self.fy(float(y), timestamp)
        )

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

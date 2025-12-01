"""Unit tests for the One-Euro Filter (1€ Filter)."""

import math
import numpy as np
import pytest
from core.smoothing import LowPassFilter, OneEuroFilter, OneEuroFilter2D


def test_low_pass_filter():
    """Verify first-order exponential smoothing."""
    lpf = LowPassFilter(alpha=0.5)
    # First step should initialize
    val1 = lpf(10.0)
    assert val1 == 10.0

    # Second step: 0.5 * 20 + 0.5 * 10 = 15.0
    val2 = lpf(20.0)
    assert pytest.approx(val2, 0.001) == 15.0

    lpf.reset()
    assert lpf(50.0) == 50.0


def test_one_euro_filter_static_noise_reduction():
    """Verify filter drastically attenuates high-frequency static jitter."""
    filter_1e = OneEuroFilter(min_cutoff=1.0, beta=0.005)

    base_signal = 100.0
    np.random.seed(42)
    timestamps = np.linspace(0.0, 1.0, 60)  # 60 FPS for 1 second

    filtered_outputs = []
    noisy_inputs = []

    for t in timestamps:
        noise = np.random.normal(0.0, 3.0)  # +/- 3px jitter
        val = base_signal + noise
        noisy_inputs.append(val)
        filtered = filter_1e(val, timestamp=t)
        filtered_outputs.append(filtered)

    raw_variance = np.var(noisy_inputs[10:])
    filtered_variance = np.var(filtered_outputs[10:])

    # Filtered signal variance should be at least 3x smaller than raw noise
    assert filtered_variance < (raw_variance * 0.35)


def test_one_euro_filter_fast_movement_tracking():
    """Verify filter adapts to fast motion without lag."""
    filter_1e = OneEuroFilter(min_cutoff=1.0, beta=0.05)

    # Simulate fast hand sweep from 0 to 500 in 0.2 seconds
    timestamps = np.linspace(0.0, 0.2, 20)
    positions = np.linspace(0.0, 500.0, 20)

    for t, pos in zip(timestamps, positions):
        filtered = filter_1e(pos, timestamp=t)

    # At the end of the high-speed sweep, filtered value should be very close to the target
    assert pytest.approx(filtered, rel=0.05) == 500.0


def test_one_euro_filter_2d():
    """Verify 2D filter operation and dynamic parameter update."""
    f2d = OneEuroFilter2D(min_cutoff=1.2, beta=0.01)

    pt1 = f2d((100.0, 200.0), timestamp=0.0)
    assert pt1 == (100.0, 200.0)

    pt2 = f2d((105.0, 205.0), timestamp=0.016)
    assert isinstance(pt2, tuple)
    assert len(pt2) == 2

    # Update parameters dynamically
    f2d.set_parameters(min_cutoff=0.8, beta=0.02)
    assert f2d.fx.min_cutoff == 0.8
    assert f2d.fy.beta == 0.02

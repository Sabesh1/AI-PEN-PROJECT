"""Unit tests for long-distance adaptation and parameter scaling."""

import pytest
from core.long_distance import DistanceZone, LongDistanceOptimizer


def test_distance_zone_classification():
    """Verify palm scale maps correctly to physical distance zones."""
    optimizer = LongDistanceOptimizer(reference_palm_size=100.0)

    # Very close: scale > 1.25 (palm size 150px)
    res_close = optimizer.analyze(150.0)
    assert res_close.zone == DistanceZone.CLOSE

    # Optimal: scale between 0.70 and 1.25 (palm size 100px)
    res_opt = optimizer.analyze(100.0)
    assert res_opt.zone == DistanceZone.OPTIMAL

    # Far: scale between 0.45 and 0.70 (palm size 55px)
    res_far = optimizer.analyze(55.0)
    assert res_far.zone == DistanceZone.FAR

    # Extreme: scale < 0.45 (palm size 35px)
    res_ext = optimizer.analyze(35.0)
    assert res_ext.zone == DistanceZone.EXTREME


def test_adaptive_filter_and_hitbox_tuning():
    """Verify adaptive filter cutoff lowers and hitbox padding increases as distance grows."""
    optimizer = LongDistanceOptimizer(reference_palm_size=100.0)

    close_params = optimizer.analyze(140.0)
    far_params = optimizer.analyze(40.0)

    # Far distance should have lower cutoff frequency (heavier smoothing for quantization noise)
    assert far_params.filter_min_cutoff < close_params.filter_min_cutoff

    # Far distance should have larger hitbox padding to make targeting easier
    assert far_params.ui_hitbox_padding > close_params.ui_hitbox_padding

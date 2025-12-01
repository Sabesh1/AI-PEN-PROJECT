"""Test adaptive reach amplifier and smoothing presets."""

import numpy as np
import pytest
from AIR_PEN.backend.hand_tracking import HandTracker, OneEuroFilter
from AIR_PEN.backend.drawing import DrawingEngine, AirStroke


def test_reach_amplifier_distance_scaling():
    tracker = HandTracker()
    tracker.reach_multiplier = 1.4

    # Simulate desktop distance (palm scale = 115)
    tracker.smoothed_palm = 115.0
    # Process simulated raw coordinate offset by +150px from center (640, 360)
    w, h = 1280, 720
    cx = w / 2.0
    REF_PALM = 115.0
    dist_ratio = REF_PALM / tracker.smoothed_palm
    base_gain = 1.0 + 1.25 * max(0.0, (dist_ratio - 1.0) / 1.4)
    near_gain = min(2.85, max(1.0, base_gain * tracker.reach_multiplier))
    assert near_gain == pytest.approx(1.35, abs=0.1)

    # Simulate room distance 2.5m (palm scale = 45)
    tracker.smoothed_palm = 45.0
    dist_ratio_far = REF_PALM / tracker.smoothed_palm
    base_gain_far = 1.0 + 1.25 * max(0.0, (dist_ratio_far - 1.0) / 1.4)
    far_gain = min(2.85, max(1.0, base_gain_far * tracker.reach_multiplier))
    assert far_gain >= 2.2  # Expanded coverage to comfortably reach canvas corners


def test_smoothing_presets():
    tracker = HandTracker()

    tracker.set_smoothing_preset("cinema")
    assert tracker.filter_x.min_cutoff == 0.20
    assert tracker.filter_x.beta == 0.025

    tracker.set_smoothing_preset("fast")
    assert tracker.filter_x.min_cutoff == 0.50
    assert tracker.filter_x.beta == 0.050

    tracker.set_smoothing_preset("balanced")
    assert tracker.filter_x.min_cutoff == 0.32
    assert tracker.filter_x.beta == 0.035


def test_drawing_engine_smooth_bezier():
    engine = DrawingEngine(1280, 720)
    engine.start_stroke()
    # Add points along an arc
    for pt in [(100, 100), (120, 140), (180, 200), (250, 230), (320, 240)]:
        engine.add_point(pt)
    engine.finish_stroke()

    assert len(engine.completed_strokes) == 1
    stroke = engine.completed_strokes[0]
    spline_pts = stroke.get_spline_points(samples_per_seg=6)
    assert len(spline_pts) > len(stroke.points)


def test_any_one_finger_and_bye_wave_gestures():
    from AIR_PEN.backend.gestures import GestureDetector, GestureType

    detector = GestureDetector(hysteresis_frames=1)

    # Test helper creating mock pixel landmarks
    class MockTracker:
        def __init__(self, ext_fingers):
            self.palm_scale = 100.0
            self.raw_landmarks = [(0.5, 0.5, 0.0)] * 21
            self.pixel_landmarks = [(300, 500)] * 21  # Wrist at (300, 500)

            # MCPs at y=380
            self.pixel_landmarks[1] = (260, 460)  # Th CMC
            self.pixel_landmarks[2] = (240, 430)  # Th MCP
            self.pixel_landmarks[3] = (230, 400)  # Th IP
            self.pixel_landmarks[4] = (220, 370)  # Th TIP
            self.pixel_landmarks[5] = (280, 380)  # Ix MCP
            self.pixel_landmarks[9] = (300, 370)  # Mid MCP
            self.pixel_landmarks[13] = (320, 380) # Ring MCP
            self.pixel_landmarks[17] = (340, 400) # Pinky MCP

            finger_map = {
                "index": (5, 6, 8),
                "middle": (9, 10, 12),
                "ring": (13, 14, 16),
                "pinky": (17, 18, 20),
            }
            for name, (mcp_i, pip_i, tip_i) in finger_map.items():
                base_x = self.pixel_landmarks[mcp_i][0]
                if name in ext_fingers:
                    self.pixel_landmarks[pip_i] = (base_x, 300)
                    self.pixel_landmarks[tip_i] = (base_x, 180)  # Extended
                else:
                    self.pixel_landmarks[pip_i] = (base_x, 350)
                    self.pixel_landmarks[tip_i] = (base_x, 400)  # Curled

    # 1. Any one finger (Index) -> DRAW
    tr_index = MockTracker(["index"])
    assert detector.classify(tr_index) == GestureType.DRAW

    # 2. Any one finger (Middle) -> DRAW
    detector.reset()
    tr_middle = MockTracker(["middle"])
    assert detector.classify(tr_middle) == GestureType.DRAW

    # 3. Bye wave / Open palm (all 4 fingers extended) -> ERASE
    detector.reset()
    tr_bye = MockTracker(["index", "middle", "ring", "pinky"])
    assert detector.classify(tr_bye) == GestureType.ERASE

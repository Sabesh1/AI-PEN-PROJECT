"""Unit tests for the drawing engine and stroke management."""

import numpy as np
import pytest
from config.settings import CanvasMode
from engine.drawing_engine import DrawingEngine
from engine.stroke import Stroke


def test_stroke_spline_interpolation():
    """Verify Catmull-Rom spline increases point density smoothly."""
    stroke = Stroke(color_bgr=(255, 0, 0), thickness=5)
    pts = [(10.0, 10.0), (30.0, 50.0), (70.0, 80.0), (120.0, 100.0)]
    for p in pts:
        stroke.add_point(p)

    interpolated = stroke.get_interpolated_points(points_per_segment=4)
    # With 4 raw points and 4 segments, should have at least 10 interpolated points
    assert len(interpolated) > len(pts)
    assert interpolated[0] == (10, 10)
    assert interpolated[-1] == (120, 100)


def test_drawing_engine_stroke_lifecycle():
    """Verify starting, drawing, and committing strokes."""
    engine = DrawingEngine(width=640, height=480)
    assert len(engine.completed_strokes) == 0

    engine.start_stroke()
    engine.add_point_to_active_stroke((100.0, 100.0))
    engine.add_point_to_active_stroke((110.0, 110.0))
    engine.add_point_to_active_stroke((120.0, 120.0))
    engine.finish_stroke()

    assert len(engine.completed_strokes) == 1
    assert len(engine.completed_strokes[0].points) == 3


def test_undo_and_redo_stacks():
    """Verify undo/redo operations and canvas recovery."""
    engine = DrawingEngine(width=640, height=480)

    # Stroke 1
    engine.start_stroke()
    engine.add_point_to_active_stroke((50.0, 50.0))
    engine.add_point_to_active_stroke((60.0, 60.0))
    engine.finish_stroke()

    # Stroke 2
    engine.start_stroke()
    engine.add_point_to_active_stroke((100.0, 100.0))
    engine.add_point_to_active_stroke((110.0, 110.0))
    engine.finish_stroke()

    assert len(engine.completed_strokes) == 2

    # Undo Stroke 2
    assert engine.undo() is True
    assert len(engine.completed_strokes) == 1
    assert len(engine.redo_stack) == 1

    # Redo Stroke 2
    assert engine.redo() is True
    assert len(engine.completed_strokes) == 2
    assert len(engine.redo_stack) == 0


def test_clear_canvas_with_undo():
    """Verify clearing canvas preserves strokes in redo stack."""
    engine = DrawingEngine(width=640, height=480)
    engine.start_stroke()
    engine.add_point_to_active_stroke((20.0, 20.0))
    engine.add_point_to_active_stroke((30.0, 30.0))
    engine.finish_stroke()

    assert len(engine.completed_strokes) == 1
    engine.clear()
    assert len(engine.completed_strokes) == 0

    # Redo should restore cleared strokes
    engine.redo()
    assert len(engine.completed_strokes) == 1


def test_canvas_modes():
    """Verify cycling through canvas background modes."""
    engine = DrawingEngine(width=640, height=480)
    assert engine.canvas_mode == CanvasMode.OVERLAY

    mode1 = engine.toggle_canvas_mode()
    assert mode1 == CanvasMode.BLACKBOARD

    mode2 = engine.toggle_canvas_mode()
    assert mode2 == CanvasMode.WHITEBOARD

    mode3 = engine.toggle_canvas_mode()
    assert mode3 == CanvasMode.OVERLAY

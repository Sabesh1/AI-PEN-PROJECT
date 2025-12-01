"""Unit tests for the AIR_PEN package structure and components."""

from pathlib import Path
import numpy as np
import pytest

from AIR_PEN.backend.camera import ThreadedCamera
from AIR_PEN.backend.drawing import AirStroke, DrawingEngine
from AIR_PEN.backend.gestures import GestureDetector, GestureType
from AIR_PEN.backend.hand_tracking import HandTracker, OneEuroFilter
from AIR_PEN.frontend.gui import AirPenGUI
from AIR_PEN.main import AirPenApp


def test_air_pen_modules_import():
    """Verify all AIR_PEN submodules can be imported cleanly."""
    assert ThreadedCamera is not None
    assert HandTracker is not None
    assert GestureDetector is not None
    assert DrawingEngine is not None
    assert AirPenGUI is not None
    assert AirPenApp is not None


def test_air_pen_drawing_engine(tmp_path: Path):
    """Verify AIR_PEN DrawingEngine strokes and export."""
    engine = DrawingEngine(width=640, height=480, output_dir=str(tmp_path))
    engine.start_stroke()
    engine.add_point((100.0, 100.0))
    engine.add_point((150.0, 150.0))
    engine.finish_stroke()

    assert len(engine.completed_strokes) == 1
    png_path, svg_path = engine.export_all()
    assert png_path.exists()
    assert svg_path.exists()


def test_air_pen_gui_touchless_dwell():
    """Verify touchless dwell-time button activation in AIR_PEN GUI."""
    engine = DrawingEngine(width=640, height=480)
    gui = AirPenGUI(width=640, height=480, dwell_sec=0.1)

    clicked = []
    gui.setup_toolbar(engine, on_save=lambda: clicked.append("save"))

    # Hit test the clear button
    clear_btn = [b for b in gui.buttons if b.id == "clear"][0]
    bx, by, bw, bh = clear_btn.rect
    pointer = (float(bx + bw // 2), float(by + bh // 2))

    # Initial hover
    gui.update_touchless(pointer, is_hover_mode=True)
    assert gui._hover_btn_id == "clear"

    # Simulate elapsed dwell time
    gui._hover_start = gui._hover_start - 0.2
    gui.update_touchless(pointer, is_hover_mode=True)
    # Dwell completed and triggered
    assert gui._hover_btn_id is None

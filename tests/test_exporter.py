"""Unit tests for SVG, PNG, and JSON exporter."""

import json
from pathlib import Path
import pytest
from engine.drawing_engine import DrawingEngine
from engine.exporter import Exporter


def test_svg_export(tmp_path: Path):
    """Verify SVG generation produces valid XML and vector paths."""
    engine = DrawingEngine(width=800, height=600)
    engine.start_stroke()
    engine.add_point_to_active_stroke((50.0, 50.0))
    engine.add_point_to_active_stroke((150.0, 150.0))
    engine.finish_stroke()

    exporter = Exporter(export_dir=str(tmp_path))
    svg_path = exporter.export_svg(engine)

    assert svg_path.exists()
    content = svg_path.read_text(encoding="utf-8")
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in content
    assert '<path d="M' in content
    assert '</svg>' in content


def test_raster_png_export(tmp_path: Path):
    """Verify high-resolution PNG raster image creation."""
    engine = DrawingEngine(width=800, height=600)
    engine.start_stroke()
    engine.add_point_to_active_stroke((100.0, 100.0))
    engine.add_point_to_active_stroke((200.0, 200.0))
    engine.finish_stroke()

    exporter = Exporter(export_dir=str(tmp_path))
    png_path = exporter.export_raster(engine, background="dark", ext="png")

    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_session_json_export(tmp_path: Path):
    """Verify JSON session contains all coordinates and metadata."""
    engine = DrawingEngine(width=800, height=600)
    engine.start_stroke()
    engine.add_point_to_active_stroke((10.0, 20.0))
    engine.add_point_to_active_stroke((30.0, 40.0))
    engine.finish_stroke()

    exporter = Exporter(export_dir=str(tmp_path))
    json_path = exporter.export_session_json(engine)

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["stroke_count"] == 1
    assert len(data["strokes"]) == 1
    assert data["strokes"][0]["points"][0] == [10.0, 20.0]

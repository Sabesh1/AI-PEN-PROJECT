"""Export engine for Air-Pen drawings.

Provides lossless SVG vector generation, high-resolution PNG/JPG raster export,
and structured JSON session persistence.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from config.settings import (
    CANVAS_BLACKBOARD_BGR,
    CANVAS_WHITEBOARD_BGR,
    COLOR_PALETTE,
)
from engine.drawing_engine import DrawingEngine
from engine.stroke import Stroke


class Exporter:
    """Exports canvas drawings to raster, vector, and session formats."""

    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, prefix: str, ext: str) -> Path:
        """Generate unique timestamped filepath."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.export_dir / f"{prefix}_{timestamp}.{ext}"

    def export_raster(
        self,
        engine: DrawingEngine,
        background: str = "dark",
        ext: str = "png"
    ) -> Path:
        """Export drawing to PNG or JPG raster image.

        background: 'dark' (blackboard), 'white' (whiteboard), or 'transparent'
        """
        filepath = self._generate_filename("airpen_artwork", ext)
        h, w = engine.height, engine.width

        if background == "transparent" and ext.lower() == "png":
            # 4-channel BGRA image
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            mask = cv2.cvtColor(engine._raster_canvas, cv2.COLOR_BGR2GRAY) > 0
            bgra[:, :, :3] = engine._raster_canvas
            bgra[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
            cv2.imwrite(str(filepath), bgra)
        else:
            base_bgr = CANVAS_BLACKBOARD_BGR if background == "dark" else CANVAS_WHITEBOARD_BGR
            canvas = np.full((h, w, 3), base_bgr, dtype=np.uint8)
            composite = engine.composite(canvas)
            cv2.imwrite(str(filepath), composite)

        print(f"[Exporter] Saved raster image to {filepath}")
        return filepath

    def export_svg(self, engine: DrawingEngine) -> Path:
        """Export strokes as infinite-resolution Scalable Vector Graphics (SVG)."""
        filepath = self._generate_filename("airpen_vector", "svg")
        w, h = engine.width, engine.height

        svg_lines: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
            f'  <rect width="100%" height="100%" fill="#16161A" />',
            '  <g id="airpen-strokes">'
        ]

        for stroke in engine.completed_strokes:
            if stroke.is_eraser or stroke.is_empty:
                continue

            pts = stroke.get_interpolated_points()
            if len(pts) < 2:
                continue

            # Convert BGR to Hex color
            b, g, r = stroke.color_bgr
            hex_color = f"#{r:02x}{g:02x}{b:02x}"

            # SVG Path construction
            path_d = [f"M {pts[0][0]} {pts[0][1]}"]
            for pt in pts[1:]:
                path_d.append(f"L {pt[0]} {pt[1]}")

            d_attr = " ".join(path_d)
            stroke_width = max(2, stroke.thickness)

            # Glowing filter or standard vector path
            svg_lines.append(
                f'    <path d="{d_attr}" fill="none" stroke="{hex_color}" '
                f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" />'
            )

        svg_lines.append('  </g>')
        svg_lines.append('</svg>')

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines))

        print(f"[Exporter] Saved SVG vector artwork to {filepath}")
        return filepath

    def export_session_json(self, engine: DrawingEngine) -> Path:
        """Export full stroke vector timeline to JSON for replay or continuation."""
        filepath = self._generate_filename("airpen_session", "json")
        session_data: Dict = {
            "version": "1.0",
            "canvas": {"width": engine.width, "height": engine.height},
            "timestamp": datetime.now().isoformat(),
            "stroke_count": len(engine.completed_strokes),
            "strokes": []
        }

        for idx, stroke in enumerate(engine.completed_strokes):
            session_data["strokes"].append({
                "id": idx,
                "color_bgr": stroke.color_bgr,
                "thickness": stroke.thickness,
                "is_eraser": stroke.is_eraser,
                "glow": stroke.glow,
                "points": stroke.points,
                "timestamps": stroke.timestamps
            })

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        print(f"[Exporter] Saved session JSON to {filepath}")
        return filepath

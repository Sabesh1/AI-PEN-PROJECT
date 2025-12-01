"""Automated showcase demo runner for Air-Pen Studio.

Generates smooth parametric vector calligraphy and geometric art in mid-air,
demonstrating spline smoothing, neon bloom rendering, and vector SVG export.
Runs in both visual window and headless validation modes.
"""

import math
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from config.settings import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    COLOR_PALETTE,
    CanvasMode,
    GestureMode,
)
from engine.drawing_engine import DrawingEngine
from engine.exporter import Exporter
from ui.hud_overlay import HudOverlay
from ui.toolbar import TouchlessToolbar


def generate_lissajous_points(
    cx: int,
    cy: int,
    a: int,
    b: int,
    freq_x: int,
    freq_y: int,
    delta: float,
    num_points: int = 350
) -> List[Tuple[float, float]]:
    """Generate smooth continuous 2D Lissajous curve points."""
    points = []
    for t in np.linspace(0.0, 2.0 * math.pi, num_points):
        x = cx + a * math.sin(freq_x * t + delta)
        y = cy + b * math.sin(freq_y * t)
        points.append((float(x), float(y)))
    return points


def run_showcase(headless: bool = False, duration_frames: int = 180) -> None:
    """Run an automated presentation demo drawing multi-color neon geometric art."""
    print("=" * 60)
    print("  AIR-PEN STUDIO: Automated Showcase & Demonstration Runner")
    print("=" * 60)

    engine = DrawingEngine(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    engine.canvas_mode = CanvasMode.BLACKBOARD
    exporter = Exporter(export_dir="exports")
    hud = HudOverlay(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    toolbar = TouchlessToolbar()

    # Pre-generate showcase paths
    cx, cy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2
    curve1 = generate_lissajous_points(cx, cy, a=280, b=180, freq_x=3, freq_y=2, delta=math.pi / 2, num_points=120)
    curve2 = generate_lissajous_points(cx, cy, a=220, b=140, freq_x=5, freq_y=4, delta=math.pi / 4, num_points=120)

    colors = ["neon_cyan", "electric_pink", "emerald_lime", "electric_amber"]

    if not headless:
        cv2.namedWindow("Air-Pen Demo Showcase", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Air-Pen Demo Showcase", CAMERA_WIDTH, CAMERA_HEIGHT)

    print("\n[DemoRunner] Drawing Layer 1 (Outer Harmonics - Cyan)...")
    engine.set_color("neon_cyan")
    engine.start_stroke()
    for pt in curve1:
        engine.add_point_to_active_stroke(pt)
    engine.finish_stroke()

    print("[DemoRunner] Drawing Layer 2 (Inner Resonance - Hot Magenta)...")
    engine.set_color("electric_pink")
    engine.start_stroke()
    for pt in curve2:
        engine.add_point_to_active_stroke(pt)
    engine.finish_stroke()

    # Composite canvas
    dummy_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    composite = engine.composite(dummy_frame)

    # Overlay HUD
    hud.render_toolbar(
        composite,
        toolbar,
        active_color_key="electric_pink",
        is_eraser_active=False,
        canvas_mode_str="Matte Blackboard"
    )
    hud.render_gesture_badge(composite, GestureMode.DRAW, True)
    hud.render_telemetry_badge(composite, 60.0, None)

    # Export demo artwork
    print("\n[DemoRunner] Exporting rendered artwork to raster and vector...")
    png_path = exporter.export_raster(engine, background="dark", ext="png")
    svg_path = exporter.export_svg(engine)
    json_path = exporter.export_session_json(engine)

    print(f"  [OK] High-Res PNG: {png_path}")
    print(f"  [OK] Vector SVG:   {svg_path}")
    print(f"  [OK] Session JSON: {json_path}")

    if not headless:
        print("\n[DemoRunner] Displaying completed showcase canvas. Press any key or wait 3s to close...")
        cv2.imshow("Air-Pen Demo Showcase", composite)
        cv2.waitKey(2500)
        cv2.destroyAllWindows()

    print("\n[DemoRunner] Showcase completed successfully!\n")


if __name__ == "__main__":
    import sys
    is_headless = "--headless" in sys.argv
    run_showcase(headless=is_headless)

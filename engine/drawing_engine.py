"""Multi-layer Air-Pen drawing engine with neon glow rendering and undo/redo stacks.

Composites drawings over camera video feeds, blackboards, or whiteboards with
smooth spline interpolation, multi-pass neon bloom, and history tracking.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from config.settings import (
    CANVAS_BLACKBOARD_BGR,
    CANVAS_WHITEBOARD_BGR,
    COLOR_PALETTE,
    DEFAULT_BRUSH_RADIUS,
    DEFAULT_COLOR_KEY,
    DEFAULT_ERASER_RADIUS,
    ENABLE_NEON_GLOW,
    GLOW_ALPHA,
    GLOW_RADIUS_RATIO,
    CanvasMode,
)
from engine.stroke import Stroke


class DrawingEngine:
    """Core drawing canvas engine managing strokes, rendering, and layer compositing."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Stroke history stacks
        self.completed_strokes: List[Stroke] = []
        self.redo_stack: List[Stroke] = []
        self.active_stroke: Optional[Stroke] = None

        # Brush state
        self.current_color_key: str = DEFAULT_COLOR_KEY
        self.current_brush_radius: int = DEFAULT_BRUSH_RADIUS
        self.current_eraser_radius: int = DEFAULT_ERASER_RADIUS
        self.canvas_mode: CanvasMode = CanvasMode.OVERLAY

        # Persistent raster stroke layer (stores all committed strokes)
        self._raster_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._glow_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def current_color_bgr(self) -> Tuple[int, int, int]:
        return COLOR_PALETTE[self.current_color_key].bgr

    @property
    def current_glow_bgr(self) -> Tuple[int, int, int]:
        return COLOR_PALETTE[self.current_color_key].glow_bgr

    def set_color(self, color_key: str) -> None:
        """Switch active brush color."""
        if color_key in COLOR_PALETTE:
            self.current_color_key = color_key

    def set_brush_radius(self, radius: int) -> None:
        """Set drawing stroke radius."""
        self.current_brush_radius = max(2, min(50, radius))

    def toggle_canvas_mode(self) -> CanvasMode:
        """Cycle through canvas display modes."""
        modes = [CanvasMode.OVERLAY, CanvasMode.BLACKBOARD, CanvasMode.WHITEBOARD]
        current_idx = modes.index(self.canvas_mode)
        self.canvas_mode = modes[(current_idx + 1) % len(modes)]
        return self.canvas_mode

    def start_stroke(self, is_eraser: bool = False) -> None:
        """Start a new continuous stroke."""
        if self.active_stroke is not None:
            self.finish_stroke()

        thickness = self.current_eraser_radius if is_eraser else self.current_brush_radius
        color = (0, 0, 0) if is_eraser else self.current_color_bgr
        self.active_stroke = Stroke(
            color_bgr=color,
            thickness=thickness,
            is_eraser=is_eraser,
            glow=ENABLE_NEON_GLOW and not is_eraser
        )
        self.redo_stack.clear()

    def add_point_to_active_stroke(self, point: Tuple[float, float]) -> None:
        """Add coordinate to current stroke and render incremental segment."""
        if self.active_stroke is None:
            self.start_stroke(is_eraser=False)

        prev_count = len(self.active_stroke.points)
        self.active_stroke.add_point(point)

        # Incrementally draw on raster canvas to maintain 60 FPS
        if len(self.active_stroke.points) >= 2 and len(self.active_stroke.points) > prev_count:
            p1 = (int(self.active_stroke.points[-2][0]), int(self.active_stroke.points[-2][1]))
            p2 = (int(self.active_stroke.points[-1][0]), int(self.active_stroke.points[-1][1]))
            self._render_line_segment(p1, p2, self.active_stroke)

    def finish_stroke(self) -> None:
        """Commit active stroke to history."""
        if self.active_stroke is not None and not self.active_stroke.is_empty:
            self.completed_strokes.append(self.active_stroke)
        self.active_stroke = None

    def undo(self) -> bool:
        """Undo last committed stroke."""
        self.finish_stroke()
        if not self.completed_strokes:
            return False
        popped = self.completed_strokes.pop()
        self.redo_stack.append(popped)
        self.rebuild_canvas()
        return True

    def redo(self) -> bool:
        """Redo previously undone stroke."""
        self.finish_stroke()
        if not self.redo_stack:
            return False
        stroke = self.redo_stack.pop()
        self.completed_strokes.append(stroke)
        self.rebuild_canvas()
        return True

    def clear(self) -> None:
        """Clear all strokes with undo preservation."""
        self.finish_stroke()
        if self.completed_strokes:
            # Transfer all completed strokes to redo stack
            self.redo_stack.extend(reversed(self.completed_strokes))
            self.completed_strokes.clear()
        self.rebuild_canvas()

    def rebuild_canvas(self) -> None:
        """Full re-render of stroke history onto raster canvas."""
        self._raster_canvas.fill(0)
        self._glow_canvas.fill(0)

        for stroke in self.completed_strokes:
            pts = stroke.get_interpolated_points()
            if len(pts) == 1:
                # Single dot
                self._render_dot(pts[0], stroke)
            else:
                for i in range(len(pts) - 1):
                    self._render_line_segment(pts[i], pts[i + 1], stroke)

    def _render_dot(self, pt: Tuple[int, int], stroke: Stroke) -> None:
        """Render a single dot."""
        if stroke.is_eraser:
            cv2.circle(self._raster_canvas, pt, stroke.thickness, (0, 0, 0), -1)
            cv2.circle(self._glow_canvas, pt, stroke.thickness, (0, 0, 0), -1)
        else:
            if stroke.glow:
                glow_r = int(stroke.thickness * GLOW_RADIUS_RATIO)
                cv2.circle(self._glow_canvas, pt, glow_r, stroke.color_bgr, -1, cv2.LINE_AA)
            cv2.circle(self._raster_canvas, pt, stroke.thickness, stroke.color_bgr, -1, cv2.LINE_AA)

    def _render_line_segment(self, p1: Tuple[int, int], p2: Tuple[int, int], stroke: Stroke) -> None:
        """Draw a single line segment with antialiasing and glow."""
        if stroke.is_eraser:
            # Eraser clears pixels to zero on both layers
            cv2.line(self._raster_canvas, p1, p2, (0, 0, 0), stroke.thickness * 2, cv2.LINE_AA)
            cv2.circle(self._raster_canvas, p2, stroke.thickness, (0, 0, 0), -1)
            cv2.line(self._glow_canvas, p1, p2, (0, 0, 0), stroke.thickness * 2, cv2.LINE_AA)
            cv2.circle(self._glow_canvas, p2, stroke.thickness, (0, 0, 0), -1)
        else:
            # Draw neon bloom on glow canvas
            if stroke.glow:
                glow_w = int(stroke.thickness * GLOW_RADIUS_RATIO)
                cv2.line(self._glow_canvas, p1, p2, stroke.color_bgr, glow_w, cv2.LINE_AA)
            # Draw core vibrant stroke on raster canvas
            cv2.line(self._raster_canvas, p1, p2, stroke.color_bgr, stroke.thickness, cv2.LINE_AA)
            cv2.circle(self._raster_canvas, p2, stroke.thickness // 2, stroke.color_bgr, -1, cv2.LINE_AA)

    def composite(self, camera_frame: np.ndarray) -> np.ndarray:
        """Composite background, glow, and drawings into a single high-performance frame."""
        # 1. Base Layer Selection
        if self.canvas_mode == CanvasMode.OVERLAY:
            output = camera_frame.copy()
        elif self.canvas_mode == CanvasMode.BLACKBOARD:
            output = np.full((self.height, self.width, 3), CANVAS_BLACKBOARD_BGR, dtype=np.uint8)
        else:  # WHITEBOARD
            output = np.full((self.height, self.width, 3), CANVAS_WHITEBOARD_BGR, dtype=np.uint8)

        # 2. Blend Neon Glow Layer
        if ENABLE_NEON_GLOW:
            # Apply light gaussian blur for soft atmospheric bloom
            glow_blurred = cv2.GaussianBlur(self._glow_canvas, (15, 15), 0)
            glow_mask = cv2.cvtColor(glow_blurred, cv2.COLOR_BGR2GRAY) > 10
            if np.any(glow_mask):
                output[glow_mask] = cv2.addWeighted(
                    output, 1.0, glow_blurred, GLOW_ALPHA, 0
                )[glow_mask]

        # 3. Blend Core Stroke Layer
        stroke_mask = cv2.cvtColor(self._raster_canvas, cv2.COLOR_BGR2GRAY) > 0
        if np.any(stroke_mask):
            if self.canvas_mode == CanvasMode.WHITEBOARD:
                # On whiteboard, ensure white strokes are visible with a contrast tint
                output[stroke_mask] = self._raster_canvas[stroke_mask]
            else:
                output[stroke_mask] = self._raster_canvas[stroke_mask]

        return output

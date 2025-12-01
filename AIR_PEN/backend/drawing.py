"""Drawing engine — smooth Bezier strokes, jitter filtering, neon glow, vector export.

Key improvements for clean letter shapes:
- Spatial deadzone: ignores sub-4px micro-jitter before recording a point
- Per-stroke EMA pre-smoother: eliminates finger tremor before curve fitting
- Catmull-Rom spline: organic smooth curve through all points
- Round-cap line joins: seamless ribbon with no gaps

Gradient brush:
- 'rainbow' mode sweeps through HSV hue along the stroke length.
- 'custom' mode linearly blends from the start hex colour to an end hex colour.
"""

from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pathlib import Path
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from backend.recognition import AirCharacterRecognizer, RecognitionResult
except ImportError:
    from .recognition import AirCharacterRecognizer, RecognitionResult


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB hex string to OpenCV BGR tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_bgr(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linear interpolate between two BGR colours."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _rainbow_bgr(t: float) -> Tuple[int, int, int]:
    """Return BGR for a full HSV rainbow at hue=t*179 (OpenCV scale)."""
    hsv = np.uint8([[[int(t * 179), 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ---------------------------------------------------------------------------
# Stroke class
# ---------------------------------------------------------------------------
@dataclass
class AirStroke:
    """Represents a continuous stroke of points."""
    color_bgr: Tuple[int, int, int]
    thickness: int
    is_eraser: bool = False
    glow: bool = True
    points: List[Tuple[float, float]] = field(default_factory=list)

    # Gradient support
    gradient_mode: str = "none"           # "none" | "rainbow" | "custom"
    gradient_end_bgr: Optional[Tuple[int, int, int]] = None  # for "custom"

    # Per-stroke EMA smoother state (removes finger tremor)
    _ema_x: float = field(default=0.0, init=False, repr=False)
    _ema_y: float = field(default=0.0, init=False, repr=False)
    _ema_init: bool = field(default=False, init=False, repr=False)

    # Spatial deadzone: 1.8px preserves fine letter loops and cursive curves
    _MIN_DIST_PX: float = field(default=1.8, init=False, repr=False)

    def add_point(self, pt: Tuple[float, float]) -> None:
        """Add a point with high-fidelity trajectory and responsive curve capture."""
        x, y = float(pt[0]), float(pt[1])

        # ── High-fidelity EMA: alpha=0.88 captures tight loops of 'a' and 'b' without lag ──
        EMA = 0.88 if not self.is_eraser else 0.95
        if not self._ema_init:
            self._ema_x, self._ema_y = x, y
            self._ema_init = True
        else:
            self._ema_x = EMA * x + (1.0 - EMA) * self._ema_x
            self._ema_y = EMA * y + (1.0 - EMA) * self._ema_y

        sx, sy = self._ema_x, self._ema_y

        # ── Spatial deadzone: 1.8px ensures sharp letter bends aren't dropped ──
        min_d = 1.8 if not self.is_eraser else 2.5
        if self.points:
            last = self.points[-1]
            d2 = (sx - last[0]) ** 2 + (sy - last[1]) ** 2
            if d2 < min_d ** 2:
                return

        self.points.append((sx, sy))

    def get_color_at(self, t: float) -> Tuple[int, int, int]:
        """Return the stroke colour at normalised position t in [0,1]."""
        if self.gradient_mode == "rainbow":
            return _rainbow_bgr(t)
        elif self.gradient_mode == "custom" and self.gradient_end_bgr is not None:
            return _lerp_bgr(self.color_bgr, self.gradient_end_bgr, t)
        return self.color_bgr

    def get_spline_points(self, samples_per_seg: int = 8) -> List[Tuple[int, int]]:
        """Catmull-Rom spline — organic smooth curve through all control points."""
        n = len(self.points)
        if n == 0:
            return []
        if n == 1:
            return [(int(round(self.points[0][0])), int(round(self.points[0][1])))]
        if n == 2:
            return [
                (int(round(self.points[0][0])), int(round(self.points[0][1]))),
                (int(round(self.points[1][0])), int(round(self.points[1][1]))),
            ]

        pts = np.array(self.points, dtype=np.float64)
        # Reflect endpoints so the curve reaches all the way to start and end
        padded = np.vstack([pts[0] * 2 - pts[1], pts, pts[-1] * 2 - pts[-2]])

        out: List[Tuple[int, int]] = []
        for i in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[i-1], padded[i], padded[i+1], padded[i+2]
            for j in range(samples_per_seg):
                t = j / samples_per_seg
                t2, t3 = t * t, t * t * t
                q = 0.5 * (
                    2.0 * p1
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                out.append((int(round(q[0])), int(round(q[1]))))

        # Append last point
        out.append((int(round(pts[-1][0])), int(round(pts[-1][1]))))
        return out


# ---------------------------------------------------------------------------
# Drawing engine
# ---------------------------------------------------------------------------
class DrawingEngine:
    """Multi-layer vector canvas with neon glow rendering pipeline."""

    PALETTE: Dict[str, Tuple[Tuple[int, int, int], str]] = {
        "cyan":    ((255, 240, 0),   "#00F0FF"),   # Neon Cyan
        "magenta": ((220, 20, 255),  "#FF14DC"),   # Hot Magenta
        "lime":    ((50, 255, 50),   "#32FF32"),   # Emerald Lime
        "amber":   ((0, 165, 255),   "#FFA500"),   # Sunburst Amber
        "white":   ((255, 255, 255), "#FFFFFF"),   # Bright White
    }

    def __init__(self, width: int = 1280, height: int = 720, output_dir: str = "output"):
        self.width = width
        self.height = height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.completed_strokes: List[AirStroke] = []
        self.redo_stack: List[AirStroke] = []
        self.active_stroke: Optional[AirStroke] = None

        self.current_color_key: str = "cyan"
        self.current_custom_color_bgr: Optional[Tuple[int, int, int]] = None
        self.current_custom_hex: str = ""

        # Gradient state
        self.gradient_mode: str = "none"          # "none" | "rainbow" | "custom"
        self.gradient_end_bgr: Optional[Tuple[int, int, int]] = None
        self.gradient_end_hex: str = ""

        self.brush_radius: int = 6
        self.eraser_radius: int = 45
        self.canvas_mode: str = "overlay"

        # AI Character & Word Recognition Engine
        self.recognizer = AirCharacterRecognizer()
        self.guidelines_enabled: bool = False
        self.recent_recognized_char: str = ""
        self.recent_recognized_conf: float = 0.0
        self.recent_char_time: float = 0.0

        self._raster_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._glow_canvas   = np.zeros((height, width, 3), dtype=np.uint8)
        self._lock = threading.RLock()

    @property
    def current_color_bgr(self) -> Tuple[int, int, int]:
        if self.current_custom_color_bgr is not None:
            return self.current_custom_color_bgr
        return self.PALETTE[self.current_color_key][0]

    def set_color(self, key: str) -> None:
        """Set colour from the predefined palette."""
        if key in self.PALETTE:
            self.current_color_key = key
            self.current_custom_color_bgr = None
            self.current_custom_hex = ""

    def set_custom_color(self, hex_color: str) -> None:
        """Set a fully custom colour from a hex string (e.g. '#FF6600')."""
        try:
            bgr = _hex_to_bgr(hex_color)
            self.current_custom_color_bgr = bgr
            self.current_custom_hex = hex_color
            self.current_color_key = "custom"
        except Exception:
            pass

    def set_gradient_mode(self, mode: str, end_hex: str = "") -> None:
        """Configure gradient brush mode.

        mode: 'none', 'rainbow', or 'custom'
        end_hex: only used when mode == 'custom' (hex string for end colour)
        """
        valid = ("none", "rainbow", "custom")
        if mode not in valid:
            mode = "none"
        self.gradient_mode = mode
        if mode == "custom" and end_hex:
            try:
                self.gradient_end_bgr = _hex_to_bgr(end_hex)
                self.gradient_end_hex = end_hex
            except Exception:
                self.gradient_mode = "none"
        elif mode != "custom":
            self.gradient_end_bgr = None
            self.gradient_end_hex = ""

    def toggle_canvas_mode(self) -> str:
        modes = ["overlay", "blackboard", "whiteboard"]
        self.canvas_mode = modes[(modes.index(self.canvas_mode) + 1) % len(modes)]
        return self.canvas_mode

    # ------------------------------------------------------------------
    # Stroke lifecycle
    # ------------------------------------------------------------------
    def start_stroke(self, is_eraser: bool = False) -> None:
        with self._lock:
            if self.active_stroke is not None:
                self._commit_active()
            thick = self.eraser_radius if is_eraser else self.brush_radius
            color = (0, 0, 0) if is_eraser else self.current_color_bgr
            self.active_stroke = AirStroke(
                color_bgr=color,
                thickness=thick,
                is_eraser=is_eraser,
                glow=(not is_eraser),
                gradient_mode="none" if is_eraser else self.gradient_mode,
                gradient_end_bgr=None if is_eraser else self.gradient_end_bgr,
            )
            self.redo_stack.clear()

    def add_point(self, pt: Tuple[float, float]) -> None:
        with self._lock:
            if self.active_stroke is None:
                self.start_stroke(is_eraser=False)

            stroke = self.active_stroke
            if stroke is None:
                return

            prev_len = len(stroke.points)
            stroke.add_point(pt)
            curr_len = len(stroke.points)

            if curr_len == prev_len:
                # No new point added (deadzone / duplicate)
                return

            pts = stroke.points
            # Compute normalised position along stroke for gradient colouring
            total_est = max(1, curr_len - 1)
            t_now = (curr_len - 1) / total_est

            if curr_len == 1:
                p0 = (int(round(pts[0][0])), int(round(pts[0][1])))
                self._render_dot(p0, stroke, t=0.0)

            elif curr_len == 2:
                p0 = (int(round(pts[0][0])), int(round(pts[0][1])))
                p1 = (int(round(pts[1][0])), int(round(pts[1][1])))
                self._render_segment(p0, p1, stroke, t=0.5)

            else:
                # Smooth quadratic bezier through rolling 3-point window
                # Uses midpoint method for C1-continuous joins
                self._render_bezier_tail(pts, stroke, t=t_now)

    def finish_stroke(self) -> None:
        with self._lock:
            stroke = self.active_stroke
            self._commit_active()
            if stroke is not None and not stroke.is_eraser and len(stroke.points) >= 4:
                try:
                    now_ts = datetime.now().timestamp()
                    if hasattr(self.recognizer, "process_stroke"):
                        res = self.recognizer.process_stroke(stroke.points, now_ts=now_ts)
                    else:
                        res = self.recognizer.recognize([stroke.points])
                    if res and res.character not in ["?", "·", "~"]:
                        self.recent_recognized_char = res.character
                        self.recent_recognized_conf = res.confidence
                        self.recent_char_time = now_ts
                except Exception as e:
                    pass

    def _commit_active(self) -> None:
        """Commit active stroke to completed list (must be called under lock)."""
        stroke = self.active_stroke
        if stroke is not None and stroke.points:
            # Draw the final tail segment to the last point
            pts = stroke.points
            if len(pts) >= 2:
                p_last = (int(round(pts[-1][0])), int(round(pts[-1][1])))
                p_prev = (int(round(pts[-2][0])), int(round(pts[-2][1])))
                self._render_segment(p_prev, p_last, stroke, t=1.0)
            self.completed_strokes.append(stroke)
        self.active_stroke = None

    def _render_bezier_tail(self, pts: List[Tuple[float, float]], stroke: "AirStroke", t: float = 0.5) -> None:
        """Draw smooth bezier through last 3 points using midpoint method."""
        p0 = pts[-3]
        p1 = pts[-2]
        p2 = pts[-1]

        # Midpoints for C1-continuous join
        m1x, m1y = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
        m2x, m2y = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0

        dist = math.hypot(m2x - m1x, m2y - m1y)
        steps = max(6, min(24, int(dist / 2.5)))   # more steps = smoother curves

        last_pt = (int(round(m1x)), int(round(m1y)))
        for s in range(1, steps + 1):
            seg_t = s / float(steps)
            omt = 1.0 - seg_t
            bx = omt * omt * m1x + 2.0 * omt * seg_t * p1[0] + seg_t * seg_t * m2x
            by = omt * omt * m1y + 2.0 * omt * seg_t * p1[1] + seg_t * seg_t * m2y
            curr_pt = (int(round(bx)), int(round(by)))
            # blend between t (progress in stroke) and local bezier seg_t for gradient
            self._render_segment(last_pt, curr_pt, stroke, t=t * seg_t + (1.0 - seg_t) * max(0.0, t - 1.0 / max(1, len(pts))))
            last_pt = curr_pt

    # ------------------------------------------------------------------
    # Undo / redo / clear
    # ------------------------------------------------------------------
    def undo(self) -> bool:
        with self._lock:
            self._commit_active()
        if not self.completed_strokes:
            return False
        self.redo_stack.append(self.completed_strokes.pop())
        self.rebuild_canvas()
        return True

    def redo(self) -> bool:
        self.finish_stroke()
        if not self.redo_stack:
            return False
        self.completed_strokes.append(self.redo_stack.pop())
        self.rebuild_canvas()
        return True

    def clear(self) -> None:
        self.finish_stroke()
        if self.completed_strokes:
            self.redo_stack.extend(reversed(self.completed_strokes))
            self.completed_strokes.clear()
        self.rebuild_canvas()

    def write_text(self, text: str, x: int = 100, y: int = 250,
                   color_key: Optional[str] = None, font_scale: float = 1.6) -> None:
        """Render glowing text onto the canvas."""
        self.finish_stroke()
        color_bgr = self.PALETTE[color_key][0] if (color_key and color_key in self.PALETTE) else self.current_color_bgr
        cv2.putText(self._glow_canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 6, cv2.LINE_AA)
        cv2.putText(self._raster_canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 2, cv2.LINE_AA)

    def rebuild_canvas(self) -> None:
        """Redraw all completed strokes from scratch (used after undo/clear)."""
        self._raster_canvas.fill(0)
        self._glow_canvas.fill(0)
        for stroke in self.completed_strokes:
            pts = stroke.get_spline_points()
            total = max(1, len(pts) - 1)
            if len(pts) == 1:
                self._render_dot(pts[0], stroke, t=0.0)
            else:
                for i in range(len(pts) - 1):
                    seg_t = i / total
                    self._render_segment(pts[i], pts[i + 1], stroke, t=seg_t)

    # ------------------------------------------------------------------
    # Render primitives
    # ------------------------------------------------------------------
    def _render_dot(self, p: Tuple[int, int], stroke: AirStroke, t: float = 0.0) -> None:
        r = max(1, stroke.thickness // 2)
        color = stroke.get_color_at(t)
        if stroke.is_eraser:
            cv2.circle(self._raster_canvas, p, stroke.thickness, (0, 0, 0), -1)
            cv2.circle(self._glow_canvas,   p, stroke.thickness, (0, 0, 0), -1)
        else:
            if stroke.glow:
                cv2.circle(self._glow_canvas, p, int(stroke.thickness * 2.5), color, -1, cv2.LINE_AA)
            cv2.circle(self._raster_canvas, p, r, color, -1, cv2.LINE_AA)

    def _render_segment(self, p1: Tuple[int, int], p2: Tuple[int, int], stroke: AirStroke, t: float = 0.5) -> None:
        r = max(1, stroke.thickness // 2)
        color = stroke.get_color_at(t)
        if stroke.is_eraser:
            cv2.line(self._raster_canvas, p1, p2, (0, 0, 0), stroke.thickness * 2, cv2.LINE_AA)
            cv2.circle(self._raster_canvas, p2, stroke.thickness, (0, 0, 0), -1)
            cv2.line(self._glow_canvas,   p1, p2, (0, 0, 0), stroke.thickness * 2, cv2.LINE_AA)
            cv2.circle(self._glow_canvas,   p2, stroke.thickness, (0, 0, 0), -1)
        else:
            if stroke.glow:
                cv2.line(self._glow_canvas, p1, p2, color, int(stroke.thickness * 2.5), cv2.LINE_AA)
                cv2.circle(self._glow_canvas, p2, int(stroke.thickness * 1.2), color, -1, cv2.LINE_AA)
            # Core crisp line
            cv2.line(self._raster_canvas, p1, p2, color, stroke.thickness, cv2.LINE_AA)
            # Round cap for seamless join
            cv2.circle(self._raster_canvas, p2, r, color, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------
    def composite(self, base_frame: np.ndarray) -> np.ndarray:
        """Blend background + neon glow bloom + crisp core lines."""
        if self.canvas_mode == "blackboard":
            canvas = np.full((self.height, self.width, 3), (22, 22, 26), dtype=np.uint8)
        elif self.canvas_mode == "whiteboard":
            canvas = np.full((self.height, self.width, 3), (248, 249, 250), dtype=np.uint8)
        else:
            canvas = base_frame.copy()

        if not self.completed_strokes and (self.active_stroke is None or not self.active_stroke.points):
            return canvas

        # Fast neon bloom: downsample → blur → upsample
        small  = cv2.resize(self._glow_canvas, (self.width // 4, self.height // 4), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small, (7, 7), 0)
        glow   = cv2.resize(blurred, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.add(canvas, cv2.scaleAdd(glow, 0.42, np.zeros_like(glow)))

        # Composite core strokes
        gray = cv2.cvtColor(self._raster_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        canvas_bg  = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
        stroke_fg  = cv2.bitwise_and(self._raster_canvas, self._raster_canvas, mask=mask)
        composite_out = cv2.add(canvas_bg, stroke_fg)

        # ── Optional 4-line notebook guidelines overlay for clean writing ──
        if self.guidelines_enabled:
            composite_out = self._draw_notebook_guidelines(composite_out)

        # ── AI Recognition HUD Badge ──
        import time as _tm
        if self.recent_recognized_char and (_tm.time() - self.recent_char_time < 3.5):
            composite_out = self._draw_recognition_badge(composite_out)

        return composite_out

    def _draw_notebook_guidelines(self, frame: np.ndarray) -> np.ndarray:
        """Renders subtle 4-line notebook guide for practicing letters like 'a' and 'b'."""
        out = frame.copy()
        h, w = frame.shape[:2]
        
        # Center horizontal band (y = 220 to 460)
        y_top = int(h * 0.35)       # 252 (ascender / capital line)
        y_mid = int(h * 0.48)       # 345 (midline / x-height)
        y_base = int(h * 0.61)      # 439 (baseline)
        y_desc = int(h * 0.74)      # 532 (descender line)

        # Draw dashed guidelines
        def draw_dashed_line(y, color, dash_len=14, gap_len=10, thickness=1):
            for x in range(40, w - 40, dash_len + gap_len):
                cv2.line(out, (x, y), (min(w - 40, x + dash_len), y), color, thickness, cv2.LINE_AA)

        draw_dashed_line(y_top, (100, 100, 150), thickness=1)
        draw_dashed_line(y_mid, (0, 200, 255), thickness=1)       # Midline yellow/orange
        cv2.line(out, (40, y_base), (w - 40, y_base), (0, 240, 160), 2, cv2.LINE_AA) # Baseline cyan
        draw_dashed_line(y_desc, (120, 80, 80), thickness=1)

        # Helper tags
        cv2.putText(out, "TOPLINE (Ascenders: b, d, h, A, B)", (45, y_top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 200), 1, cv2.LINE_AA)
        cv2.putText(out, "MIDLINE (x-height: a, c, e, o)", (45, y_mid - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "BASELINE (Writing Base)", (45, y_base - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 160), 1, cv2.LINE_AA)
        cv2.putText(out, "DESCENDER (Tails: g, j, p, q, y)", (45, y_desc - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 120, 120), 1, cv2.LINE_AA)
        return out

    def _draw_recognition_badge(self, frame: np.ndarray) -> np.ndarray:
        """Renders floating translucent badge with recognized character and word."""
        out = frame.copy()
        h, w = frame.shape[:2]
        char = self.recent_recognized_char
        conf = int(self.recent_recognized_conf * 100)
        word = self.recognizer.get_word()

        badge_w, badge_h = 320, 48
        bx = (w - badge_w) // 2
        by = h - 70

        # Semi-transparent dark pill background
        sub_img = out[by:by+badge_h, bx:bx+badge_w]
        pill = np.full(sub_img.shape, (16, 17, 24), dtype=np.uint8)
        cv2.addWeighted(pill, 0.82, sub_img, 0.18, 0, sub_img)
        cv2.rectangle(out, (bx, by), (bx+badge_w, by+badge_h), (0, 200, 255), 1, cv2.LINE_AA)

        # Content
        cv2.putText(out, f"AI CHAR:", (bx + 12, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 190), 1, cv2.LINE_AA)
        cv2.putText(out, f"[ {char} ]", (bx + 90, by + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"WORD: {word}", (bx + 165, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 160), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_all(self) -> Tuple[Path, Path]:
        """Save PNG and SVG to output/.  Gradient strokes export as linearGradient SVG defs."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = self.output_dir / f"airpen_{ts}.png"
        svg_path = self.output_dir / f"airpen_{ts}.svg"

        base = np.full((self.height, self.width, 3), (22, 22, 26), dtype=np.uint8)
        comp = self.composite(base)
        cv2.imwrite(str(png_path), comp)

        # ── Build SVG with linearGradient support for gradient strokes ──
        defs_parts: List[str] = []
        stroke_grad_ids: Dict[int, str] = {}

        for idx, stroke in enumerate(self.completed_strokes):
            if stroke.is_eraser or len(stroke.points) < 2 or stroke.gradient_mode == "none":
                continue
            gid = f"grad_{idx}"
            stroke_grad_ids[idx] = gid
            pts = stroke.points
            x1, y1 = pts[0]
            x2, y2 = pts[-1]

            if stroke.gradient_mode == "rainbow":
                stops = [
                    (0.0,  "#ff0000"),
                    (0.17, "#ff8800"),
                    (0.33, "#ffff00"),
                    (0.5,  "#00ff00"),
                    (0.67, "#0088ff"),
                    (0.83, "#8800ff"),
                    (1.0,  "#ff0088"),
                ]
                stops_svg = "\n".join(
                    f'      <stop offset="{o}" stop-color="{c}"/>'
                    for o, c in stops
                )
            else:
                c1 = _bgr_to_hex(stroke.color_bgr)
                c2 = _bgr_to_hex(stroke.gradient_end_bgr) if stroke.gradient_end_bgr else c1
                stops_svg = (
                    f'      <stop offset="0" stop-color="{c1}"/>\n'
                    f'      <stop offset="1" stop-color="{c2}"/>'
                )

            defs_parts.append(
                f'    <linearGradient id="{gid}" x1="{x1:.1f}" y1="{y1:.1f}" '
                f'x2="{x2:.1f}" y2="{y2:.1f}" gradientUnits="userSpaceOnUse">\n'
                f'{stops_svg}\n    </linearGradient>'
            )

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}">',
            '  <rect width="100%" height="100%" fill="#16161A" />',
        ]
        if defs_parts:
            svg_lines.append("  <defs>")
            svg_lines.extend(defs_parts)
            svg_lines.append("  </defs>")

        for idx, stroke in enumerate(self.completed_strokes):
            if stroke.is_eraser or len(stroke.points) < 2:
                continue
            pts = stroke.get_spline_points()
            d_parts = " ".join(
                f"{'M' if i == 0 else 'L'} {p[0]} {p[1]}" for i, p in enumerate(pts)
            )
            if idx in stroke_grad_ids:
                stroke_attr = f'stroke="url(#{stroke_grad_ids[idx]})"'
            else:
                b, g, r = stroke.color_bgr
                hex_c = f"#{r:02x}{g:02x}{b:02x}"
                stroke_attr = f'stroke="{hex_c}"'

            svg_lines.append(
                f'  <path d="{d_parts}" fill="none" {stroke_attr} '
                f'stroke-width="{stroke.thickness}" stroke-linecap="round" stroke-linejoin="round" />'
            )

        svg_lines.append("</svg>")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines))

        print(f"[DrawingEngine] Saved: {png_path} and {svg_path}")
        return png_path, svg_path

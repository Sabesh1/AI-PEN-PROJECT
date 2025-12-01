"""Glassmorphic Heads-Up Display (HUD) overlay renderer for Air-Pen.

Draws translucent glass-styled toolbars, interactive dwell progress arcs,
active gesture badges, telemetry diagnostics, and dynamic pointer reticles.
"""

import math
from typing import Optional, Tuple
import cv2
import numpy as np
from config.settings import (
    COLOR_PALETTE,
    HUD_ACCENT_COLOR,
    HUD_BG_COLOR,
    HUD_BORDER_COLOR,
    HUD_TEXT_COLOR,
    GestureMode,
)
from core.long_distance import AdaptiveParameters
from engine.drawing_engine import DrawingEngine
from ui.toolbar import TouchlessToolbar


class HudOverlay:
    """Renders premium on-screen HUD elements directly over video frames."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

    @staticmethod
    def draw_glass_card(
        image: np.ndarray,
        rect: Tuple[int, int, int, int],
        bg_color: Tuple[int, int, int] = HUD_BG_COLOR,
        border_color: Tuple[int, int, int] = HUD_BORDER_COLOR,
        alpha: float = 0.82,
        radius: int = 10
    ) -> None:
        """Draw a rounded glassmorphic translucent panel."""
        x, y, w, h = rect
        sub_img = image[y:y+h, x:x+w]
        if sub_img.shape[0] <= 0 or sub_img.shape[1] <= 0:
            return

        overlay = sub_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), bg_color, -1)
        cv2.addWeighted(overlay, alpha, sub_img, 1.0 - alpha, 0, sub_img)

        # Crisp outer border
        cv2.rectangle(image, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)

    def render_toolbar(
        self,
        frame: np.ndarray,
        toolbar: TouchlessToolbar,
        active_color_key: str,
        is_eraser_active: bool,
        canvas_mode_str: str
    ) -> None:
        """Render floating top toolbar with buttons, swatches, and dwell arcs."""
        if not toolbar.buttons:
            return

        # Overall toolbar container geometry
        min_x = min(btn.rect[0] for btn in toolbar.buttons) - 12
        max_x = max(btn.rect[0] + btn.rect[2] for btn in toolbar.buttons) + 12
        min_y = min(btn.rect[1] for btn in toolbar.buttons) - 8
        max_y = max(btn.rect[1] + btn.rect[3] for btn in toolbar.buttons) + 8

        # Draw main floating glass panel
        self.draw_glass_card(
            frame,
            (min_x, min_y, max_x - min_x, max_y - min_y),
            bg_color=(20, 20, 25),
            border_color=(60, 60, 75),
            alpha=0.85
        )

        for btn in toolbar.buttons:
            bx, by, bw, bh = btn.rect

            # Determine button active state
            is_active = (btn.icon_type == "color" and btn.id == active_color_key and not is_eraser_active) or \
                        (btn.id == "eraser" and is_eraser_active)

            # Button background
            btn_bg = (55, 55, 70) if is_active else (35, 35, 45)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), btn_bg, -1)
            border_col = HUD_ACCENT_COLOR if is_active else HUD_BORDER_COLOR
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_col, 1, cv2.LINE_AA)

            # Render button contents
            if btn.icon_type == "color" and btn.color_bgr is not None:
                # Color swatch circle
                center = (bx + bw // 2, by + bh // 2)
                cv2.circle(frame, center, 14, btn.color_bgr, -1, cv2.LINE_AA)
                if is_active:
                    cv2.circle(frame, center, 17, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Text button label
                text = btn.label
                font_scale = 0.42
                thickness = 1
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tx = bx + (bw - tw) // 2
                ty = by + (bh + th) // 2
                text_col = (255, 255, 255) if is_active else HUD_TEXT_COLOR
                cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, thickness, cv2.LINE_AA)

            # Draw dwell progress fill ring around button if hovered
            if btn.dwell_progress > 0.0:
                progress_h = int(bh * btn.dwell_progress)
                cv2.rectangle(
                    frame,
                    (bx, by + bh - progress_h),
                    (bx + bw, by + bh),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

    def render_gesture_badge(
        self,
        frame: np.ndarray,
        mode: GestureMode,
        is_drawing: bool
    ) -> None:
        """Render active gesture status pill at bottom-left."""
        # Color-coded badges for modes
        badge_configs = {
            GestureMode.DRAW: ("AIR-DRAW (Active)", (0, 255, 200), (0, 60, 40)),
            GestureMode.HOVER: ("HOVER / SELECT", (255, 200, 0), (50, 40, 0)),
            GestureMode.ERASE: ("ERASER ACTIVE", (0, 100, 255), (0, 20, 60)),
            GestureMode.PINCH_SIZE: ("BRUSH ADJUST", (255, 50, 200), (50, 10, 40)),
            GestureMode.IDLE: ("IDLE (Open Palm)", (180, 180, 190), (35, 35, 40)),
            GestureMode.UNKNOWN: ("SEARCHING...", (120, 120, 120), (25, 25, 30)),
        }

        label, border_col, bg_col = badge_configs.get(
            mode, ("TRACKING", (200, 200, 200), (30, 30, 30))
        )

        rx, ry, rw, rh = 24, self.height - 60, 220, 38
        self.draw_glass_card(frame, (rx, ry, rw, rh), bg_color=bg_col, border_color=border_col, alpha=0.88)

        # Glowing indicator dot
        cv2.circle(frame, (rx + 20, ry + 19), 6, border_col, -1, cv2.LINE_AA)
        # Text label
        cv2.putText(
            frame,
            label,
            (rx + 36, ry + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def render_telemetry_badge(
        self,
        frame: np.ndarray,
        fps: float,
        adaptive_params: Optional[AdaptiveParameters]
    ) -> None:
        """Render performance metrics and distance telemetry at bottom-right."""
        rx, ry, rw, rh = self.width - 240, self.height - 60, 216, 38
        zone_str = adaptive_params.zone.value if adaptive_params else "Detecting..."

        self.draw_glass_card(
            frame,
            (rx, ry, rw, rh),
            bg_color=(20, 20, 25),
            border_color=(60, 60, 75),
            alpha=0.85
        )

        info_text = f"FPS: {fps:4.1f} | {zone_str.split(' ')[0]}"
        cv2.putText(
            frame,
            info_text,
            (rx + 16, ry + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 255, 200) if fps >= 30 else (255, 180, 0),
            1,
            cv2.LINE_AA
        )

    def render_pointer(
        self,
        frame: np.ndarray,
        pointer: Tuple[float, float],
        mode: GestureMode,
        brush_radius: int,
        color_bgr: Tuple[int, int, int],
        dwell_progress: float = 0.0
    ) -> None:
        """Render dynamic reticle / pointer with dwell animation."""
        px, py = int(pointer[0]), int(pointer[1])

        if mode == GestureMode.DRAW:
            # Solid brush reticle with center dot
            cv2.circle(frame, (px, py), brush_radius, color_bgr, 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

        elif mode == GestureMode.HOVER:
            # Hover cursor with circular dwell arc
            radius = 16
            cv2.circle(frame, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1, cv2.LINE_AA)

            if dwell_progress > 0.0:
                # Draw animated circular progress arc
                angle = int(360.0 * dwell_progress)
                cv2.ellipse(
                    frame,
                    (px, py),
                    (radius + 4, radius + 4),
                    -90,
                    0,
                    angle,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        elif mode == GestureMode.ERASE:
            # Dashed or red circle representing eraser footprint
            cv2.circle(frame, (px, py), brush_radius, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px - 6, py), (px + 6, py), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px, py - 6), (px, py + 6), (0, 0, 255), 2, cv2.LINE_AA)

        elif mode == GestureMode.PINCH_SIZE:
            # Dynamic size ring showing real-time adjustment
            cv2.circle(frame, (px, py), brush_radius, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"R: {brush_radius}px",
                (px + brush_radius + 8, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
                cv2.LINE_AA
            )

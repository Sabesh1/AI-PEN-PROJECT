"""Frontend Graphical User Interface (GUI) and Glassmorphism Heads-Up Display."""

from dataclasses import dataclass
import time
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np
try:
    from backend.drawing import DrawingEngine
    from backend.gestures import GestureType
except ImportError:
    from ..backend.drawing import DrawingEngine
    from ..backend.gestures import GestureType


@dataclass
class GUIButton:
    """Touchless UI Button with dwell activation."""
    id: str
    label: str
    rect: Tuple[int, int, int, int]  # (x, y, w, h)
    action: Callable[[], None]
    is_color: bool = False
    color_bgr: Optional[Tuple[int, int, int]] = None
    dwell: float = 0.0


class AirPenGUI:
    """Modern Glassmorphic Heads-Up Display and Touchless Interface."""

    def __init__(self, width: int = 1280, height: int = 720, dwell_sec: float = 0.65):
        self.width = width
        self.height = height
        self.dwell_sec = dwell_sec

        self.buttons: List[GUIButton] = []
        self._hover_btn_id: Optional[str] = None
        self._hover_start: Optional[float] = None
        self._cooldown: float = 0.0

        self.toast_msg: Optional[str] = None
        self.toast_expiry: float = 0.0

    def show_toast(self, text: str, duration: float = 2.0) -> None:
        """Display temporary floating notification toast."""
        self.toast_msg = text
        self.toast_expiry = time.perf_counter() + duration

    def setup_toolbar(self, engine: DrawingEngine, on_save: Callable[[], None]) -> None:
        """Build top interactive toolbar with colors and actions."""
        self.buttons.clear()
        bx, by, bw, bh = 40, 18, 68, 44
        gap = 10

        # 1. Color Palette swatches
        for key, (bgr, _) in engine.PALETTE.items():
            self.buttons.append(
                GUIButton(
                    id=f"col_{key}",
                    label=key.upper()[:4],
                    rect=(bx, by, bw, bh),
                    action=lambda k=key: self._on_color(engine, k),
                    is_color=True,
                    color_bgr=bgr
                )
            )
            bx += bw + gap

        bx += 16

        # 2. Canvas Mode Toggle
        self.buttons.append(
            GUIButton(
                id="board",
                label="BOARD",
                rect=(bx, by, 76, bh),
                action=lambda: self.show_toast(f"Canvas: {engine.toggle_canvas_mode().capitalize()}")
            )
        )
        bx += 76 + gap

        # 3. Undo Button
        self.buttons.append(
            GUIButton(
                id="undo",
                label="UNDO",
                rect=(bx, by, 68, bh),
                action=lambda: self.show_toast("Undone") if engine.undo() else None
            )
        )
        bx += 68 + gap

        # 4. Clear Button
        self.buttons.append(
            GUIButton(
                id="clear",
                label="CLEAR",
                rect=(bx, by, 68, bh),
                action=lambda: (engine.clear(), self.show_toast("Cleared"))
            )
        )
        bx += 68 + gap

        # 5. Save Button
        self.buttons.append(
            GUIButton(
                id="save",
                label="SAVE",
                rect=(bx, by, 72, bh),
                action=on_save
            )
        )

    def _on_color(self, engine: DrawingEngine, color_key: str) -> None:
        engine.set_color(color_key)
        self.show_toast(f"Brush: {color_key.capitalize()}")

    def update_touchless(self, pointer: Optional[Tuple[float, float]], is_hover_mode: bool) -> None:
        """Process contactless dwell-time hover clicks."""
        now = time.perf_counter()
        if now < self._cooldown or not is_hover_mode or pointer is None:
            self._reset_hover()
            return

        px, py = pointer
        hit: Optional[GUIButton] = None
        for b in self.buttons:
            x, y, w, h = b.rect
            if x <= px <= x + w and y <= py <= y + h:
                hit = b
                break

        if hit is None:
            self._reset_hover()
            return

        if self._hover_btn_id != hit.id:
            self._hover_btn_id = hit.id
            self._hover_start = now
            for b in self.buttons:
                b.dwell = 0.0

        elapsed = now - (self._hover_start or now)
        hit.dwell = min(1.0, elapsed / self.dwell_sec)

        if hit.dwell >= 1.0:
            hit.action()
            self._cooldown = now + 0.8
            self._reset_hover()

    def _reset_hover(self) -> None:
        self._hover_btn_id = None
        self._hover_start = None
        for b in self.buttons:
            b.dwell = 0.0

    def render(
        self,
        frame: np.ndarray,
        engine: DrawingEngine,
        gesture: GestureType,
        pointer: Optional[Tuple[float, float]],
        fps: float,
        palm_scale: float
    ) -> None:
        """Render floating glassmorphic HUD directly on frame."""
        # 1. Floating Top Toolbar Panel
        if self.buttons:
            min_x = min(b.rect[0] for b in self.buttons) - 12
            max_x = max(b.rect[0] + b.rect[2] for b in self.buttons) + 12
            min_y = min(b.rect[1] for b in self.buttons) - 8
            max_y = max(b.rect[1] + b.rect[3] for b in self.buttons) + 8

            # Frosted glass panel
            panel = frame[min_y:max_y, min_x:max_x]
            if panel.shape[0] > 0 and panel.shape[1] > 0:
                overlay = panel.copy()
                cv2.rectangle(overlay, (0, 0), (max_x - min_x, max_y - min_y), (20, 20, 26), -1)
                cv2.addWeighted(overlay, 0.85, panel, 0.15, 0, panel)
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (60, 60, 75), 1, cv2.LINE_AA)

            for b in self.buttons:
                bx, by, bw, bh = b.rect
                active = (b.is_color and b.id == f"col_{engine.current_color_key}")

                # Button box
                btn_bg = (55, 55, 70) if active else (35, 35, 45)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), btn_bg, -1)
                border_c = (255, 200, 0) if active else (70, 70, 85)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_c, 1, cv2.LINE_AA)

                if b.is_color and b.color_bgr is not None:
                    # Color circle
                    center = (bx + bw // 2, by + bh // 2)
                    cv2.circle(frame, center, 14, b.color_bgr, -1, cv2.LINE_AA)
                    if active:
                        cv2.circle(frame, center, 17, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    # Text label
                    (tw, th), _ = cv2.getTextSize(b.label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    tx = bx + (bw - tw) // 2
                    ty = by + (bh + th) // 2
                    col = (255, 255, 255) if active else (220, 220, 230)
                    cv2.putText(frame, b.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

                # Dwell progress indicator fill
                if b.dwell > 0.0:
                    prog_h = int(bh * b.dwell)
                    cv2.rectangle(frame, (bx, by + bh - prog_h), (bx + bw, by + bh), (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Bottom-Left Gesture Status Badge
        badge_configs = {
            GestureType.DRAW: ("AIR-DRAW", (0, 255, 200), (0, 50, 30)),
            GestureType.HOVER: ("HOVER / SELECT", (255, 200, 0), (50, 40, 0)),
            GestureType.ERASE: ("ERASER", (0, 100, 255), (0, 20, 50)),
            GestureType.PINCH_SIZE: ("BRUSH ADJUST", (255, 50, 200), (50, 10, 40)),
            GestureType.IDLE: ("IDLE (Open Palm)", (180, 180, 190), (35, 35, 40)),
            GestureType.UNKNOWN: ("SEARCHING...", (120, 120, 120), (25, 25, 30)),
        }
        g_name, g_col, g_bg = badge_configs.get(gesture, ("TRACKING", (200, 200, 200), (30, 30, 30)))

        rx, ry, rw, rh = 24, self.height - 60, 210, 38
        p = frame[ry:ry+rh, rx:rx+rw]
        if p.shape[0] > 0 and p.shape[1] > 0:
            ov = p.copy()
            cv2.rectangle(ov, (0, 0), (rw, rh), g_bg, -1)
            cv2.addWeighted(ov, 0.85, p, 0.15, 0, p)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), g_col, 1, cv2.LINE_AA)
            cv2.circle(frame, (rx + 20, ry + 19), 6, g_col, -1, cv2.LINE_AA)
            cv2.putText(frame, g_name, (rx + 36, ry + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Bottom-Right Performance & Distance Telemetry
        dist_zone = "Close" if palm_scale > 140 else ("Optimal" if palm_scale >= 75 else "Far")
        tx, ty, tw, th = self.width - 230, self.height - 60, 206, 38
        tp = frame[ty:ty+th, tx:tx+tw]
        if tp.shape[0] > 0 and tp.shape[1] > 0:
            ov = tp.copy()
            cv2.rectangle(ov, (0, 0), (tw, th), (20, 20, 25), -1)
            cv2.addWeighted(ov, 0.85, tp, 0.15, 0, tp)
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (60, 60, 75), 1, cv2.LINE_AA)
            info = f"FPS: {fps:4.1f} | {dist_zone}"
            cv2.putText(frame, info, (tx + 16, ty + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 200) if fps >= 30 else (255, 180, 0), 1, cv2.LINE_AA)

        # 4. Pointer Reticle
        if pointer is not None:
            px, py = int(pointer[0]), int(pointer[1])
            if gesture == GestureType.DRAW:
                cv2.circle(frame, (px, py), engine.brush_radius, engine.current_color_bgr, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
            elif gesture == GestureType.HOVER:
                cv2.circle(frame, (px, py), 16, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1, cv2.LINE_AA)
            elif gesture == GestureType.ERASE:
                cv2.circle(frame, (px, py), engine.eraser_radius, (0, 0, 255), 2, cv2.LINE_AA)
            elif gesture == GestureType.PINCH_SIZE:
                cv2.circle(frame, (px, py), engine.brush_radius, (255, 0, 255), 2, cv2.LINE_AA)

        # 5. Toast Notification
        now = time.perf_counter()
        if self.toast_msg and now < self.toast_expiry:
            (msg_w, msg_h), _ = cv2.getTextSize(self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cx, cy = self.width // 2, 92
            cv2.rectangle(frame, (cx - msg_w // 2 - 12, cy - msg_h - 10), (cx + msg_w // 2 + 12, cy + 10), (25, 25, 32), -1)
            cv2.rectangle(frame, (cx - msg_w // 2 - 12, cy - msg_h - 10), (cx + msg_w // 2 + 12, cy + 10), (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, self.toast_msg, (cx - msg_w // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

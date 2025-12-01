"""Interactive touchless toolbar with dwell-time button activation.

Enables users to trigger actions (color selection, eraser, canvas mode,
undo, clear, save) by hovering the index finger over buttons without physical input.
"""

from dataclasses import dataclass
import time
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np
from config.settings import (
    BUTTON_DWELL_TIME_SEC,
    COLOR_PALETTE,
    HUD_ACCENT_COLOR,
    HUD_BG_COLOR,
    HUD_BORDER_COLOR,
    HUD_TEXT_COLOR,
)


@dataclass
class ToolbarButton:
    """Individual interactive UI button."""
    id: str
    label: str
    rect: Tuple[int, int, int, int]  # (x, y, width, height)
    action: Callable[[], None]
    icon_type: str = "text"          # "text", "color", "icon"
    color_bgr: Optional[Tuple[int, int, int]] = None
    is_active: bool = False
    dwell_progress: float = 0.0      # 0.0 to 1.0


class TouchlessToolbar:
    """Manages hit-testing and contactless dwell-time activation of UI buttons."""

    def __init__(self, dwell_time_sec: float = BUTTON_DWELL_TIME_SEC):
        self.dwell_time_sec = dwell_time_sec
        self.buttons: List[ToolbarButton] = []
        self._hovered_button_id: Optional[str] = None
        self._hover_start_time: Optional[float] = None
        self._last_triggered_id: Optional[str] = None
        self._trigger_cooldown_time: float = 0.0

    def add_button(
        self,
        btn_id: str,
        label: str,
        rect: Tuple[int, int, int, int],
        action: Callable[[], None],
        icon_type: str = "text",
        color_bgr: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """Register a new interactive button."""
        self.buttons.append(
            ToolbarButton(
                id=btn_id,
                label=label,
                rect=rect,
                action=action,
                icon_type=icon_type,
                color_bgr=color_bgr
            )
        )

    def hit_test(self, point: Tuple[float, float], padding: int = 0) -> Optional[ToolbarButton]:
        """Check if pointer coordinate falls inside any button bounding box."""
        px, py = point
        for btn in self.buttons:
            x, y, w, h = btn.rect
            if (x - padding) <= px <= (x + w + padding) and (y - padding) <= py <= (y + h + padding):
                return btn
        return None

    def update(self, pointer: Optional[Tuple[float, float]], is_hover_mode: bool, padding: int = 0) -> Optional[str]:
        """Update hover state and trigger action upon dwell-time completion.

        Returns button ID if an action was triggered this frame, else None.
        """
        now = time.perf_counter()

        # Enforce trigger cooldown to avoid accidental double-taps
        if now < self._trigger_cooldown_time:
            return None

        # If hand not in hover mode or no pointer, reset dwell
        if not is_hover_mode or pointer is None:
            self._reset_hover()
            return None

        hit_btn = self.hit_test(pointer, padding=padding)

        if hit_btn is None:
            self._reset_hover()
            return None

        # Hovering over a button
        if self._hovered_button_id != hit_btn.id:
            self._hovered_button_id = hit_btn.id
            self._hover_start_time = now
            for b in self.buttons:
                b.dwell_progress = 0.0

        elapsed = now - (self._hover_start_time or now)
        hit_btn.dwell_progress = min(1.0, elapsed / self.dwell_time_sec)

        # Trigger action when dwell completes
        if hit_btn.dwell_progress >= 1.0:
            hit_btn.action()
            triggered_id = hit_btn.id
            self._trigger_cooldown_time = now + 0.8  # Cooldown before next click
            self._reset_hover()
            return triggered_id

        return None

    def _reset_hover(self) -> None:
        """Reset internal hover timers and progress."""
        self._hovered_button_id = None
        self._hover_start_time = None
        for b in self.buttons:
            b.dwell_progress = 0.0

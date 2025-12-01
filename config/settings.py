"""Air-Pen Studio Configuration Settings.

Centralized configuration parameters for computer vision processing,
gesture recognition, drawing engine, UI styling, and export targets.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class CanvasMode(str, Enum):
    """Background canvas display mode."""
    OVERLAY = "Camera Overlay"
    BLACKBOARD = "Matte Blackboard"
    WHITEBOARD = "Studio Whiteboard"


class GestureMode(str, Enum):
    """Recognized user hand gestures."""
    IDLE = "Idle"
    HOVER = "Hover / Select"
    DRAW = "Air-Draw"
    ERASE = "Eraser"
    PINCH_SIZE = "Brush Adjust"
    UNKNOWN = "Tracking..."


# ---------------------------------------------------------
# Camera & Video Feed Settings
# ---------------------------------------------------------
CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
CAMERA_FPS: int = 60
MIRROR_FEED: bool = True

# ---------------------------------------------------------
# MediaPipe Hand Tracking Settings
# ---------------------------------------------------------
MAX_NUM_HANDS: int = 1
MIN_DETECTION_CONFIDENCE: float = 0.70
MIN_TRACKING_CONFIDENCE: float = 0.65
MODEL_COMPLEXITY: int = 1  # 0: Lite, 1: Full

# ---------------------------------------------------------
# One-Euro Filter (Jitter Removal) Parameters
# ---------------------------------------------------------
FILTER_MIN_CUTOFF: float = 1.2    # Lower values remove more static jitter (Hz)
FILTER_BETA: float = 0.015         # Higher values reduce lag during fast strokes
FILTER_D_CUTOFF: float = 1.0       # Cutoff frequency for derivative (Hz)

# ---------------------------------------------------------
# Long-Distance Adaptive Scaling
# ---------------------------------------------------------
REFERENCE_PALM_SIZE: float = 110.0  # Typical palm pixel size at ~0.6m distance
MIN_SCALE_FACTOR: float = 0.35     # Extreme distance clamp
MAX_SCALE_FACTOR: float = 2.50     # Close-up clamp

# ---------------------------------------------------------
# Gesture Recognition Thresholds
# ---------------------------------------------------------
FINGER_EXTENDED_ANGLE_THRESHOLD: float = 150.0  # Degrees for straight finger
FINGER_CURLED_ANGLE_THRESHOLD: float = 95.0     # Degrees for curled finger
PINCH_NORMALIZED_DISTANCE: float = 0.08         # Tip distance / palm_size
GESTURE_HYSTERESIS_FRAMES: int = 2              # Smoothing state transitions
BUTTON_DWELL_TIME_SEC: float = 0.65             # Hover duration to trigger UI button

# ---------------------------------------------------------
# Color Palette (BGR format for OpenCV)
# ---------------------------------------------------------
@dataclass(frozen=True)
class PaletteColor:
    name: str
    bgr: Tuple[int, int, int]
    hex_code: str
    glow_bgr: Tuple[int, int, int]


COLOR_PALETTE: Dict[str, PaletteColor] = {
    "neon_cyan": PaletteColor(
        name="Neon Cyan",
        bgr=(255, 240, 0),
        hex_code="#00F0FF",
        glow_bgr=(255, 180, 0)
    ),
    "electric_pink": PaletteColor(
        name="Hot Magenta",
        bgr=(220, 20, 255),
        hex_code="#FF14DC",
        glow_bgr=(180, 0, 220)
    ),
    "emerald_lime": PaletteColor(
        name="Emerald Lime",
        bgr=(50, 255, 50),
        hex_code="#32FF32",
        glow_bgr=(0, 200, 0)
    ),
    "electric_amber": PaletteColor(
        name="Sunburst Amber",
        bgr=(0, 165, 255),
        hex_code="#FFA500",
        glow_bgr=(0, 100, 255)
    ),
    "pure_white": PaletteColor(
        name="Bright White",
        bgr=(255, 255, 255),
        hex_code="#FFFFFF",
        glow_bgr=(200, 200, 200)
    ),
}

DEFAULT_COLOR_KEY: str = "neon_cyan"

# ---------------------------------------------------------
# Brush & Canvas Defaults
# ---------------------------------------------------------
DEFAULT_BRUSH_RADIUS: int = 6
MIN_BRUSH_RADIUS: int = 2
MAX_BRUSH_RADIUS: int = 40
DEFAULT_ERASER_RADIUS: int = 35

ENABLE_NEON_GLOW: bool = True
GLOW_RADIUS_RATIO: float = 2.4
GLOW_ALPHA: float = 0.45

CANVAS_BLACKBOARD_BGR: Tuple[int, int, int] = (22, 22, 26)
CANVAS_WHITEBOARD_BGR: Tuple[int, int, int] = (248, 249, 250)

# ---------------------------------------------------------
# UI Styling
# ---------------------------------------------------------
HUD_BG_COLOR: Tuple[int, int, int] = (30, 30, 36)
HUD_BORDER_COLOR: Tuple[int, int, int] = (70, 70, 85)
HUD_TEXT_COLOR: Tuple[int, int, int] = (240, 240, 245)
HUD_ACCENT_COLOR: Tuple[int, int, int] = (255, 200, 0)

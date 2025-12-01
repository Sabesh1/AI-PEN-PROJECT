"""Unit tests for hand gesture classification and hysteresis."""

import pytest
from config.settings import GestureMode
from core.gesture_recognizer import GestureRecognizer
from core.hand_detector import HandResult, LandmarkPoint


def create_mock_hand(extended_fingers: list[str], pinch: bool = False) -> HandResult:
    """Generate synthetic hand landmarks with specified fingers extended."""
    lms = []
    # Wrist at (300, 500)
    wrist_x, wrist_y = 300.0, 500.0

    # Initialize all 21 points at wrist
    for i in range(21):
        lms.append(LandmarkPoint(index=i, x=wrist_x, y=wrist_y, z=0.0, norm_x=0.5, norm_y=0.7))

    # Base joints (MCPs)
    lms[1].x, lms[1].y = 260.0, 460.0  # Thumb CMC
    lms[2].x, lms[2].y = 240.0, 430.0  # Thumb MCP
    lms[3].x, lms[3].y = 230.0, 400.0  # Thumb IP
    lms[4].x, lms[4].y = 220.0, 370.0  # Thumb TIP

    lms[5].x, lms[5].y = 280.0, 380.0  # Index MCP
    lms[9].x, lms[9].y = 300.0, 370.0  # Middle MCP
    lms[13].x, lms[13].y = 320.0, 380.0 # Ring MCP
    lms[17].x, lms[17].y = 340.0, 400.0 # Pinky MCP

    finger_map = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    for name, (mcp_i, pip_i, dip_i, tip_i) in finger_map.items():
        base_x = lms[mcp_i].x
        if name in extended_fingers:
            # Extended straight upward
            lms[pip_i].x, lms[pip_i].y = base_x, 300.0
            lms[dip_i].x, lms[dip_i].y = base_x, 240.0
            lms[tip_i].x, lms[tip_i].y = base_x, 180.0
        else:
            # Curled into palm
            lms[pip_i].x, lms[pip_i].y = base_x, 350.0
            lms[dip_i].x, lms[dip_i].y = base_x, 380.0
            lms[tip_i].x, lms[tip_i].y = base_x, 400.0

    if pinch:
        # Bring thumb and index tip together
        lms[4].x, lms[4].y = lms[8].x - 5.0, lms[8].y + 5.0

    return HandResult(landmarks=lms, handedness="Right", confidence=0.95, bbox=(200, 150, 160, 360))


def test_drawing_gesture_classification():
    """Verify only index extended classifies as DRAW."""
    recognizer = GestureRecognizer(hysteresis_frames=1)
    hand = create_mock_hand(extended_fingers=["index"])
    mode = recognizer.classify_gesture(hand, palm_scale=130.0)
    assert mode == GestureMode.DRAW


def test_hover_gesture_classification():
    """Verify index and middle extended (peace sign) classifies as HOVER."""
    recognizer = GestureRecognizer(hysteresis_frames=1)
    hand = create_mock_hand(extended_fingers=["index", "middle"])
    mode = recognizer.classify_gesture(hand, palm_scale=130.0)
    assert mode == GestureMode.HOVER


def test_eraser_gesture_classification():
    """Verify three fingers extended classifies as ERASE."""
    recognizer = GestureRecognizer(hysteresis_frames=1)
    hand = create_mock_hand(extended_fingers=["index", "middle", "ring"])
    mode = recognizer.classify_gesture(hand, palm_scale=130.0)
    assert mode == GestureMode.ERASE


def test_idle_open_palm_classification():
    """Verify all fingers extended classifies as IDLE."""
    recognizer = GestureRecognizer(hysteresis_frames=1)
    hand = create_mock_hand(extended_fingers=["index", "middle", "ring", "pinky"])
    mode = recognizer.classify_gesture(hand, palm_scale=130.0)
    assert mode == GestureMode.IDLE


def test_pinch_gesture_classification():
    """Verify pinch between thumb and index classifies as PINCH_SIZE."""
    recognizer = GestureRecognizer(hysteresis_frames=1)
    hand = create_mock_hand(extended_fingers=["index"], pinch=True)
    mode = recognizer.classify_gesture(hand, palm_scale=130.0)
    assert mode == GestureMode.PINCH_SIZE

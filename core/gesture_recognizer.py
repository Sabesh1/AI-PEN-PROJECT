"""Gesture recognition engine with orientation-invariant kinematics and hysteresis.

Recognizes drawing, hovering, erasing, brush resizing, and idle gestures using
joint distance ratios, angular extension checks, and temporal debounce filters.
"""

from collections import deque
import math
from typing import Dict, List, Optional, Tuple
from config.settings import (
    GESTURE_HYSTERESIS_FRAMES,
    GestureMode,
    PINCH_NORMALIZED_DISTANCE,
)
from core.finger_tracker import FingerTracker
from core.hand_detector import HandResult, LandmarkPoint


class GestureRecognizer:
    """Classifies natural hand gestures with high robustness and low false positives."""

    def __init__(self, hysteresis_frames: int = GESTURE_HYSTERESIS_FRAMES):
        self.hysteresis_frames = max(1, hysteresis_frames)
        self._history: deque[GestureMode] = deque(maxlen=self.hysteresis_frames)
        self.current_mode: GestureMode = GestureMode.IDLE
        self.pinch_distance_ratio: float = 0.0

    @staticmethod
    def _is_finger_extended(
        wrist: LandmarkPoint,
        mcp: LandmarkPoint,
        pip: LandmarkPoint,
        tip: LandmarkPoint
    ) -> bool:
        """Determine if a finger is extended using orientation-invariant distance & vector angles."""
        # 1. Distance from wrist to tip vs wrist to pip
        d_wrist_tip = math.hypot(wrist.x - tip.x, wrist.y - tip.y)
        d_wrist_pip = math.hypot(wrist.x - pip.x, wrist.y - pip.y)

        # 2. Angle at PIP joint (between MCP->PIP and PIP->TIP)
        v1 = (pip.x - mcp.x, pip.y - mcp.y)
        v2 = (tip.x - pip.x, tip.y - pip.y)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])

        cos_angle = dot / (mag1 * mag2 + 1e-6)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))

        # Extended if tip is significantly farther than pip AND joint is straight (angle < 45 deg)
        return (d_wrist_tip > d_wrist_pip * 1.15) and (angle_deg < 50.0)

    @staticmethod
    def _is_thumb_extended(
        wrist: LandmarkPoint,
        mcp: LandmarkPoint,
        ip: LandmarkPoint,
        tip: LandmarkPoint,
        pinky_mcp: LandmarkPoint
    ) -> bool:
        """Determine if thumb is extended away from palm."""
        d_thumb_pinky = math.hypot(tip.x - pinky_mcp.x, tip.y - pinky_mcp.y)
        d_ip_pinky = math.hypot(ip.x - pinky_mcp.x, ip.y - pinky_mcp.y)
        return d_thumb_pinky > d_ip_pinky * 1.10

    def analyze_fingers(self, hand: HandResult) -> Dict[str, bool]:
        """Detect extended status for all 5 fingers."""
        lms = hand.landmarks
        wrist = lms[FingerTracker.WRIST]
        pinky_mcp = lms[FingerTracker.PINKY_MCP]

        thumb_extended = self._is_thumb_extended(
            wrist,
            lms[FingerTracker.THUMB_MCP],
            lms[FingerTracker.THUMB_IP],
            lms[FingerTracker.THUMB_TIP],
            pinky_mcp
        )
        index_extended = self._is_finger_extended(
            wrist,
            lms[FingerTracker.INDEX_MCP],
            lms[FingerTracker.INDEX_PIP],
            lms[FingerTracker.INDEX_TIP]
        )
        middle_extended = self._is_finger_extended(
            wrist,
            lms[FingerTracker.MIDDLE_MCP],
            lms[FingerTracker.MIDDLE_PIP],
            lms[FingerTracker.MIDDLE_TIP]
        )
        ring_extended = self._is_finger_extended(
            wrist,
            lms[FingerTracker.RING_MCP],
            lms[FingerTracker.RING_PIP],
            lms[FingerTracker.RING_TIP]
        )
        pinky_extended = self._is_finger_extended(
            wrist,
            lms[FingerTracker.PINKY_MCP],
            lms[FingerTracker.PINKY_PIP],
            lms[FingerTracker.PINKY_TIP]
        )

        return {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended,
        }

    def classify_gesture(self, hand: HandResult, palm_scale: float) -> GestureMode:
        """Classify gesture into one of the defined interaction modes."""
        fingers = self.analyze_fingers(hand)
        lms = hand.landmarks

        # Calculate normalized pinch distance between thumb and index tips
        d_thumb_index = math.hypot(
            lms[FingerTracker.THUMB_TIP].x - lms[FingerTracker.INDEX_TIP].x,
            lms[FingerTracker.THUMB_TIP].y - lms[FingerTracker.INDEX_TIP].y
        )
        self.pinch_distance_ratio = d_thumb_index / max(10.0, palm_scale)

        idx = fingers["index"]
        mid = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]
        thumb = fingers["thumb"]

        # Count extended non-thumb fingers
        extended_count = sum([idx, mid, ring, pinky])

        # 1. PINCH ADJUST MODE: Thumb and Index in close proximity while others curled
        if self.pinch_distance_ratio < PINCH_NORMALIZED_DISTANCE and not mid and not ring and not pinky:
            raw_mode = GestureMode.PINCH_SIZE

        # 2. DRAW MODE: Only Index finger extended (classic Air-Pen pointer)
        elif idx and not mid and not ring and not pinky:
            raw_mode = GestureMode.DRAW

        # 3. HOVER / SELECT MODE: Index and Middle extended (Peace Sign ✌️)
        elif idx and mid and not ring and not pinky:
            raw_mode = GestureMode.HOVER

        # 4. ERASER MODE: Three fingers extended (Index, Middle, Ring) OR Fist (all curled)
        elif (idx and mid and ring and not pinky) or (extended_count == 0 and not thumb):
            raw_mode = GestureMode.ERASE

        # 5. IDLE / PAUSE: Open palm (all extended) or loose resting hand
        elif extended_count >= 4:
            raw_mode = GestureMode.IDLE

        else:
            raw_mode = GestureMode.HOVER if idx else GestureMode.IDLE

        # Temporal Hysteresis filter: mode must appear consistently
        self._history.append(raw_mode)

        # If all recent frames match the candidate, switch immediately
        if self._history.count(raw_mode) >= max(1, self.hysteresis_frames - 1):
            self.current_mode = raw_mode

        return self.current_mode

    def reset(self) -> None:
        """Reset gesture history."""
        self._history.clear()
        self.current_mode = GestureMode.IDLE
        self.pinch_distance_ratio = 0.0

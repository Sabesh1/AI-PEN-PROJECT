"""Finger tracking and scale normalization module.

Isolates critical fingertips (Index, Thumb, Middle) and estimates palm scale
to ensure all geometric computations are invariant to camera distance.
"""

import math
import time
from typing import Dict, Optional, Tuple
from core.hand_detector import HandResult, LandmarkPoint


class FingerTracker:
    """Tracks fingertips and normalizes spatial distances."""

    # Key landmark indices from MediaPipe Hands
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self):
        self._prev_index_pos: Optional[Tuple[float, float]] = None
        self._prev_time: Optional[float] = None
        self.instantaneous_speed: float = 0.0

    @staticmethod
    def euclidean_distance(p1: LandmarkPoint, p2: LandmarkPoint) -> float:
        """Calculate 2D Euclidean pixel distance between two landmarks."""
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.hypot(dx, dy)

    @staticmethod
    def calculate_palm_scale(hand: HandResult) -> float:
        """Calculate palm scale based on distance between Wrist and Middle MCP joint.

        This bone length does not flex or change during finger bending,
        making it an ideal scale reference across camera distances.
        """
        wrist = hand.landmarks[FingerTracker.WRIST]
        middle_mcp = hand.landmarks[FingerTracker.MIDDLE_MCP]
        scale = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
        return max(10.0, scale)

    def extract_key_points(self, hand: HandResult) -> Dict[str, Tuple[float, float]]:
        """Extract pixel coordinates for key interaction fingertips."""
        lms = hand.landmarks
        return {
            "index_tip": (lms[self.INDEX_TIP].x, lms[self.INDEX_TIP].y),
            "thumb_tip": (lms[self.THUMB_TIP].x, lms[self.THUMB_TIP].y),
            "middle_tip": (lms[self.MIDDLE_TIP].x, lms[self.MIDDLE_TIP].y),
            "ring_tip": (lms[self.RING_TIP].x, lms[self.RING_TIP].y),
            "pinky_tip": (lms[self.PINKY_TIP].x, lms[self.PINKY_TIP].y),
            "wrist": (lms[self.WRIST].x, lms[self.WRIST].y),
        }

    def update_velocity(self, current_pos: Tuple[float, float], timestamp: Optional[float] = None) -> float:
        """Update and return pointer velocity in pixels/second."""
        if timestamp is None:
            timestamp = time.perf_counter()

        if self._prev_index_pos is None or self._prev_time is None:
            self._prev_index_pos = current_pos
            self._prev_time = timestamp
            self.instantaneous_speed = 0.0
            return 0.0

        dt = timestamp - self._prev_time
        if dt > 0.0001:
            dx = current_pos[0] - self._prev_index_pos[0]
            dy = current_pos[1] - self._prev_index_pos[1]
            dist = math.hypot(dx, dy)
            self.instantaneous_speed = dist / dt
        else:
            self.instantaneous_speed = 0.0

        self._prev_index_pos = current_pos
        self._prev_time = timestamp
        return self.instantaneous_speed

    def reset(self) -> None:
        """Reset velocity and position tracking."""
        self._prev_index_pos = None
        self._prev_time = None
        self.instantaneous_speed = 0.0

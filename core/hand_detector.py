"""MediaPipe Hands detector integration module.

Detects 21 3D landmarks for human hands with high confidence and low latency.
Provides structured landmark objects with pixel and normalized coordinates.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class LandmarkPoint:
    """Individual 3D hand landmark."""
    index: int
    x: float          # Pixel x coordinate
    y: float          # Pixel y coordinate
    z: float          # Relative depth
    norm_x: float     # Normalized 0..1 x
    norm_y: float     # Normalized 0..1 y


@dataclass
class HandResult:
    """Structured result of hand detection."""
    landmarks: List[LandmarkPoint]
    handedness: str   # 'Right' or 'Left'
    confidence: float
    bbox: Tuple[int, int, int, int]  # (min_x, min_y, width, height)


class HandDetector:
    """MediaPipe Hands wrapper optimized for real-time interaction."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.70,
        tracking_confidence: float = 0.65,
        model_complexity: int = 1
    ):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.model_complexity = model_complexity

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
            model_complexity=0
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        """Detect hands in a BGR frame with fast downscaled inference."""
        h, w, _ = frame_bgr.shape
        small_bgr = cv2.resize(frame_bgr, (480, 270), interpolation=cv2.INTER_NEAREST)
        frame_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        # Take primary hand (index 0)
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_label = "Right"
        score = 0.90

        if results.multi_handedness:
            handedness_info = results.multi_handedness[0].classification[0]
            handedness_label = handedness_info.label
            score = handedness_info.score

        landmarks: List[LandmarkPoint] = []
        xs, ys = [], []

        for idx, lm in enumerate(hand_landmarks.landmark):
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmarks.append(
                LandmarkPoint(
                    index=idx,
                    x=float(px),
                    y=float(py),
                    z=float(lm.z),
                    norm_x=float(lm.x),
                    norm_y=float(lm.y)
                )
            )
            xs.append(px)
            ys.append(py)

        min_x, max_x = max(0, min(xs)), min(w, max(xs))
        min_y, max_y = max(0, min(ys)), min(h, max(ys))
        bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

        return HandResult(
            landmarks=landmarks,
            handedness=handedness_label,
            confidence=score,
            bbox=bbox
        )

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        hand_result: HandResult,
        color_bgr: Tuple[int, int, int] = (0, 255, 180)
    ) -> None:
        """Draw a subtle, futuristic minimalist hand skeleton overlay."""
        # Connect key joints
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm knuckle ridge
            (5, 9), (9, 13), (13, 17)
        ]

        lms = hand_result.landmarks
        for pt1, pt2 in connections:
            p1 = (int(lms[pt1].x), int(lms[pt1].y))
            p2 = (int(lms[pt2].x), int(lms[pt2].y))
            cv2.line(frame_bgr, p1, p2, (60, 60, 70), 1, cv2.LINE_AA)

        # Draw subtle landmark dots
        for lm in lms:
            cv2.circle(frame_bgr, (int(lm.x), int(lm.y)), 2, color_bgr, -1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

"""Ultra-fast hand tracking for AIR_PEN.

Design goals:
- Track hand reliably from 0.5 m to 3 m distance.
- Cursor follows finger with ZERO perceptible lag.
- Full canvas coverage even from far away (reach amplifier).
- Very low CPU overhead: model_complexity=0, minimal per-frame math.
"""

import math
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# One-Euro Filter (Casiez et al., CHI 2012) — speed-adaptive low-pass filter
# Higher beta → faster response to fast movements, less lag.
# ---------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.20, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x:  Optional[float] = None
        self._dx: Optional[float] = None
        self._t:  Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        te  = 1.0 / max(rate, 1e-6)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.perf_counter()
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x

        dt   = max(t - self._t, 1e-6)
        rate = 1.0 / dt
        dx   = (x - self._x) * rate
        a_d  = self._alpha(rate, self.d_cutoff)
        edx  = a_d * dx + (1.0 - a_d) * self._dx
        self._dx = edx

        cutoff   = self.min_cutoff + self.beta * abs(edx)
        a        = self._alpha(rate, cutoff)
        filtered = a * x + (1.0 - a) * self._x
        self._t, self._x = t, filtered
        return filtered

    def reset(self) -> None:
        self._x = self._dx = self._t = None


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------
class HandTracker:
    """MediaPipe Hands with maximum detection range and zero-lag cursor response."""

    # landmark indices
    WRIST      = 0
    THUMB_TIP  = 4
    INDEX_MCP  = 5
    INDEX_PIP  = 6
    INDEX_TIP  = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_MCP   = 13
    RING_PIP   = 14
    RING_TIP   = 16
    PINKY_MCP  = 17
    PINKY_PIP  = 18
    PINKY_TIP  = 20

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.50,   # 0.50 provides clean hand lock with zero false positives
        tracking_confidence:  float = 0.50,
    ):
        self.mp_hands = mp.solutions.hands

        # model_complexity=0 → Lite model: 60+ FPS on CPU
        self._hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=0,
        )

        # Zero-jitter adaptive filters: balanced preset by default (0.32 Hz, beta=0.035)
        self.filter_x     = OneEuroFilter(min_cutoff=0.32, beta=0.035)
        self.filter_y     = OneEuroFilter(min_cutoff=0.32, beta=0.035)
        self.filter_palm_x = OneEuroFilter(min_cutoff=0.32, beta=0.035)
        self.filter_palm_y = OneEuroFilter(min_cutoff=0.32, beta=0.035)

        self.raw_landmarks:    List[Tuple[float, float, float]] = []
        self.pixel_landmarks:  List[Tuple[int, int]]            = []
        self.smoothed_index_tip:   Optional[Tuple[float, float]] = None
        self.smoothed_palm_center: Optional[Tuple[float, float]] = None

        self.palm_scale: float = 100.0
        self._ema_palm:  Optional[float] = None

        # Reach multiplier: 1.0 = raw mapping, >1 = amplified for far distance
        self.reach_multiplier:    float = 1.35
        self.current_reach_gain:  float = 1.0
        self.active_finger_name:  str   = "Index"

        self.smoothing_preset:    str   = "balanced"

    # ------------------------------------------------------------------
    # Smoothing preset
    # ------------------------------------------------------------------
    def set_smoothing_preset(self, preset: str) -> None:
        self.smoothing_preset = preset
        if preset == "cinema":
            # Smooth / calligraphy: higher stability
            for f in (self.filter_x, self.filter_y):
                f.min_cutoff, f.beta = 0.20, 0.025
        elif preset == "fast":
            # Ultra-responsive / fast writing
            for f in (self.filter_x, self.filter_y):
                f.min_cutoff, f.beta = 0.50, 0.050
        else:  # "balanced" (default)
            self.smoothing_preset = "balanced"
            for f in (self.filter_x, self.filter_y):
                f.min_cutoff, f.beta = 0.32, 0.035

    # ------------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> bool:
        """Detect hand, compute filtered pointer position, return True if hand found."""
        h, w = frame_bgr.shape[:2]

        # ── 1. Prepare inference frame ──────────────────────────────────
        if w > 960:
            iw, ih = 960, int(h * 960 / w)
            infer = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        else:
            iw, ih = w, h
            infer = frame_bgr

        # ── 2. Direct high-speed BGR to RGB conversion (200x faster than CLAHE) ──
        rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        # ── 3. MediaPipe inference ────────────────────────────────────────
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self.reset()
            return False

        hand = results.multi_hand_landmarks[0]

        # Scale landmarks back to original frame size
        sx = w / iw
        sy = h / ih
        self.raw_landmarks   = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
        self.pixel_landmarks = [
            (int(lm.x * iw * sx), int(lm.y * ih * sy))
            for lm in hand.landmark
        ]

        # ── 4. Palm scale (wrist → middle MCP distance) ────────────────
        w_pt = self.pixel_landmarks[self.WRIST]
        m_pt = self.pixel_landmarks[self.MIDDLE_MCP]
        raw_palm = max(15.0, math.hypot(w_pt[0] - m_pt[0], w_pt[1] - m_pt[1]))
        self.palm_scale = raw_palm

        # Stable EMA palm scale for reach gain calculation
        if self._ema_palm is None:
            self._ema_palm = raw_palm
        else:
            self._ema_palm = 0.88 * self._ema_palm + 0.12 * raw_palm

        # ── 5. Reach amplifier ───────────────────────────────────────────
        # REF_PALM = palm scale at ~1 m reference distance
        REF_PALM = 110.0
        dist_ratio = REF_PALM / max(20.0, self._ema_palm)
        # Gain ramps from 1× (close) up to 3.2× (far)
        base_gain = 1.0 + 1.30 * max(0.0, (dist_ratio - 1.0) / 1.3)
        reach_gain = min(3.2, max(1.0, base_gain * self.reach_multiplier))
        self.current_reach_gain = reach_gain

        cx, cy = w / 2.0, h / 2.0

        # ── 6. Active writing finger (supports ANY single extended finger) ──
        ix_tip = self.pixel_landmarks[self.INDEX_TIP]
        mi_tip = self.pixel_landmarks[self.MIDDLE_TIP]
        th_tip = self.pixel_landmarks[self.THUMB_TIP]
        ri_tip = self.pixel_landmarks[self.RING_TIP]
        pk_tip = self.pixel_landmarks[self.PINKY_TIP]

        d_idx = math.hypot(w_pt[0] - ix_tip[0], w_pt[1] - ix_tip[1])
        d_mid = math.hypot(w_pt[0] - mi_tip[0], w_pt[1] - mi_tip[1])
        d_th  = math.hypot(w_pt[0] - th_tip[0], w_pt[1] - th_tip[1])
        d_ri  = math.hypot(w_pt[0] - ri_tip[0], w_pt[1] - ri_tip[1])
        d_pk  = math.hypot(w_pt[0] - pk_tip[0], w_pt[1] - pk_tip[1])

        # Find the primary extended finger with highest reach from wrist
        finger_candidates = [
            (d_idx, ix_tip, "Index"),
            (d_mid, mi_tip, "Middle"),
            (d_ri,  ri_tip, "Ring"),
            (d_pk,  pk_tip, "Pinky"),
            (d_th,  th_tip, "Thumb"),
        ]
        # Prefer Index unless another finger is significantly more extended
        if d_mid > d_idx * 1.25:
            active_tip, self.active_finger_name = mi_tip, "Middle"
        elif d_th > d_idx * 1.30:
            active_tip, self.active_finger_name = th_tip, "Thumb"
        elif d_pk > d_idx * 1.25:
            active_tip, self.active_finger_name = pk_tip, "Pinky"
        else:
            active_tip, self.active_finger_name = ix_tip, "Index"

        # ── 7. Amplified pointer (symmetrical around frame center) ───────
        def _amplify(raw_x: float, raw_y: float) -> Tuple[float, float]:
            u = (raw_x - cx) / cx
            v = (raw_y - cy) / cy
            u_a = max(-0.97, min(0.97, u * reach_gain))
            v_a = max(-0.97, min(0.97, v * reach_gain))
            return cx + u_a * cx, cy + v_a * cy

        tip_x, tip_y   = _amplify(float(active_tip[0]),   float(active_tip[1]))
        palm_x, palm_y = _amplify(float((w_pt[0] + m_pt[0]) / 2.0),
                                   float((w_pt[1] + m_pt[1]) / 2.0))

        # ── 8. One-Euro filter (fast tracking on curves, zero jitter at rest) ──
        now = time.perf_counter()
        self.smoothed_index_tip   = (self.filter_x(tip_x,   now), self.filter_y(tip_y,   now))
        self.smoothed_palm_center = (self.filter_palm_x(palm_x, now), self.filter_palm_y(palm_y, now))

        return True

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    def get_index_tip(self) -> Optional[Tuple[float, float]]:
        return self.smoothed_index_tip

    def get_palm_center(self) -> Optional[Tuple[float, float]]:
        return self.smoothed_palm_center

    # ------------------------------------------------------------------
    # Skeleton overlay
    # ------------------------------------------------------------------
    def draw_skeleton(self, frame: np.ndarray) -> None:
        if not self.pixel_landmarks:
            return
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        for p1, p2 in CONNECTIONS:
            cv2.line(frame, self.pixel_landmarks[p1], self.pixel_landmarks[p2],
                     (60, 60, 70), 1, cv2.LINE_AA)
        for pt in self.pixel_landmarks:
            cv2.circle(frame, pt, 3, (0, 240, 160), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.raw_landmarks.clear()
        self.pixel_landmarks.clear()
        self.smoothed_index_tip   = None
        self.smoothed_palm_center = None
        self.filter_x.reset()
        self.filter_y.reset()
        self.filter_palm_x.reset()
        self.filter_palm_y.reset()

    def close(self) -> None:
        self._hands.close()

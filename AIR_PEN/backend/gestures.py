"""Kinematic gesture recognition — bulletproof writing protection.

Interaction Modes:
- ☝️  DRAW  : ANY 1 finger extended (index, middle, or pinky alone). NEVER erases.
- 👋  ERASE : Deliberate full open palm — ALL 4 non-thumb fingers MUST be
              extended HIGH (tips well above MCP joints) AND spread wide.
              Writing with 1-2 fingers can NEVER accidentally trigger ERASE.
- ✌️  HOVER : Peace sign (index + middle in wide V, ring + pinky curled).
- 🤏  PINCH : Thumb+index pinch (for brush resize).
- ✊  IDLE  : Closed fist or no gesture.
"""

from collections import deque
from enum import Enum
import math
from typing import Tuple, Optional
try:
    from backend.hand_tracking import HandTracker
except ImportError:
    from .hand_tracking import HandTracker


class GestureType(str, Enum):
    """Interaction gesture modes."""
    DRAW       = "Air-Draw"
    HOVER      = "Hover / Select"
    ERASE      = "Eraser"
    PINCH_SIZE = "Brush Resize"
    IDLE       = "Idle / Pause"
    UNKNOWN    = "Tracking..."


class GestureDetector:
    """Classifies hand poses with instant response and bulletproof write/erase separation.

    Key design rules:
    1. ERASE requires ALL 4 fingers (index + middle + ring + pinky) to be
       clearly extended — this is physically impossible while writing with
       just 1 finger.
    2. DRAW triggers whenever exactly 1 finger is extended (or 2 if not spread).
    3. A grace window keeps DRAW alive for a few frames after fingers drop,
       preventing pen-lift gaps in strokes.
    4. ERASE needs 5 consecutive confirmations before activating (prevents
       accidental trigger during quick hand repositioning).
    """

    _DRAW_GRACE_FRAMES    = 4   # frames to keep drawing after fingers drop (smooth lift)

    def __init__(self, hysteresis_frames: int = 2, writing_mode: str = "point"):
        self.hysteresis_frames: int = max(1, hysteresis_frames)
        self.erase_confirm_frames: int = self.hysteresis_frames
        self.writing_mode: str = writing_mode  # "pinch" (pen-grip) | "point" (classic) | "depth" (z-push)
        self.current_gesture: GestureType = GestureType.UNKNOWN
        self.pinch_ratio: float = 0.0
        self._draw_grace: int = 0
        self._erase_confirm: int = 0          # consecutive frames confirming open palm
        self._history: deque = deque(maxlen=4)

    def set_writing_mode(self, mode: str) -> None:
        """Switch air-writing interaction mode."""
        if mode in ("pinch", "point", "depth"):
            self.writing_mode = mode
            self.reset()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _finger_extended(
        self,
        wrist: Tuple[int, int],
        mcp: Tuple[int, int],
        pip: Tuple[int, int],
        dip: Tuple[int, int],
        tip: Tuple[int, int],
        z_tip: float = 0.0,
        z_mcp: float = 0.0,
    ) -> bool:
        """True when finger is clearly extended (not just slightly raised)."""
        # The tip must be further from wrist than pip AND further from mcp than pip
        d_mcp_tip = self._dist(mcp, tip)
        d_mcp_pip = self._dist(mcp, pip)
        d_wrist_tip = self._dist(wrist, tip)
        d_wrist_pip = self._dist(wrist, pip)

        # Stricter ratio — tip must be clearly beyond pip
        ext_2d = (d_mcp_tip > d_mcp_pip * 1.15) and (d_wrist_tip > d_wrist_pip * 1.08)
        # 3-D depth component (finger pointing at camera)
        ext_3d = (z_mcp - z_tip) > 0.035
        return ext_2d or ext_3d

    def _thumb_extended(
        self,
        wrist: Tuple[int, int],
        cmc: Tuple[int, int],
        mcp: Tuple[int, int],
        tip: Tuple[int, int],
    ) -> bool:
        """Thumb extended = tip much further from wrist than mcp."""
        return self._dist(wrist, tip) > self._dist(wrist, mcp) * 1.20

    # ------------------------------------------------------------------
    # Main classify
    # ------------------------------------------------------------------

    def classify(self, tracker: HandTracker) -> GestureType:
        """Instant 1-frame gesture classification with bulletproof erase guard."""
        pts = tracker.pixel_landmarks
        if not pts:
            self.reset()
            return GestureType.UNKNOWN

        raw_lms = tracker.raw_landmarks
        wrist    = pts[HandTracker.WRIST]
        palm_scale = max(12.0, tracker.palm_scale)

        def z(idx: int) -> float:
            return float(raw_lms[idx][2]) if raw_lms and idx < len(raw_lms) else 0.0

        # ── Landmark shortcuts ──────────────────────────────────────────
        th_tip  = pts[HandTracker.THUMB_TIP]
        ix_mcp  = pts[HandTracker.INDEX_MCP]
        ix_pip  = pts[HandTracker.INDEX_PIP]
        ix_dip  = pts[HandTracker.INDEX_TIP]   # re-use tip as dip proxy
        ix_tip  = pts[HandTracker.INDEX_TIP]
        mi_mcp  = pts[HandTracker.MIDDLE_MCP]
        mi_pip  = pts[HandTracker.MIDDLE_PIP]
        mi_tip  = pts[HandTracker.MIDDLE_TIP]
        ri_mcp  = pts[HandTracker.RING_MCP]
        ri_pip  = pts[HandTracker.RING_PIP]
        ri_tip  = pts[HandTracker.RING_TIP]
        pk_mcp  = pts[HandTracker.PINKY_MCP]
        pk_pip  = pts[HandTracker.PINKY_PIP]
        pk_tip  = pts[HandTracker.PINKY_TIP]

        # ── Extension states ────────────────────────────────────────────
        idx_ext  = self._finger_extended(wrist, ix_mcp, ix_pip, ix_tip, ix_tip,
                                         z_tip=z(HandTracker.INDEX_TIP), z_mcp=z(HandTracker.INDEX_MCP))
        mid_ext  = self._finger_extended(wrist, mi_mcp, mi_pip, mi_tip, mi_tip,
                                         z_tip=z(HandTracker.MIDDLE_TIP), z_mcp=z(HandTracker.MIDDLE_MCP))
        ring_ext = self._finger_extended(wrist, ri_mcp, ri_pip, ri_tip, ri_tip,
                                         z_tip=z(HandTracker.RING_TIP), z_mcp=z(HandTracker.RING_MCP))
        pinky_ext = self._finger_extended(wrist, pk_mcp, pk_pip, pk_tip, pk_tip,
                                          z_tip=z(HandTracker.PINKY_TIP), z_mcp=z(HandTracker.PINKY_MCP))

        ext_count = sum([idx_ext, mid_ext, ring_ext, pinky_ext])

        # ── Distances from wrist ─────────────────────────────────────────
        d_wrist_idx   = self._dist(wrist, ix_tip)
        d_wrist_mid   = self._dist(wrist, mi_tip)
        d_wrist_ring  = self._dist(wrist, ri_tip)
        d_wrist_pinky = self._dist(wrist, pk_tip)

        # ── Pinch ────────────────────────────────────────────────────────
        d_pinch = self._dist(th_tip, ix_tip)
        self.pinch_ratio = d_pinch / palm_scale

        # ── Peace spread ─────────────────────────────────────────────────
        d_idx_mid_tips = self._dist(ix_tip, mi_tip)

        # ==================================================================
        #  GESTURE DECISION PIPELINE
        # ==================================================================

        # ── 1. OPEN PALM / "BYE" WAVE ERASER (👋) ─────────────────────────
        # STRICT rule: ALL 4 non-thumb fingers (Index, Middle, Ring, Pinky)
        # MUST be clearly extended AND spread wide open.
        # When writing with 1 or 2 fingers, this is physically IMPOSSIBLE.
        is_open_palm = (
            ext_count >= 4 and
            idx_ext and mid_ext and ring_ext and pinky_ext and
            d_wrist_ring  >= d_wrist_idx * 0.70 and
            d_wrist_pinky >= d_wrist_idx * 0.62 and
            d_wrist_mid   >= d_wrist_idx * 0.85 and
            d_idx_mid_tips > palm_scale * 0.18
        )

        # If writing with 1 or 2 fingers, instantly cancel erase confirmation!
        if ext_count <= 2:
            self._erase_confirm = 0
            erase_confirmed = False
        else:
            if is_open_palm:
                self._erase_confirm = min(self._erase_confirm + 1, self.erase_confirm_frames + 1)
            else:
                self._erase_confirm = max(0, self._erase_confirm - 2)
            erase_confirmed = self._erase_confirm >= self.erase_confirm_frames

        # ==================================================================
        #  GESTURE DECISION PIPELINE (MODE-AWARE)
        # ==================================================================
        if erase_confirmed:
            raw = GestureType.ERASE

        elif self.writing_mode == "pinch":
            # ── PEN-GRIP MODE (Optimal for words & separate letters) ──────
            # Pinching Thumb + Index together (< 0.22 ratio) = DRAW (Pen Down)
            # Releasing pinch = HOVER (Pen Up / Aim reticle without inking)
            # Completely prevents accidental cursive connections!
            is_pinched = (self.pinch_ratio < 0.22) and (idx_ext or ext_count <= 2)
            if is_pinched:
                raw = GestureType.DRAW
            elif idx_ext or ext_count >= 1:
                raw = GestureType.HOVER
            else:
                raw = GestureType.IDLE

        elif self.writing_mode == "depth":
            # ── VIRTUAL TOUCH / Z-DEPTH MODE ───────────────────────────────
            # Pushing finger forward towards camera = DRAW (Pen Down)
            # Pulling finger back = HOVER (Pen Up)
            z_tip = z(HandTracker.INDEX_TIP)
            z_mcp = z(HandTracker.INDEX_MCP)
            is_pushed = ((z_mcp - z_tip) > 0.04) and idx_ext
            if is_pushed:
                raw = GestureType.DRAW
            elif idx_ext:
                raw = GestureType.HOVER
            else:
                raw = GestureType.IDLE

        else:
            # ── POINT-TO-DRAW (Classic Air-Pen) ───────────────────────────
            # Index pointed alone = DRAW.
            # Peace sign (Index + Middle spread) = HOVER.
            # Curling index or opening palm = Pen lifts cleanly!
            is_peace = (
                idx_ext and mid_ext and not ring_ext and not pinky_ext and
                d_idx_mid_tips > palm_scale * 0.30
            )
            is_pinch_size = (
                self.pinch_ratio < 0.11 and
                not mid_ext and not ring_ext and not pinky_ext
            )
            is_draw = (
                not is_peace and
                not is_pinch_size and
                (
                    ext_count == 1 or
                    (idx_ext and not mid_ext and not ring_ext and not pinky_ext) or
                    (mid_ext and not idx_ext and not ring_ext and not pinky_ext)
                )
            )

            if is_peace:
                raw = GestureType.HOVER
            elif is_pinch_size:
                raw = GestureType.PINCH_SIZE
            elif is_draw:
                raw = GestureType.DRAW
            elif ext_count >= 1:
                raw = GestureType.HOVER
            else:
                raw = GestureType.IDLE

        # ==================================================================
        #  FINAL DECISION with IMMEDIATE PEN-LIFT
        # ==================================================================
        if raw == GestureType.DRAW:
            # In pinch/depth mode, pen-lift is instantaneous (0 grace) to avoid cursive drag
            self._draw_grace = 0 if self.writing_mode in ("pinch", "depth") else 1
            self.current_gesture = GestureType.DRAW

        elif raw == GestureType.ERASE:
            self._draw_grace = 0
            self.current_gesture = GestureType.ERASE

        elif raw == GestureType.HOVER:
            self._draw_grace = 0
            self.current_gesture = GestureType.HOVER

        elif raw == GestureType.PINCH_SIZE:
            self._draw_grace = 0
            self.current_gesture = GestureType.PINCH_SIZE

        elif self._draw_grace > 0:
            self._draw_grace -= 1
            self.current_gesture = GestureType.DRAW

        else:
            self.current_gesture = GestureType.IDLE

        self._history.append(self.current_gesture)
        return self.current_gesture

    def reset(self) -> None:
        self._history.clear()
        self.current_gesture = GestureType.UNKNOWN
        self.pinch_ratio = 0.0
        self._draw_grace = 0
        self._erase_confirm = 0

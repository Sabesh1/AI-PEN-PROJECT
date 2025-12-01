"""Air-Pen Studio: Main Application Entry Point.

Orchestrates real-time video capture, MediaPipe hand tracking, Casiez 1€ jitter filtering,
kinematic gesture classification, multi-layer canvas rendering, and the touchless HUD.
"""

from pathlib import Path
import sys
import time
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from config.settings import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    COLOR_PALETTE,
    MIRROR_FEED,
    CanvasMode,
    GestureMode,
)
from core.camera import ThreadedCamera
from core.finger_tracker import FingerTracker
from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.long_distance import AdaptiveParameters, LongDistanceOptimizer
from core.smoothing import OneEuroFilter2D
from engine.drawing_engine import DrawingEngine
from engine.exporter import Exporter
from ui.hud_overlay import HudOverlay
from ui.toolbar import TouchlessToolbar
from utils.profiler import PerformanceProfiler


class AirPenApp:
    """Master application controller for Air-Pen Studio."""

    def __init__(self):
        print("=" * 60)
        print("  AIR-PEN STUDIO: Touchless Computer Vision Drawing System")
        print("=" * 60)

        # Core subsystems
        self.camera = ThreadedCamera(
            src=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            mirror=MIRROR_FEED
        )
        self.hand_detector = HandDetector(max_hands=1)
        self.finger_tracker = FingerTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.smoothing_filter = OneEuroFilter2D(min_cutoff=1.2, beta=0.015)
        self.distance_optimizer = LongDistanceOptimizer()
        self.drawing_engine = DrawingEngine(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        self.exporter = Exporter()
        self.hud = HudOverlay(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        self.toolbar = TouchlessToolbar()
        self.profiler = PerformanceProfiler()

        self.is_eraser_selected = False
        self._notification_text: Optional[str] = None
        self._notification_expiry: float = 0.0

        self._setup_toolbar()

    def _show_notification(self, text: str, duration: float = 2.0) -> None:
        """Display an on-screen toast notification."""
        self._notification_text = text
        self._notification_expiry = time.perf_counter() + duration

    def _setup_toolbar(self) -> None:
        """Configure touchless interactive top toolbar."""
        btn_w, btn_h, btn_y = 68, 44, 18
        start_x = 40
        gap = 10

        # 1. Color swatches
        for color_key, pal in COLOR_PALETTE.items():
            self.toolbar.add_button(
                btn_id=color_key,
                label=pal.name[:4],
                rect=(start_x, btn_y, btn_w, btn_h),
                action=lambda k=color_key: self._select_color(k),
                icon_type="color",
                color_bgr=pal.bgr
            )
            start_x += btn_w + gap

        start_x += 16

        # 2. Eraser tool
        self.toolbar.add_button(
            btn_id="eraser",
            label="ERASER",
            rect=(start_x, btn_y, 76, btn_h),
            action=self._toggle_eraser
        )
        start_x += 76 + gap

        # 3. Canvas Mode Toggle
        self.toolbar.add_button(
            btn_id="canvas_mode",
            label="BOARD",
            rect=(start_x, btn_y, 76, btn_h),
            action=self._toggle_board
        )
        start_x += 76 + gap

        # 4. Undo
        self.toolbar.add_button(
            btn_id="undo",
            label="UNDO",
            rect=(start_x, btn_y, 68, btn_h),
            action=self._undo_action
        )
        start_x += 68 + gap

        # 5. Clear
        self.toolbar.add_button(
            btn_id="clear",
            label="CLEAR",
            rect=(start_x, btn_y, 68, btn_h),
            action=self._clear_action
        )
        start_x += 68 + gap

        # 6. Save / Export
        self.toolbar.add_button(
            btn_id="save",
            label="SAVE",
            rect=(start_x, btn_y, 72, btn_h),
            action=self._save_artwork
        )

    def _select_color(self, color_key: str) -> None:
        self.is_eraser_selected = False
        self.drawing_engine.set_color(color_key)
        self._show_notification(f"Brush: {COLOR_PALETTE[color_key].name}")

    def _toggle_eraser(self) -> None:
        self.is_eraser_selected = not self.is_eraser_selected
        status = "Enabled" if self.is_eraser_selected else "Disabled"
        self._show_notification(f"Eraser {status}")

    def _toggle_board(self) -> None:
        new_mode = self.drawing_engine.toggle_canvas_mode()
        self._show_notification(f"Canvas: {new_mode.value}")

    def _undo_action(self) -> None:
        if self.drawing_engine.undo():
            self._show_notification("Stroke Undone")

    def _clear_action(self) -> None:
        self.drawing_engine.clear()
        self._show_notification("Canvas Cleared")

    def _save_artwork(self) -> None:
        png_path = self.exporter.export_raster(self.drawing_engine, background="dark", ext="png")
        svg_path = self.exporter.export_svg(self.drawing_engine)
        self._show_notification(f"Saved PNG + SVG to {png_path.name}")

    def run(self) -> None:
        """Run main processing loop."""
        try:
            self.camera.start()
        except RuntimeError as e:
            print(f"[AirPenApp] Warning: Camera initialization failed: {e}")
            print("[AirPenApp] Falling back to synthetic virtual canvas mode.")
            self._run_fallback_loop()
            return

        window_name = "Air-Pen Studio"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CAMERA_WIDTH, CAMERA_HEIGHT)

        print("\n[AirPenApp] Air-Pen Studio running at 60 FPS!")
        print("  Controls:")
        print("  - Point Index finger: DRAW")
        print("  - Peace sign (Index+Middle): HOVER & SELECT toolbar buttons")
        print("  - 3 Fingers (Index+Middle+Ring): ERASER")
        print("  - Pinch (Index+Thumb): DYNAMIC BRUSH SIZE")
        print("  - Open Palm: IDLE / PAUSE")
        print("  Keyboard shortcuts: [U]ndo, [C]lear, [S]ave, [M]ode, [Q]uit\n")

        smoothed_pt: Optional[Tuple[float, float]] = None
        current_mode = GestureMode.UNKNOWN
        adaptive_params: Optional[AdaptiveParameters] = None

        while True:
            self.profiler.mark_frame_start()
            ret, frame = self.camera.read()

            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # 1. Vision & Landmark Inference
            t_inf_start = time.perf_counter()
            hand_result = self.hand_detector.process(frame)
            self.profiler.record_inference_latency(time.perf_counter() - t_inf_start)

            t_render_start = time.perf_counter()

            if hand_result is not None:
                # Calculate palm scale and adapt filtering
                palm_scale = self.finger_tracker.calculate_palm_scale(hand_result)
                adaptive_params = self.distance_optimizer.analyze(palm_scale)
                self.smoothing_filter.set_parameters(
                    adaptive_params.filter_min_cutoff,
                    adaptive_params.filter_beta
                )

                # Track index fingertip
                raw_index_tip = (
                    hand_result.landmarks[FingerTracker.INDEX_TIP].x,
                    hand_result.landmarks[FingerTracker.INDEX_TIP].y
                )
                smoothed_pt = self.smoothing_filter(raw_index_tip)

                # Classify hand gesture
                current_mode = self.gesture_recognizer.classify_gesture(hand_result, palm_scale)

                # Optional Hand Skeleton overlay (minimalist)
                self.hand_detector.draw_skeleton(frame, hand_result)

                # Handle gesture state machine
                if current_mode == GestureMode.PINCH_SIZE:
                    # Pinch distance dynamically adjusts brush radius
                    ratio = self.gesture_recognizer.pinch_distance_ratio
                    new_radius = int(max(2, min(45, ratio * 200)))
                    self.drawing_engine.set_brush_radius(new_radius)

                elif current_mode == GestureMode.HOVER:
                    self.drawing_engine.finish_stroke()
                    # Hit-test touchless buttons with distance-adaptive padding
                    self.toolbar.update(
                        smoothed_pt,
                        is_hover_mode=True,
                        padding=adaptive_params.ui_hitbox_padding
                    )

                elif current_mode == GestureMode.DRAW:
                    effective_eraser = self.is_eraser_selected
                    if self.drawing_engine.active_stroke is None:
                        self.drawing_engine.start_stroke(is_eraser=effective_eraser)
                    self.drawing_engine.add_point_to_active_stroke(smoothed_pt)

                elif current_mode == GestureMode.ERASE:
                    if self.drawing_engine.active_stroke is None or not self.drawing_engine.active_stroke.is_eraser:
                        self.drawing_engine.start_stroke(is_eraser=True)
                    self.drawing_engine.add_point_to_active_stroke(smoothed_pt)

                elif current_mode == GestureMode.IDLE:
                    self.drawing_engine.finish_stroke()
                    self.toolbar.update(None, is_hover_mode=False)

            else:
                # No hand visible in frame
                self.drawing_engine.finish_stroke()
                self.smoothing_filter.reset()
                self.gesture_recognizer.reset()
                self.toolbar.update(None, is_hover_mode=False)
                current_mode = GestureMode.UNKNOWN
                smoothed_pt = None

            # 2. Composite Drawing Layers
            composite = self.drawing_engine.composite(frame)

            # 3. Render In-Stream Glassmorphic HUD
            self.hud.render_toolbar(
                composite,
                self.toolbar,
                active_color_key=self.drawing_engine.current_color_key,
                is_eraser_active=self.is_eraser_selected,
                canvas_mode_str=self.drawing_engine.canvas_mode.value
            )

            self.hud.render_gesture_badge(
                composite,
                mode=current_mode,
                is_drawing=(self.drawing_engine.active_stroke is not None)
            )

            self.hud.render_telemetry_badge(
                composite,
                fps=self.profiler.fps,
                adaptive_params=adaptive_params
            )

            # Render pointer reticle if fingertip is tracked
            if smoothed_pt is not None:
                hovered_btn = self.toolbar.hit_test(smoothed_pt)
                dwell_prog = hovered_btn.dwell_progress if hovered_btn else 0.0
                active_radius = (
                    self.drawing_engine.current_eraser_radius
                    if (self.is_eraser_selected or current_mode == GestureMode.ERASE)
                    else self.drawing_engine.current_brush_radius
                )
                self.hud.render_pointer(
                    composite,
                    pointer=smoothed_pt,
                    mode=current_mode,
                    brush_radius=active_radius,
                    color_bgr=self.drawing_engine.current_color_bgr,
                    dwell_progress=dwell_prog
                )

            # 4. On-screen Toast Notification
            now = time.perf_counter()
            if self._notification_text and now < self._notification_expiry:
                self._draw_toast(composite, self._notification_text)

            self.profiler.record_render_latency(time.perf_counter() - t_render_start)

            cv2.imshow(window_name, composite)

            # 5. Handle Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('u'), ord('U')):
                self._undo_action()
            elif key in (ord('c'), ord('C')):
                self._clear_action()
            elif key in (ord('s'), ord('S')):
                self._save_artwork()
            elif key in (ord('m'), ord('M')):
                self._toggle_board()
            elif key in (ord('e'), ord('E')):
                self._toggle_eraser()

        self._cleanup()

    def _draw_toast(self, frame: np.ndarray, text: str) -> None:
        """Render floating notification message centered at top."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cx = CAMERA_WIDTH // 2
        cy = 90
        pad = 14
        cv2.rectangle(
            frame,
            (cx - tw // 2 - pad, cy - th - pad),
            (cx + tw // 2 + pad, cy + pad),
            (30, 30, 38),
            -1
        )
        cv2.rectangle(
            frame,
            (cx - tw // 2 - pad, cy - th - pad),
            (cx + tw // 2 + pad, cy + pad),
            (0, 255, 200),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            text,
            (cx - tw // 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    def _run_fallback_loop(self) -> None:
        """Run synthetic fallback mode when physical camera is unavailable."""
        window_name = "Air-Pen Studio (Synthetic Demo Mode)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CAMERA_WIDTH, CAMERA_HEIGHT)

        self.drawing_engine.canvas_mode = CanvasMode.BLACKBOARD
        t = 0.0

        print("[AirPenApp] Simulating animated demo strokes on virtual canvas...")

        while True:
            # Generate moving parametric Lissajous demo stroke
            t += 0.03
            px = int(CAMERA_WIDTH // 2 + 300 * np.sin(t))
            py = int(CAMERA_HEIGHT // 2 + 180 * np.sin(2 * t))

            self.drawing_engine.add_point_to_active_stroke((float(px), float(py)))

            base_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            composite = self.drawing_engine.composite(base_frame)

            self.hud.render_toolbar(
                composite,
                self.toolbar,
                active_color_key=self.drawing_engine.current_color_key,
                is_eraser_active=False,
                canvas_mode_str="Synthetic Demo"
            )

            self.hud.render_gesture_badge(composite, GestureMode.DRAW, True)
            self.hud.render_telemetry_badge(composite, 60.0, None)

            cv2.imshow(window_name, composite)
            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

        cv2.destroyAllWindows()

    def _cleanup(self) -> None:
        """Release all hardware and visual window handles."""
        self.camera.stop()
        self.hand_detector.close()
        cv2.destroyAllWindows()
        print("[AirPenApp] Application cleanly exited.")


if __name__ == "__main__":
    app = AirPenApp()
    app.run()

"""AIR_PEN: Main Application Orchestrator.

Integrates camera video capture, MediaPipe hand tracking, Casiez 1€ jitter smoothing,
kinematic gesture detection, vector drawing engine, and touchless glassmorphic GUI.
"""

from pathlib import Path
import sys
import time
from typing import Optional, Tuple

# Ensure AIR_PEN directory is on sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import cv2
import numpy as np

from backend.camera import ThreadedCamera
from backend.drawing import DrawingEngine
from backend.gestures import GestureDetector, GestureType
from backend.hand_tracking import HandTracker
from frontend.gui import AirPenGUI


class AirPenApp:
    """Master controller connecting backend vision pipeline and frontend GUI."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        print("=" * 60)
        print("  AIR-PEN: Touchless Computer Vision Drawing System")
        print("=" * 60)

        output_path = APP_DIR / "output"
        output_path.mkdir(parents=True, exist_ok=True)

        self.camera = ThreadedCamera(src=0, width=width, height=height, fps=60, mirror=True)
        self.tracker = HandTracker(max_hands=1)
        self.gesture_detector = GestureDetector(hysteresis_frames=2)
        self.drawing_engine = DrawingEngine(width=width, height=height, output_dir=str(output_path))
        self.gui = AirPenGUI(width=width, height=height)

        self.gui.setup_toolbar(self.drawing_engine, on_save=self._save_artwork)
        self.fps: float = 60.0
        self._last_frame_ts = time.perf_counter()

    def _save_artwork(self) -> None:
        png, svg = self.drawing_engine.export_all()
        self.gui.show_toast(f"Saved: {png.name}")

    def run(self) -> None:
        """Main real-time application loop."""
        try:
            self.camera.start()
        except RuntimeError as e:
            print(f"[AIR_PEN] Camera unavailable ({e}). Starting synthetic demo mode...")
            self._run_fallback()
            return

        win = "AIR-PEN Studio"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.width, self.height)

        print("\n[AIR_PEN] Running at 60 FPS.")
        print("  Gestures:")
        print("  - Index finger point:  DRAW")
        print("  - Peace sign (Index+Mid): HOVER / SELECT buttons")
        print("  - 3 Fingers up:       ERASER")
        print("  - Pinch (Thumb+Index): RESIZE BRUSH")
        print("  - Open palm:          IDLE / PAUSE")
        print("  Shortcuts: [U]ndo, [C]lear, [S]ave, [M]ode, [Q]uit\n")

        last_fid = -1
        while True:
            ret, frame, fid = self.camera.read(last_fid)
            if not ret or frame is None:
                time.sleep(0.003)
                continue
            last_fid = fid

            t_now = time.perf_counter()
            dt = t_now - self._last_frame_ts
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self._last_frame_ts = t_now

            # 1. Vision & Hand Tracking
            has_hand = self.tracker.process(frame)
            current_gesture = GestureType.UNKNOWN
            pointer: Optional[Tuple[float, float]] = None

            if has_hand:
                pointer = self.tracker.get_index_tip()
                current_gesture = self.gesture_detector.classify(self.tracker)
                self.tracker.draw_skeleton(frame)

                # 2. Gesture Actions
                if current_gesture == GestureType.PINCH_SIZE:
                    ratio = self.gesture_detector.pinch_ratio
                    new_r = int(max(2, min(40, ratio * 150)))
                    self.drawing_engine.brush_radius = new_r

                elif current_gesture == GestureType.HOVER:
                    self.drawing_engine.finish_stroke()
                    self.gui.update_touchless(pointer, is_hover_mode=True)

                elif current_gesture == GestureType.DRAW:
                    if pointer is not None:
                        self.drawing_engine.add_point(pointer)

                elif current_gesture == GestureType.ERASE:
                    if pointer is not None:
                        if self.drawing_engine.active_stroke is None or not self.drawing_engine.active_stroke.is_eraser:
                            self.drawing_engine.start_stroke(is_eraser=True)
                        self.drawing_engine.add_point(pointer)

                elif current_gesture == GestureType.IDLE:
                    self.drawing_engine.finish_stroke()
                    self.gui.update_touchless(None, is_hover_mode=False)

            else:
                self.hand_lost_frames = getattr(self, "hand_lost_frames", 0) + 1
                if self.hand_lost_frames > 4:
                    self.drawing_engine.finish_stroke()
                    self.gesture_detector.reset()
                    self.gui.update_touchless(None, is_hover_mode=False)

            # 3. Canvas Compositing
            composite = self.drawing_engine.composite(frame)

            # 4. GUI & HUD Rendering
            self.gui.render(
                composite,
                engine=self.drawing_engine,
                gesture=current_gesture,
                pointer=pointer,
                fps=self.fps,
                palm_scale=self.tracker.palm_scale
            )

            cv2.imshow(win, composite)

            # 5. Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('u'), ord('U')):
                if self.drawing_engine.undo():
                    self.gui.show_toast("Undone")
            elif key in (ord('c'), ord('C')):
                self.drawing_engine.clear()
                self.gui.show_toast("Cleared")
            elif key in (ord('s'), ord('S')):
                self._save_artwork()
            elif key in (ord('m'), ord('M')):
                mode = self.drawing_engine.toggle_canvas_mode()
                self.gui.show_toast(f"Canvas: {mode.capitalize()}")

        self._cleanup()

    def _run_fallback(self) -> None:
        """Synthetic fallback canvas when physical camera is absent."""
        win = "AIR-PEN Studio (Demo Mode)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.width, self.height)

        self.drawing_engine.canvas_mode = "blackboard"
        t = 0.0

        while True:
            t += 0.04
            px = int(self.width // 2 + 250 * np.cos(t))
            py = int(self.height // 2 + 150 * np.sin(2 * t))

            self.drawing_engine.add_point((float(px), float(py)))

            base = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            composite = self.drawing_engine.composite(base)

            self.gui.render(
                composite,
                engine=self.drawing_engine,
                gesture=GestureType.DRAW,
                pointer=(float(px), float(py)),
                fps=60.0,
                palm_scale=100.0
            )

            cv2.imshow(win, composite)
            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

        cv2.destroyAllWindows()

    def _cleanup(self) -> None:
        self.camera.stop()
        self.tracker.close()
        cv2.destroyAllWindows()
        print("[AIR_PEN] Exited cleanly.")


if __name__ == "__main__":
    app = AirPenApp()
    app.run()

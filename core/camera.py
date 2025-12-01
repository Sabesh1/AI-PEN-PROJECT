"""Low-latency threaded camera capture module.

Standard cv2.VideoCapture buffers frames internally in an OS queue.
If frame processing takes even slightly longer than camera exposure time,
the buffer fills up, creating progressive latency (input lag).
This module runs a background thread that continuously flushes the buffer,
ensuring the application always processes the freshest available frame.
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class ThreadedCamera:
    """Zero-lag threaded camera capture wrapper."""

    def __init__(
        self,
        src: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        mirror: bool = True
    ):
        self.src = src
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.mirror = mirror

        self.cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps_actual: float = 0.0

    def start(self) -> "ThreadedCamera":
        """Initialize camera device and start worker thread."""
        # On Windows, cv2.CAP_DSHOW provides fast, low-latency device initialization
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to default backend if DirectShow is unavailable
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {self.src}")

        # Request desired resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read actual camera properties
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[ThreadedCamera] Initialized: {actual_w}x{actual_h} @ {self.target_fps} FPS (mirror={self.mirror})")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
        self._thread.start()

        # Wait for first valid frame
        start_wait = time.time()
        while self._latest_frame is None and (time.time() - start_wait) < 3.0:
            time.sleep(0.01)

        return self

    def _capture_loop(self) -> None:
        """Background loop continuously grabbing frames to keep buffer empty."""
        frame_count = 0
        t_start = time.perf_counter()

        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            with self._lock:
                self._latest_frame = frame

            frame_count += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                self._fps_actual = frame_count / elapsed
                frame_count = 0
                t_start = time.perf_counter()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieve the most recent frame."""
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    @property
    def fps(self) -> float:
        """Actual hardware capture frame rate."""
        return self._fps_actual

    def stop(self) -> None:
        """Stop worker thread and release hardware resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        print("[ThreadedCamera] Released camera resources.")

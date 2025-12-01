"""Threaded camera module for zero-lag video capture."""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class ThreadedCamera:
    """High-performance threaded camera capture to eliminate OS buffer delay."""

    def __init__(
        self,
        src: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        mirror: bool = True
    ):
        self.src = src
        self.width = width
        self.height = height
        self.target_fps = fps
        self.mirror = mirror

        self.cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.fps_actual: float = 0.0
        self.frame_id: int = 0

    def start(self) -> "ThreadedCamera":
        """Start background capture thread."""
        # Try DirectShow first on Windows for instant initialization
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {self.src}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

        # Wait for first frame
        start_wait = time.time()
        while self._latest_frame is None and (time.time() - start_wait) < 2.0:
            time.sleep(0.01)

        return self

    def _capture_worker(self) -> None:
        """Background thread continuously pulling frames to empty buffer."""
        frames = 0
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
                self.frame_id += 1

            frames += 1
            now = time.perf_counter()
            if (now - t_start) >= 1.0:
                self.fps_actual = frames / (now - t_start)
                frames = 0
                t_start = now

    def read(self, last_id: int = -1) -> Tuple[bool, Optional[np.ndarray], int]:
        """Return the freshest frame and its unique sequence ID."""
        with self._lock:
            if self._latest_frame is None:
                return False, None, self.frame_id
            if self.frame_id == last_id:
                return False, None, self.frame_id
            return True, self._latest_frame, self.frame_id

    def stop(self) -> None:
        """Stop worker and release camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

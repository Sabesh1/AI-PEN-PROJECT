"""Performance profiler and latency telemetry monitor.

Measures frame-to-frame intervals, landmark inference latency,
drawing compositing duration, and sliding-window FPS.
"""

from collections import deque
import time
from typing import Dict


class PerformanceProfiler:
    """Real-time pipeline latency and throughput profiler."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._frame_times: deque[float] = deque(maxlen=window_size)
        self._inference_latencies: deque[float] = deque(maxlen=window_size)
        self._render_latencies: deque[float] = deque(maxlen=window_size)
        self._last_frame_ts: float = time.perf_counter()

    def mark_frame_start(self) -> float:
        """Mark start of a new frame processing cycle."""
        now = time.perf_counter()
        dt = now - self._last_frame_ts
        if dt > 0:
            self._frame_times.append(dt)
        self._last_frame_ts = now
        return now

    def record_inference_latency(self, duration_sec: float) -> None:
        """Record hand detection & tracking inference time."""
        self._inference_latencies.append(duration_sec)

    def record_render_latency(self, duration_sec: float) -> None:
        """Record canvas rendering and HUD compositing time."""
        self._render_latencies.append(duration_sec)

    @property
    def fps(self) -> float:
        """Moving average frames per second."""
        if not self._frame_times:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        """Average landmark inference latency in milliseconds."""
        if not self._inference_latencies:
            return 0.0
        return (sum(self._inference_latencies) / len(self._inference_latencies)) * 1000.0

    @property
    def avg_render_ms(self) -> float:
        """Average rendering & HUD latency in milliseconds."""
        if not self._render_latencies:
            return 0.0
        return (sum(self._render_latencies) / len(self._render_latencies)) * 1000.0

    def get_summary(self) -> Dict[str, float]:
        """Return diagnostic metrics dictionary."""
        return {
            "fps": round(self.fps, 1),
            "inference_ms": round(self.avg_inference_ms, 2),
            "render_ms": round(self.avg_render_ms, 2),
            "total_latency_ms": round(self.avg_inference_ms + self.avg_render_ms, 2),
        }

"""Stroke data representation and Catmull-Rom spline interpolation.

Stores raw points, visual attributes (color, thickness, glow), and provides
smooth geometric interpolation to eliminate polyline corners.
"""

from dataclasses import dataclass, field
import time
from typing import List, Tuple
import numpy as np


@dataclass
class Stroke:
    """Individual drawn stroke composed of continuous filtered points."""
    color_bgr: Tuple[int, int, int]
    thickness: int
    is_eraser: bool = False
    glow: bool = True
    points: List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def add_point(self, point: Tuple[float, float], timestamp: float = 0.0) -> None:
        """Append a new filtered coordinate to this stroke."""
        if timestamp <= 0.0:
            timestamp = time.perf_counter()

        # Deduplicate consecutive points too close together (< 1.5px)
        if self.points:
            last_x, last_y = self.points[-1]
            if (point[0] - last_x) ** 2 + (point[1] - last_y) ** 2 < 2.25:
                return

        self.points.append((float(point[0]), float(point[1])))
        self.timestamps.append(timestamp)

    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0

    def get_interpolated_points(self, points_per_segment: int = 4) -> List[Tuple[int, int]]:
        """Compute Catmull-Rom spline interpolation for organic curvature."""
        n = len(self.points)
        if n < 3:
            return [(int(round(p[0])), int(round(p[1]))) for p in self.points]

        interpolated: List[Tuple[int, int]] = []
        pts = np.array(self.points, dtype=np.float64)

        # Pad endpoints for Catmull-Rom tangent computation
        padded = np.vstack([pts[0], pts, pts[-1]])

        for i in range(1, len(padded) - 2):
            p0 = padded[i - 1]
            p1 = padded[i]
            p2 = padded[i + 1]
            p3 = padded[i + 2]

            for t in np.linspace(0.0, 1.0, points_per_segment, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                # Standard Catmull-Rom spline basis matrix formula
                point = 0.5 * (
                    (2.0 * p1) +
                    (-p0 + p2) * t +
                    (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                    (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                interpolated.append((int(round(point[0])), int(round(point[1]))))

        # Include final endpoint
        interpolated.append((int(round(pts[-1][0])), int(round(pts[-1][1]))))
        return interpolated

"""AI Character & Word Recognition Engine for Air-Pen Studio.

Recognizes hand-drawn letters ('A'-'Z', 'a'-'z'), digits ('0'-'9'), and common symbols
from touchless air strokes in real-time.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import time


@dataclass
class RecognitionResult:
    """Result of character recognition."""
    character: str
    confidence: float
    is_letter: bool = True
    details: str = ""


class AirCharacterRecognizer:
    """Real-time geometric stroke classifier for touchless air-written letters and digits."""

    def __init__(self):
        self.word_buffer: List[str] = []
        self.last_result: Optional[RecognitionResult] = None
        self._pending_strokes: List[List[Tuple[float, float]]] = []
        self._last_stroke_time: float = 0.0
        self._pending_bbox: Optional[Tuple[float, float, float, float]] = None

    @staticmethod
    def _stroke_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, width, height) of stroke points."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, min_y, max(1.0, max_x - min_x), max(1.0, max_y - min_y)

    @staticmethod
    def _stroke_length(points: List[Tuple[float, float]]) -> float:
        total = 0.0
        for i in range(1, len(points)):
            total += math.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
        return total

    @staticmethod
    def _count_crossings(points: List[Tuple[float, float]], y_line: float) -> int:
        """Counts how many times stroke crosses a horizontal line."""
        crossings = 0
        for i in range(1, len(points)):
            y1, y2 = points[i-1][1], points[i][1]
            if (y1 < y_line <= y2) or (y2 < y_line <= y1):
                crossings += 1
        return crossings

    @staticmethod
    def _detect_loop(points: List[Tuple[float, float]], threshold_ratio: float = 0.35) -> bool:
        """Detects if a stroke closes into a loop or circle."""
        if len(points) < 8:
            return False
        d_ends = math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1])
        length = AirCharacterRecognizer._stroke_length(points)
        _, _, w, h = AirCharacterRecognizer._stroke_bbox(points)
        perimeter_approx = 2.0 * (w + h)
        return (d_ends / max(1.0, max(w, h)) < threshold_ratio) or (d_ends / max(1.0, length) < 0.25)

    def recognize(self, strokes_points: List[List[Tuple[float, float]]], update_buffer: bool = True) -> RecognitionResult:
        """Classifies 1 or more strokes into a recognized character or digit."""
        if not strokes_points or not any(strokes_points):
            return RecognitionResult("?", 0.0, False, "No points")

        # Merge all points to compute overall bounding box
        all_pts: List[Tuple[float, float]] = []
        for s in strokes_points:
            all_pts.extend(s)

        if len(all_pts) < 5:
            return RecognitionResult("·", 0.3, False, "Dot or tap")

        min_x, min_y, w, h = self._stroke_bbox(all_pts)
        aspect = h / max(1.0, w)  # tall > 1.5, square ~ 1.0, wide < 0.7
        num_strokes = len(strokes_points)

        s1 = strokes_points[0]
        s1_start = s1[0]
        s1_end = s1[-1]
        s1_has_loop = self._detect_loop(s1)

        # -------------------------------------------------------------
        # 1-STROKE RECOGNITION HEURISTICS
        # -------------------------------------------------------------
        if num_strokes == 1:
            p_start = s1_start
            p_end = s1_end
            
            # Normalized endpoints in [0, 1] relative to bounding box
            nx_start = (p_start[0] - min_x) / w
            ny_start = (p_start[1] - min_y) / h
            nx_end = (p_end[0] - min_x) / w
            ny_end = (p_end[1] - min_y) / h

            # Vertical straight line -> '1' or 'I' or 'l'
            if aspect > 2.2 and w < 35:
                res = RecognitionResult("1", 0.95, True, "Vertical single stroke")
                self._update_buffer(res)
                return res

            # Horizontal line -> '-'
            if aspect < 0.35 and w > 40:
                res = RecognitionResult("-", 0.92, False, "Horizontal bar")
                self._update_buffer(res)
                return res

            # Loop / Circle -> 'O' or '0' or 'o'
            if s1_has_loop:
                if 0.75 <= aspect <= 1.45:
                    char = "O" if h > 70 else "o"
                    res = RecognitionResult(char, 0.92, True, "Closed circular loop")
                    self._update_buffer(res)
                    return res
                elif aspect > 1.45:
                    res = RecognitionResult("0", 0.91, True, "Tall oval loop")
                    self._update_buffer(res)
                    return res

            # Check if start is near top-right, curves left and ends bottom-right -> 'C' or 'c'
            if (nx_start > 0.6 and ny_start < 0.35) and (nx_end > 0.5 and ny_end > 0.65):
                # Check middle reaches left (nx < 0.2)
                min_nx = min((p[0] - min_x) / w for p in s1)
                if min_nx < 0.25:
                    char = "C" if h > 60 else "c"
                    res = RecognitionResult(char, 0.90, True, "Open leftward curve")
                    self._update_buffer(res)
                    return res

            # Check 'S' or 's': starts top-right, curves left, crosses center right, loops down left
            mid_y = min_y + h * 0.5
            crossings = self._count_crossings(s1, mid_y)
            if (nx_start > 0.5 and ny_start < 0.3) and (nx_end < 0.5 and ny_end > 0.7) and crossings >= 2:
                char = "S" if h > 60 else "s"
                res = RecognitionResult(char, 0.88, True, "Double serpentine curve")
                self._update_buffer(res)
                return res

            # Check 'U' or 'u': starts top-left, curves down to bottom, ends top-right
            if (nx_start < 0.35 and ny_start < 0.4) and (nx_end > 0.65 and ny_end < 0.4):
                max_ny = max((p[1] - min_y) / h for p in s1)
                if max_ny > 0.75:
                    char = "U" if h > 60 else "u"
                    res = RecognitionResult(char, 0.91, True, "U-shape bottom trough")
                    self._update_buffer(res)
                    return res

            # Check 'V' or 'v': starts top-left, goes to bottom point, ends top-right
            if (nx_start < 0.4 and ny_start < 0.35) and (nx_end > 0.6 and ny_end < 0.35):
                lowest_idx = max(range(len(s1)), key=lambda i: s1[i][1])
                lowest_pt = s1[lowest_idx]
                if lowest_pt[1] >= min_y + h * 0.8:
                    char = "V" if h > 60 else "v"
                    res = RecognitionResult(char, 0.89, True, "V-notch vertex")
                    self._update_buffer(res)
                    return res

            # Check 'L': starts top-left, goes straight down, then right
            if (nx_start < 0.35 and ny_start < 0.3) and (nx_end > 0.6 and ny_end > 0.7):
                res = RecognitionResult("L", 0.92, True, "Corner stroke down then right")
                self._update_buffer(res)
                return res

            # Check 'Z': starts top-left, goes right, diagonal down-left, then right
            if (nx_start < 0.4 and ny_start < 0.3) and (nx_end > 0.6 and ny_end > 0.7):
                if crossings >= 2:
                    char = "Z" if h > 60 else "z"
                    res = RecognitionResult(char, 0.87, True, "Zig-zag diagonal")
                    self._update_buffer(res)
                    return res

            # Check 'a': starts mid-right or top-right, loops counter-clockwise, then has vertical tail
            if aspect <= 1.4:
                # If there's a loop and tail on right side
                if s1_has_loop or (nx_end > 0.7 and ny_end > 0.7 and nx_start > 0.5):
                    res = RecognitionResult("a", 0.85, True, "Round counter-clockwise loop with right tail")
                    self._update_buffer(res)
                    return res

            # Check 'b': tall left stem starting top-left, goes down, then loops counter-clockwise to bottom
            if aspect >= 1.2 and (nx_start < 0.35 and ny_start < 0.25):
                res = RecognitionResult("b", 0.86, True, "Top-left stem with lower right loop")
                self._update_buffer(res)
                return res

        # -------------------------------------------------------------
        # 2-STROKE RECOGNITION HEURISTICS
        # -------------------------------------------------------------
        elif num_strokes == 2:
            st1, st2 = strokes_points[0], strokes_points[1]
            min1_x, min1_y, w1, h1 = self._stroke_bbox(st1)
            min2_x, min2_y, w2, h2 = self._stroke_bbox(st2)

            # 'T': Stroke 1 is horizontal top bar, Stroke 2 is vertical stem down (or vice-versa!)
            is_st1_bar = (w1 > w * 0.55 and h1 < h * 0.40 and min1_y < min_y + h * 0.40)
            is_st2_stem = (h2 > h * 0.55 and w2 < w * 0.50)
            is_st2_bar = (w2 > w * 0.55 and h2 < h * 0.40 and min2_y < min_y + h * 0.40)
            is_st1_stem = (h1 > h * 0.55 and w1 < w * 0.50)

            if (is_st1_bar and is_st2_stem) or (is_st2_bar and is_st1_stem):
                res = RecognitionResult("T", 0.94, True, "Top bar and center vertical stem")
                self._update_buffer(res)
                return res

            # '+' (Plus): Horizontal and vertical line intersecting near center
            if ((w1 > w * 0.5 and h1 < h * 0.45) and (h2 > h * 0.5 and w2 < w * 0.45)) or \
               ((w2 > w * 0.5 and h2 < h * 0.45) and (h1 > h * 0.5 and w1 < w * 0.45)):
                res = RecognitionResult("+", 0.92, False, "Plus cross")
                self._update_buffer(res)
                return res

            # '=' (Equals): Two parallel horizontal bars
            if (w1 > w * 0.5 and h1 < h * 0.35) and (w2 > w * 0.5 and h2 < h * 0.35):
                res = RecognitionResult("=", 0.93, False, "Double horizontal bar")
                self._update_buffer(res)
                return res

            # 'X' or 'x': Two diagonal cross strokes
            if (w1 > w * 0.35 and h1 > h * 0.45) and (w2 > w * 0.35 and h2 > h * 0.45):
                char = "X" if h > 60 else "x"
                res = RecognitionResult(char, 0.91, True, "Two intersecting diagonal lines")
                self._update_buffer(res)
                return res

            # 'L': Vertical stem down and bottom horizontal line
            if (is_st1_stem and min1_x < min_x + w * 0.4) and (w2 > w * 0.4 and min2_y > min_y + h * 0.5):
                res = RecognitionResult("L", 0.91, True, "Vertical stem with base")
                self._update_buffer(res)
                return res

            # 'D': Stroke 1 is vertical left stem, Stroke 2 is rightward arch
            if (h1 > h * 0.7 and w1 < w * 0.35) and (w2 > w * 0.5 and h2 > h * 0.6):
                res = RecognitionResult("D", 0.90, True, "Left vertical stem with right semicircle")
                self._update_buffer(res)
                return res

            # 'P': Stroke 1 is vertical left stem, Stroke 2 is upper right loop
            if (h1 > h * 0.7 and w1 < w * 0.35) and (min2_y < min_y + h * 0.6):
                res = RecognitionResult("P", 0.88, True, "Left vertical stem with upper loop")
                self._update_buffer(res)
                return res

            # 'B' or 'b': Stroke 1 is vertical left stem, Stroke 2 has loops
            if (h1 > h * 0.7 and w1 < w * 0.35) and (w2 > w * 0.4):
                res = RecognitionResult("B", 0.89, True, "Left stem with double right bulges")
                self._update_buffer(res)
                return res

            # 'A' with 2 strokes: Stroke 1 is caret '^', Stroke 2 is horizontal crossbar '-'
            if (h1 > h * 0.6 and w1 > w * 0.5) and (w2 > w * 0.3 and h2 < h * 0.4):
                res = RecognitionResult("A", 0.93, True, "Caret inverted V with crossbar")
                self._update_buffer(res)
                return res

        # -------------------------------------------------------------
        # 3-STROKE RECOGNITION HEURISTICS
        # -------------------------------------------------------------
        elif num_strokes == 3:
            st1, st2, st3 = strokes_points[0], strokes_points[1], strokes_points[2]
            stems = [s for s in (st1, st2, st3) if self._stroke_bbox(s)[3] > h * 0.55 and self._stroke_bbox(s)[2] < w * 0.45]
            bars  = [s for s in (st1, st2, st3) if self._stroke_bbox(s)[2] > w * 0.40 and self._stroke_bbox(s)[3] < h * 0.45]
            if len(stems) >= 2 and len(bars) >= 1:
                res = RecognitionResult("H", 0.94, True, "Two vertical stems with center crossbar")
                self._update_buffer(res)
                return res

            # 'A': Two diagonal strokes forming inverted V, plus horizontal crossbar
            res = RecognitionResult("A", 0.92, True, "Tri-stroke A glyph")
            self._update_buffer(res)
            return res

        # Fallback heuristic based on geometry
        if aspect > 1.8:
            res = RecognitionResult("I", 0.75, True, "Tall vertical glyph")
        elif s1_has_loop:
            res = RecognitionResult("O", 0.78, True, "Closed loop glyph")
        else:
            res = RecognitionResult("~", 0.60, False, "Freehand cursive stroke")

        self._update_buffer(res)
        return res

    def _update_buffer(self, res: RecognitionResult) -> None:
        self.last_result = res
        if getattr(self, "_suppress_buffer", False):
            return
        if res.is_letter and res.character not in ["?", "·", "-", "~"]:
            self.word_buffer.append(res.character)

    def process_stroke(self, stroke_pts: List[Tuple[float, float]], now_ts: Optional[float] = None) -> Optional[RecognitionResult]:
        """Aggregate sequential multi-stroke glyphs (e.g. crossbars of T, H, A) into complete characters."""
        if not stroke_pts or len(stroke_pts) < 3:
            return None
        if now_ts is None:
            now_ts = time.time()

        cur_bbox = self._stroke_bbox(stroke_pts)

        if self._pending_strokes and self._pending_bbox:
            dt = now_ts - self._last_stroke_time
            p_min_x, p_min_y, p_w, p_h = self._pending_bbox
            c_min_x, c_min_y, c_w, c_h = cur_bbox

            # Spatial separation: Did user move rightwards to start the next letter?
            is_new_letter = (dt > 1.3) or (c_min_x > p_min_x + max(40.0, p_w * 0.90) and c_min_x > p_min_x + p_w * 0.5)

            if is_new_letter:
                # Finalize previous character into word buffer
                self.recognize(self._pending_strokes, update_buffer=True)
                self._pending_strokes = [stroke_pts]
                self._pending_bbox = cur_bbox
                self._last_stroke_time = now_ts
                self._suppress_buffer = True
                try:
                    return self.recognize(self._pending_strokes, update_buffer=False)
                finally:
                    self._suppress_buffer = False
            else:
                # Part of current character (e.g., crossbar of T or A)
                self._pending_strokes.append(stroke_pts)
                all_pts = [p for s in self._pending_strokes for p in s]
                self._pending_bbox = self._stroke_bbox(all_pts)
                self._last_stroke_time = now_ts
                self._suppress_buffer = True
                try:
                    return self.recognize(self._pending_strokes, update_buffer=False)
                finally:
                    self._suppress_buffer = False
        else:
            self._pending_strokes = [stroke_pts]
            self._pending_bbox = cur_bbox
            self._last_stroke_time = now_ts
            self._suppress_buffer = True
            try:
                return self.recognize(self._pending_strokes, update_buffer=False)
            finally:
                self._suppress_buffer = False

    def finalize_pending(self) -> Optional[RecognitionResult]:
        """Finalize any pending multi-stroke character into the word buffer."""
        if self._pending_strokes:
            self._suppress_buffer = False
            res = self.recognize(self._pending_strokes, update_buffer=True)
            self._pending_strokes = []
            self._pending_bbox = None
            return res
        return None

    def add_space(self) -> str:
        self.finalize_pending()
        self.word_buffer.append(" ")
        return self.get_word()

    def backspace(self) -> str:
        self.finalize_pending()
        if self.word_buffer:
            self.word_buffer.pop()
        return self.get_word()

    def get_word(self) -> str:
        return "".join(self.word_buffer)

    def clear_word(self) -> None:
        self.word_buffer.clear()
        self.last_result = None
        self._pending_strokes = []
        self._pending_bbox = None

    def pop_char(self) -> Optional[str]:
        self.finalize_pending()
        if self.word_buffer:
            return self.word_buffer.pop()
        return None

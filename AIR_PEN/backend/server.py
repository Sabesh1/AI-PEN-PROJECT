"""AIR_PEN Web Server: Serves live browser UI and MJPEG video stream on localhost:8000."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Ensure AIR_PEN directory is on sys.path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import cv2
import numpy as np

from backend.camera import ThreadedCamera
from backend.drawing import DrawingEngine
from backend.gestures import GestureDetector, GestureType
from backend.hand_tracking import HandTracker

WEB_DIR = APP_DIR / "frontend" / "web"
OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class WebAirPenService:
    """Manages background camera processing and frame broadcasting."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        self.camera = ThreadedCamera(src=0, width=width, height=height, fps=60, mirror=True)
        self.tracker = HandTracker(max_hands=1)
        self.gesture_detector = GestureDetector(hysteresis_frames=2, writing_mode="pinch")
        self.drawing_engine = DrawingEngine(width=width, height=height, output_dir=str(OUTPUT_DIR))

        self.fps: float = 60.0
        self.current_gesture: GestureType = GestureType.UNKNOWN
        self.distance_zone: str = "Detecting..."
        self._latest_jpeg: bytes = b""
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self.hand_lost_frames = 0

    def start(self) -> None:
        try:
            self.camera.start()
        except RuntimeError as e:
            print(f"[Server] Physical camera failed ({e}). Running virtual canvas.")

        self._running = True
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()

    def _process_loop(self) -> None:
        t_last = time.perf_counter()
        last_fid = -1

        while self._running:
            ret, frame, fid = self.camera.read(last_fid)
            if not ret or frame is None:
                if self.camera.cap is None or not self.camera.cap.isOpened():
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    time.sleep(0.016)
                else:
                    time.sleep(0.003)
                    continue
            else:
                last_fid = fid

            t_now = time.perf_counter()
            dt = t_now - t_last
            if dt > 0:
                self.fps = 0.90 * self.fps + 0.10 * (1.0 / dt)
            t_last = t_now

            has_hand = self.tracker.process(frame)
            pointer = None

            if has_hand:
                self.hand_lost_frames = 0
                pointer = self.tracker.get_index_tip()
                self.current_gesture = self.gesture_detector.classify(self.tracker)
                self.tracker.draw_skeleton(frame)

                # Distance telemetry
                ps = self.tracker.palm_scale
                if ps > 130:
                    self.distance_zone = "Close (<0.8m)"
                elif ps >= 65:
                    self.distance_zone = "Optimal (1-1.8m)"
                else:
                    self.distance_zone = "Far (2m-3.2m)"

                # Gesture actions
                if self.current_gesture == GestureType.PINCH_SIZE:
                    ratio = self.gesture_detector.pinch_ratio
                    self.drawing_engine.brush_radius = int(max(2, min(40, ratio * 150)))
                elif self.current_gesture == GestureType.HOVER:
                    self.drawing_engine.finish_stroke()
                elif self.current_gesture == GestureType.DRAW:
                    if pointer is not None:
                        self.drawing_engine.add_point(pointer)
                elif self.current_gesture == GestureType.ERASE:
                    erase_pt = self.tracker.get_palm_center() or pointer
                    if erase_pt is not None:
                        self.drawing_engine.eraser_radius = 55
                        if self.drawing_engine.active_stroke is None or not self.drawing_engine.active_stroke.is_eraser:
                            self.drawing_engine.start_stroke(is_eraser=True)
                        self.drawing_engine.add_point(erase_pt)
                elif self.current_gesture == GestureType.IDLE:
                    self.drawing_engine.finish_stroke()
            else:
                self.hand_lost_frames += 1
                if self.hand_lost_frames > 12:
                    self.drawing_engine.finish_stroke()
                    self.gesture_detector.reset()
                    self.current_gesture = GestureType.UNKNOWN
                    self.distance_zone = "Searching..."

            # Composite drawing over frame
            composite = self.drawing_engine.composite(frame)

            # Draw pointer cursor on frame
            if self.current_gesture == GestureType.ERASE:
                erase_pt = self.tracker.get_palm_center() or pointer
                if erase_pt is not None:
                    ex, ey = int(erase_pt[0]), int(erase_pt[1])
                    cv2.circle(composite, (ex, ey), 55, (50, 50, 255), 3, cv2.LINE_AA)
                    cv2.putText(composite, "ERASING", (ex - 40, max(25, ey - 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2, cv2.LINE_AA)
            elif pointer is not None:
                px, py = int(pointer[0]), int(pointer[1])
                if self.current_gesture == GestureType.DRAW:
                    cv2.circle(composite, (px, py), self.drawing_engine.brush_radius, self.drawing_engine.current_color_bgr, 2, cv2.LINE_AA)
                    cv2.circle(composite, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
                elif self.current_gesture == GestureType.HOVER:
                    cv2.circle(composite, (px, py), 16, (0, 255, 255), 2, cv2.LINE_AA)

            # Fast JPEG encoding for web stream (quality 70 for instant transmission)
            success, jpeg_buf = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if success:
                with self._lock:
                    self._latest_jpeg = jpeg_buf.tobytes()

            # Buffer encoded JPEG
            pass

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def get_status_dict(self) -> dict:
        char = getattr(self.drawing_engine, "recent_recognized_char", "")
        conf = getattr(self.drawing_engine, "recent_recognized_conf", 0.0)
        word = ""
        if hasattr(self.drawing_engine, "recognizer") and self.drawing_engine.recognizer:
            word = self.drawing_engine.recognizer.get_word()
        guidelines = getattr(self.drawing_engine, "guidelines_enabled", False)
        finger = getattr(self.tracker, "active_finger_name", "Index")

        return {
            "fps": round(self.fps, 1),
            "gesture": self.current_gesture.value,
            "distance_zone": self.distance_zone,
            "reach_gain": f"{self.tracker.current_reach_gain:.1f}x",
            "reach_multiplier": round(self.tracker.reach_multiplier, 2),
            "palm_scale": int(self.tracker.palm_scale),
            "stroke_count": len(self.drawing_engine.completed_strokes),
            "color_key": self.drawing_engine.current_color_key,
            "color_hex": self.drawing_engine.current_custom_hex or (
                self.drawing_engine.PALETTE.get(self.drawing_engine.current_color_key, (None, "#00F0FF"))[1]
            ),
            "brush_radius": self.drawing_engine.brush_radius,
            "canvas_mode": self.drawing_engine.canvas_mode,
            "smoothing_preset": getattr(self.tracker, "smoothing_preset", "balanced"),
            "recognized_char": char,
            "recognized_conf": round(conf, 2),
            "recognized_word": word,
            "guidelines_enabled": guidelines,
            "active_finger": finger,
            "gradient_mode": self.drawing_engine.gradient_mode,
            "gradient_end_hex": self.drawing_engine.gradient_end_hex,
            "writing_mode": getattr(self.gesture_detector, "writing_mode", "pinch"),
        }


# Singleton service instance
service = WebAirPenService()


class AirPenHTTPHandler(SimpleHTTPRequestHandler):
    """Handles static web files, REST APIs, and MJPEG video streaming."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. MJPEG Video Stream
        if path == "/video_feed":
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            try:
                while True:
                    frame = service.get_jpeg()
                    if frame:
                        self.wfile.write(b'--FRAME\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.016)
            except (ConnectionResetError, BrokenPipeError):
                pass
            return

        # 2. REST API: Status Telemetry
        elif path == "/api/status":
            data = service.get_status_dict()
            self._send_json(data)
            return

        # 3. REST API: Actions
        elif path.startswith("/api/action/"):
            action = path.replace("/api/action/", "")
            params = parse_qs(parsed.query)

            if action == "undo":
                res = service.drawing_engine.undo()
                self._send_json({"ok": True, "msg": "Stroke Undone" if res else "Nothing to undo"})
            elif action == "clear":
                service.drawing_engine.clear()
                self._send_json({"ok": True, "msg": "Canvas Cleared"})
            elif action == "guidelines":
                curr = getattr(service.drawing_engine, "guidelines_enabled", False)
                service.drawing_engine.guidelines_enabled = not curr
                st = "ON" if service.drawing_engine.guidelines_enabled else "OFF"
                self._send_json({"ok": True, "guidelines": service.drawing_engine.guidelines_enabled, "msg": f"Lined Guide: {st}"})
            elif action == "clear_word":
                if hasattr(service.drawing_engine, "recognizer") and service.drawing_engine.recognizer:
                    service.drawing_engine.recognizer.clear_word()
                service.drawing_engine.recent_recognized_char = ""
                service.drawing_engine.recent_recognized_conf = 0.0
                self._send_json({"ok": True, "msg": "Word Buffer Cleared"})
            elif action == "add_space":
                w = ""
                if hasattr(service.drawing_engine, "recognizer") and service.drawing_engine.recognizer:
                    w = service.drawing_engine.recognizer.add_space()
                self._send_json({"ok": True, "word": w, "msg": "Space Added"})
            elif action == "backspace":
                w = ""
                if hasattr(service.drawing_engine, "recognizer") and service.drawing_engine.recognizer:
                    w = service.drawing_engine.recognizer.backspace()
                self._send_json({"ok": True, "word": w, "msg": "Backspace"})
            elif action == "type_word":
                w = ""
                if hasattr(service.drawing_engine, "recognizer") and service.drawing_engine.recognizer:
                    w = service.drawing_engine.recognizer.get_word()
                if w.strip():
                    service.drawing_engine.write_text(w.strip(), x=120, y=280)
                    service.drawing_engine.recognizer.clear_word()
                    self._send_json({"ok": True, "word": "", "msg": f"Stamped to Canvas: '{w.strip()}'"})
                else:
                    self._send_json({"ok": False, "msg": "Word Buffer is Empty"})
            elif action == "pen_mode":
                m = params.get("m", ["pinch"])[0]
                service.gesture_detector.set_writing_mode(m)
                labels = {"pinch": "Pen-Grip (Pinch to Write)", "point": "Point-to-Draw (Classic)", "depth": "Virtual Touch (Z-Depth)"}
                self._send_json({"ok": True, "writing_mode": m, "msg": f"Mode: {labels.get(m, m.title())}"})
            elif action == "recognize":
                if hasattr(service.drawing_engine, "recognizer") and service.drawing_engine.recognizer:
                    raw_strokes = [s.points for s in service.drawing_engine.completed_strokes if not s.is_eraser and len(s.points) >= 3]
                    if raw_strokes:
                        res = service.drawing_engine.recognizer.recognize(raw_strokes)
                        service.drawing_engine.recent_recognized_char = res.character
                        service.drawing_engine.recent_recognized_conf = res.confidence
                w = service.drawing_engine.recognizer.get_word() if getattr(service.drawing_engine, "recognizer", None) else ""
                self._send_json({"ok": True, "char": service.drawing_engine.recent_recognized_char, "word": w})
            elif action == "mode":
                new_mode = service.drawing_engine.toggle_canvas_mode()
                self._send_json({"ok": True, "mode": new_mode, "msg": f"Canvas: {new_mode.capitalize()}"})
            elif action == "save":
                png, svg = service.drawing_engine.export_all()
                self._send_json({"ok": True, "png": png.name, "svg": svg.name, "msg": f"Saved {png.name}"})
            elif action == "color":
                c = params.get("c", ["cyan"])[0]
                service.drawing_engine.set_color(c)
                self._send_json({"ok": True, "color": c, "msg": f"Brush: {c.capitalize()}"})
            elif action == "custom_color":
                hex_val = params.get("hex", ["#00F0FF"])[0]
                service.drawing_engine.set_custom_color(hex_val)
                self._send_json({"ok": True, "color_hex": hex_val, "msg": f"Custom Colour: {hex_val}"})
            elif action == "gradient":
                mode = params.get("mode", ["none"])[0]
                end_hex = params.get("end", [""])[0]
                service.drawing_engine.set_gradient_mode(mode, end_hex)
                label = {"none": "Off", "rainbow": "Rainbow", "custom": "Custom Blend"}.get(mode, mode)
                self._send_json({"ok": True, "gradient_mode": mode, "msg": f"Gradient: {label}"})
            elif action == "reach":
                m = float(params.get("m", [1.35])[0])
                service.tracker.reach_multiplier = max(0.8, min(2.5, m))
                self._send_json({"ok": True, "reach_multiplier": service.tracker.reach_multiplier, "msg": f"Reach: {service.tracker.reach_multiplier:.1f}x"})
            elif action == "smooth":
                p = params.get("p", ["balanced"])[0]
                service.tracker.set_smoothing_preset(p)
                self._send_json({"ok": True, "preset": p, "msg": f"Smoothing: {p.capitalize()}"})
            elif action == "demo":
                def _draw_demo():
                    import math
                    cx, cy = service.width // 2, service.height // 2
                    service.drawing_engine.canvas_mode = "blackboard"
                    service.drawing_engine.set_color("cyan")
                    service.drawing_engine.start_stroke()
                    for t in np.linspace(0.0, 2.0 * math.pi, 140):
                        x = cx + 290 * math.sin(3 * t + math.pi / 2)
                        y = cy + 190 * math.sin(2 * t)
                        service.drawing_engine.add_point((float(x), float(y)))
                        time.sleep(0.007)
                    service.drawing_engine.finish_stroke()

                    service.drawing_engine.set_color("magenta")
                    service.drawing_engine.start_stroke()
                    for t in np.linspace(0.0, 2.0 * math.pi, 140):
                        x = cx + 230 * math.sin(5 * t + math.pi / 4)
                        y = cy + 150 * math.sin(4 * t)
                        service.drawing_engine.add_point((float(x), float(y)))
                        time.sleep(0.007)
                    service.drawing_engine.finish_stroke()
                    service.drawing_engine.export_all()

                threading.Thread(target=_draw_demo, daemon=True).start()
                self._send_json({"ok": True, "msg": "Drawing AI Harmonic Demo..."})
            elif action == "brush":
                r = int(params.get("r", [6])[0])
                service.drawing_engine.brush_radius = r
                self._send_json({"ok": True, "radius": r})
            elif action == "draw_point":
                x = float(params.get("x", [0])[0])
                y = float(params.get("y", [0])[0])
                state = params.get("state", ["move"])[0]
                if state == "down":
                    service.drawing_engine.start_stroke()
                    service.drawing_engine.add_point((x, y))
                elif state == "up":
                    service.drawing_engine.finish_stroke()
                else:
                    service.drawing_engine.add_point((x, y))
                self._send_json({"ok": True, "x": x, "y": y, "state": state})
            elif action == "write":
                text = params.get("text", ["AIR-PEN"])[0]
                x = int(params.get("x", [140])[0])
                y = int(params.get("y", [360])[0])
                service.drawing_engine.write_text(text, x=x, y=y)
                self._send_json({"ok": True, "msg": f"Wrote: {text}"})
            else:
                self._send_json({"error": "Unknown action"}, status=404)
            return

        # 4. REST API: Gallery
        elif path == "/api/gallery":
            items = []
            for png in sorted(OUTPUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
                svg_name = png.stem + ".svg"
                items.append({
                    "png": png.name,
                    "svg": svg_name if (OUTPUT_DIR / svg_name).exists() else png.name
                })
            self._send_json(items)
            return

        # 5. Serve Output Directory Files (PNG, SVG)
        elif path.startswith("/output/"):
            filename = path.replace("/output/", "")
            file_path = OUTPUT_DIR / filename
            if file_path.exists() and file_path.is_file():
                self._serve_file(file_path)
            else:
                self.send_error(404, "File not found")
            return

        # 6. Serve Static Web Frontend (index.html, style.css, app.js)
        else:
            if path == "/" or path == "":
                target_file = WEB_DIR / "index.html"
            else:
                target_file = WEB_DIR / path.lstrip("/")

            if target_file.exists() and target_file.is_file():
                self._serve_file(target_file)
            else:
                self.send_error(404, "Not Found")

    def _serve_file(self, file_path: Path):
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Mute repetitive MJPEG / polling HTTP logs
        return


def run_server(port: int = 8000):
    """Start threaded HTTP server."""
    service.start()
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, AirPenHTTPHandler)
    print(f"===========================================================")
    print(f">> AIR-PEN STUDIO IS LIVE AT: http://localhost:{port}")
    print(f"   Native 45+ FPS, Dynamic 2.85x Reach Coverage, Bezier Smooth")
    print(f"===========================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        service._running = False
        service.camera.stop()
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)

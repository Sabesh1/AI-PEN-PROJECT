import pytest
from AIR_PEN.backend.server import WebAirPenService

def test_server_status_dict_fields():
    service = WebAirPenService(1280, 720)
    status = service.get_status_dict()
    assert "fps" in status
    assert "gesture" in status
    assert "distance_zone" in status
    assert "reach_gain" in status
    assert "reach_multiplier" in status
    assert "stroke_count" in status
    assert "color_key" in status
    assert "color_hex" in status
    assert "canvas_mode" in status
    assert "smoothing_preset" in status
    assert "recognized_char" in status
    assert "recognized_conf" in status
    assert "recognized_word" in status
    assert "guidelines_enabled" in status
    assert "active_finger" in status
    assert "gradient_mode" in status
    assert "gradient_end_hex" in status

def test_server_guidelines_and_word_buffer():
    service = WebAirPenService(1280, 720)
    assert service.drawing_engine.guidelines_enabled is False
    service.drawing_engine.guidelines_enabled = True
    assert service.get_status_dict()["guidelines_enabled"] is True
    service.drawing_engine.recognizer.word_buffer = ["A", "I", "R"]
    assert service.get_status_dict()["recognized_word"] == "AIR"
    service.drawing_engine.recognizer.clear_word()
    assert service.get_status_dict()["recognized_word"] == ""

def test_custom_color_picker():
    service = WebAirPenService(1280, 720)
    de = service.drawing_engine
    de.set_custom_color("#FF6600")
    assert de.current_color_key == "custom"
    assert de.current_custom_hex == "#FF6600"
    assert de.current_custom_color_bgr == (0, 102, 255)
    assert de.current_color_bgr == (0, 102, 255)
    de.set_color("cyan")
    assert de.current_color_key == "cyan"
    assert de.current_custom_color_bgr is None

def test_gradient_mode_rainbow():
    service = WebAirPenService(1280, 720)
    de = service.drawing_engine
    de.set_gradient_mode("rainbow")
    assert de.gradient_mode == "rainbow"
    de.start_stroke()
    stroke = de.active_stroke
    assert stroke is not None
    assert stroke.gradient_mode == "rainbow"
    c0 = stroke.get_color_at(0.0)
    c1 = stroke.get_color_at(0.5)
    assert c0 != c1

def test_gradient_mode_custom():
    service = WebAirPenService(1280, 720)
    de = service.drawing_engine
    de.set_color("cyan")
    de.set_gradient_mode("custom", "#FF0000")
    assert de.gradient_mode == "custom"
    assert de.gradient_end_hex == "#FF0000"
    de.start_stroke()
    stroke = de.active_stroke
    assert stroke is not None
    assert stroke.gradient_mode == "custom"
    c_start = stroke.get_color_at(0.0)
    c_end   = stroke.get_color_at(1.0)
    assert c_start == de.PALETTE["cyan"][0]
    assert c_end == (0, 0, 255)

def test_gradient_mode_none_clears():
    service = WebAirPenService(1280, 720)
    de = service.drawing_engine
    de.set_gradient_mode("rainbow")
    de.set_gradient_mode("none")
    assert de.gradient_mode == "none"
    assert de.gradient_end_bgr is None

def test_pen_modes():
    service = WebAirPenService(1280, 720)
    status = service.get_status_dict()
    assert "writing_mode" in status
    assert status["writing_mode"] == "pinch"
    service.gesture_detector.set_writing_mode("point")
    assert service.get_status_dict()["writing_mode"] == "point"
    service.gesture_detector.set_writing_mode("depth")
    assert service.get_status_dict()["writing_mode"] == "depth"

def test_word_actions():
    service = WebAirPenService(1280, 720)
    rec = service.drawing_engine.recognizer
    rec.clear_word()
    rec.word_buffer = ["H", "I"]
    assert rec.get_word() == "HI"
    rec.add_space()
    assert rec.get_word() == "HI "
    rec.word_buffer.extend(["A", "L", "L"])
    assert rec.get_word() == "HI ALL"
    rec.backspace()
    assert rec.get_word() == "HI AL"
    rec.clear_word()
    assert rec.get_word() == ""

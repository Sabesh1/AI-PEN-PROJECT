import math
import pytest
from AIR_PEN.backend.recognition import AirCharacterRecognizer

def test_recognize_vertical_line_as_one():
    rec = AirCharacterRecognizer()
    # Vertical line from (100, 50) to (100, 200)
    pts = [[(100.0, float(y)) for y in range(50, 201, 5)]]
    res = rec.recognize(pts)
    assert res.character in ["1", "I", "l"]
    assert res.confidence > 0.8

def test_recognize_circle_as_o():
    rec = AirCharacterRecognizer()
    # Circle radius 50 at (100, 100)
    pts = []
    for deg in range(0, 361, 10):
        rad = math.radians(deg)
        pts.append((100.0 + 50.0 * math.cos(rad), 100.0 + 50.0 * math.sin(rad)))
    res = rec.recognize([pts])
    assert res.character in ["O", "o", "0"]
    assert res.confidence > 0.85

def test_recognize_two_stroke_t():
    rec = AirCharacterRecognizer()
    # Horizontal top bar
    bar = [(float(x), 50.0) for x in range(50, 151, 10)]
    # Vertical stem
    stem = [(100.0, float(y)) for y in range(50, 181, 10)]
    res = rec.recognize([bar, stem])
    assert res.character == "T"
    assert res.confidence > 0.9

def test_word_buffer_accumulation():
    rec = AirCharacterRecognizer()
    rec.clear_word()
    # Write 'T'
    bar = [(float(x), 50.0) for x in range(50, 151, 10)]
    stem = [(100.0, float(y)) for y in range(50, 181, 10)]
    rec.recognize([bar, stem])
    
    # Write 'O'
    circle = []
    for deg in range(0, 361, 15):
        rad = math.radians(deg)
        circle.append((100.0 + 40.0 * math.cos(rad), 100.0 + 40.0 * math.sin(rad)))
    rec.recognize([circle])

    word = rec.get_word()
    assert "T" in word
    assert "O" in word
    rec.clear_word()
    assert rec.get_word() == ""

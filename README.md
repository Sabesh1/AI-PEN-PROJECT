# Air-Pen Studio: Next-Gen Touchless Gesture Drawing System

> **A real-time, touchless computer vision air-writing and drawing engine** powered by **MediaPipe Hands**, **OpenCV**, and the **Casiez One-Euro (1€) Jitter Filter**. Turn any standard webcam into a high-precision digital whiteboard without physical markers, special gloves, or touchscreens.

---

## Highlights & Features

- ☝️ **Natural Touchless Gestures**:
  - **Index Pointing**: Seamless air-drawing with sub-millimeter precision.
  - **Peace Sign (✌️)**: Hover cursor without marking; contactless dwell-time activation on UI buttons.
  - **Three Fingers (☝️✌️)**: Instant eraser mode.
  - **Pinch (🤏)**: Real-time dynamic brush radius adjustment.
  - **Open Palm (🖐️)**: Pause / idle mode.
- ⚡ **Zero-Lag Architecture**: Multi-threaded camera acquisition completely flushes OS driver frame queues for instantaneous feedback.
- 🎯 **One-Euro (1€) Jitter Elimination**: Speed-adaptive low-pass filtering eliminates hand trembling during slow, deliberate writing while maintaining zero lag during rapid strokes.
- 🔭 **Long-Distance Optimization**: Normalizes 3D spatial coordinates against palm skeletal anatomy; automatically increases filter stability and expands UI button hitboxes when standing 2+ meters away.
- 🌟 **Neon Glow Rendering Engine**: Multi-pass Gaussian bloom and Catmull-Rom spline interpolation turn discrete webcam frames into silky smooth vector curves.
- 🖥️ **Glassmorphism In-Stream HUD**: Translucent floating control toolbar with radial dwell progress rings, telemetry diagnostics, and real-time FPS counter.
- 💾 **Multi-Format Export**:
  - Infinite-resolution Scalable Vector Graphics (**SVG**)
  - High-res **PNG** with transparent or dark/white blackboard backgrounds
  - Full **JSON** stroke session timeline for replay and analysis

---

## 18-Step Engineering Architecture

```
[Webcam Feed]
      │ (ThreadedCamera - 60 FPS)
      ▼
[MediaPipe Hands 21 3D Landmarks]
      │
      ├───> [FingerTracker] ──> Palm Scale Normalization
      │                             │
      ├───> [LongDistanceOptimizer] ─┴──> Adaptive 1€ Filter Gain & UI Padding
      │
      ├───> [OneEuroFilter2D] ──────────> Sub-Millimeter Jitter Removal
      │
      ├───> [GestureRecognizer] ────────> State Machine (DRAW, HOVER, ERASE, PINCH, IDLE)
      │
      ▼
[DrawingEngine] ────────────────────────> Spline Interpolation & Neon Glow Shader
      │
      ├───> [TouchlessToolbar] ─────────> Dwell-Time Interactive Buttons
      ├───> [HudOverlay] ───────────────> Glassmorphic Interface
      └───> [Exporter] ─────────────────> PNG / SVG / JSON
```

---

## Gesture Control Cheatsheet

| Gesture | Pose Description | Action |
| :--- | :--- | :--- |
| ☝️ **Point** | Index extended, others curled | **Air-Draw**: Inks glowing strokes on the canvas |
| ✌️ **Peace** | Index + Middle extended | **Hover / Select**: Moves pointer without drawing; hover over top buttons for 0.65s to click |
| ☝️✌️ **Three**| Index + Middle + Ring extended | **Eraser**: Erases artwork under the reticle |
| 🤏 **Pinch** | Thumb & Index tips close together | **Brush Adjust**: Pinch wider/narrower to resize stroke width |
| 🖐️ **Palm**  | All fingers spread wide | **Idle / Freeze**: Pauses interaction to avoid unwanted markings |

### Keyboard Shortcuts (Optional Fallbacks)
- `U` : Undo last stroke
- `C` : Clear canvas
- `S` : Save artwork (PNG + SVG)
- `M` : Cycle canvas mode (Camera Overlay → Blackboard → Whiteboard)
- `E` : Toggle eraser tool
- `Q` / `ESC` : Quit application

---

## Quick Start & Installation

### 1. Prerequisites
- Python 3.11 installed (or managed via `uv`).
- Standard USB or built-in webcam.

### 2. Environment Setup
```powershell
# Create virtual environment and install dependencies
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Launching Air-Pen Studio
```powershell
# Launch main real-time application with live webcam
python main.py
```

### 4. Running the Automated Showcase Demo
If you do not have a webcam connected or want to verify rendering and export:
```powershell
# Run synthetic demonstration mode
python demo/demo_runner.py
```
Exported artwork is automatically saved to the `exports/` folder.

---

## Running the Automated Test Suite

The system includes a comprehensive pytest suite testing smoothing filters, geometric gesture recognition, undo/redo stacks, SVG/PNG exporters, and long-distance adaptations:

```powershell
pytest tests/ -v
```

---

## Project Directory Structure

```
pen project/
├── config/
│   ├── __init__.py
│   └── settings.py             # Configurable color palettes, camera resolutions, thresholds
├── core/
│   ├── __init__.py
│   ├── camera.py               # Threaded low-latency video capture
│   ├── hand_detector.py        # MediaPipe Hands 21-landmark detector
│   ├── finger_tracker.py       # Fingertip tracking and palm scale normalization
│   ├── gesture_recognizer.py   # Geometric gesture classifier with hysteresis
│   ├── smoothing.py            # Casiez One-Euro (1€) adaptive jitter filter
│   └── long_distance.py        # Distance estimator and adaptive parameter scaling
├── engine/
│   ├── __init__.py
│   ├── drawing_engine.py       # Multi-layer canvas, neon glow shader, undo/redo
│   ├── stroke.py               # Stroke data model and Catmull-Rom spline interpolation
│   └── exporter.py             # High-res PNG, Scalable SVG, and JSON export
├── ui/
│   ├── __init__.py
│   ├── hud_overlay.py          # Glassmorphic HUD overlay and telemetry badges
│   └── toolbar.py              # Touchless button manager with dwell hit-testing
├── utils/
│   ├── __init__.py
│   └── profiler.py             # Real-time FPS and latency diagnostic profiler
├── tests/
│   ├── test_engine.py
│   ├── test_exporter.py
│   ├── test_gestures.py
│   ├── test_long_distance.py
│   └── test_smoothing.py
├── demo/
│   ├── demo_runner.py          # Automated synthetic showcase script
│   └── presentation.md         # Demo pitch script and presentation outline
├── exports/                    # Saved artwork (PNG, SVG, JSON)
├── main.py                     # Primary executable entry point
├── requirements.txt            # Python dependencies
└── README.md                   # Full documentation
```

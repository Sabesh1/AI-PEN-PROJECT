"""Backend package for AIR_PEN."""

from pathlib import Path
import sys

_app_dir = Path(__file__).resolve().parent.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

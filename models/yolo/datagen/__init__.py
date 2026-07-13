"""Synthetic minefield dataset generation (YOLO format) via Blender.

Replaces IARC_mission_10/blender/scatter_mines.py. Design rule: the pure
pipeline (config / scatter / flightpath / labels / scene / manifest —
stdlib only) decides EVERYTHING: mine placement, camera stations, label
boxes. Blender (generate.py, the only bpy importer) is a dumb renderer
that mirrors the scene. Labels are never derived from bpy state, so a
scene-assembly bug shows up as a visible image/label mismatch in spot
checks instead of silently wrong training data.

Geometry comes from mission_engine.core — the SAME CameraModel,
serpentine and projection code the drone flies with. Training-time
geometry == runtime geometry by construction, including the 10-degree
forward camera tilt the old script ignored.
"""

import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[3] / "ros" / "mission_engine"
if _ENGINE.is_dir() and str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

try:
    import mission_engine.core  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "datagen reuses mission_engine.core (the runtime geometry). Expected "
        f"the package at {_ENGINE} (monorepo layout) or on PYTHONPATH."
    ) from e

"""Config loading: configs/default.yaml + configs/arms/<arm_profile>.yaml (deep-merged, the arm file wins).

The arm profile comes from (in order): the `arm` argument, the ROVER_ARM environment variable, `arm_profile` in the
base file. Pure python + PyYAML so the Isaac bridge (isaacenv6) can import it too.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, arm: str | None = None) -> dict:
    base = yaml.safe_load(open(path or ROOT / "configs/default.yaml"))
    profile = arm or os.environ.get("ROVER_ARM") or base.get("arm_profile", "rover2026")
    arm_file = ROOT / "configs/arms" / f"{profile}.yaml"
    cfg = _merge(base, yaml.safe_load(open(arm_file)))
    cfg["arm_profile"] = profile
    return cfg


def top_down(tool_axis) -> "np.ndarray":
    """Rotation (tool frame -> base frame) that points the tool axis straight down (-z), keeping it deterministic."""
    import numpy as np
    t = np.asarray(tool_axis, float)
    t = t / np.linalg.norm(t)
    # choose tool-frame axes (a, b, t) right-handed, then map t -> -z, a -> +x or +y
    helper = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.cross(helper, t); a /= np.linalg.norm(a)
    b = np.cross(t, a)
    tool = np.stack([a, b, t], axis=1)                     # columns in tool coords
    world = np.stack([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], axis=1)  # a -> x, b -> -y, t -> -z (det +1)
    return world @ tool.T

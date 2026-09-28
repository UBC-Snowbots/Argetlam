"""Human wrist (relative to shoulder) -> robot TCP target in the arm base frame.

p_robot = origin + scale * A @ (wrist - shoulder), then EMA-smoothed and workspace-clipped by safety.
`A` maps MediaPipe world axes (x right, y down, z toward camera) to the robot base frame;
calibrate it (and `scale`, `origin`) once the camera placement is fixed.
"""
from __future__ import annotations

import numpy as np

from ..types import BodyPose


class Retargeter:
    def __init__(self, axes, scale: float, origin, alpha: float = 0.3):
        self.A = np.asarray(axes, float)
        self.scale = float(scale)
        self.origin = np.asarray(origin, float)
        self.alpha = alpha
        self._p: np.ndarray | None = None

    def __call__(self, pose: BodyPose) -> np.ndarray:
        rel = np.asarray(pose.wrist) - np.asarray(pose.shoulder)
        p = self.origin + self.scale * self.A @ rel
        self._p = p if self._p is None else (1 - self.alpha) * self._p + self.alpha * p
        return self._p.copy()

    def reset(self):
        self._p = None

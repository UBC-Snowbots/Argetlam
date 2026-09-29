"""Human wrist (relative to shoulder) -> robot TCP target, in RELATIVE ("clutch") mode.

On `engage(tcp_now)` (when the user says "follow me") the arm stays where it is; the first wrist sample after that
becomes the neutral pose. From then on:
    p_robot = tcp_at_engage + scale * A @ ((wrist - shoulder) - neutral)
EMA-smoothed; the safety layer clips it to the workspace. `A` maps MediaPipe world axes (x right, y down,
z toward camera) to the robot base frame (x forward, y left, z up). Absolute mapping would make the arm jump to
wherever your arm happens to be (a hanging arm maps below the table).
"""
from __future__ import annotations

import numpy as np

from ..types import BodyPose


class Retargeter:
    def __init__(self, axes, scale: float, origin, alpha: float = 0.3):
        self.A = np.asarray(axes, float)
        self.scale = float(scale)
        self.anchor = np.asarray(origin, float)  # TCP at engage (default until the first engage)
        self.alpha = alpha
        self._neutral: np.ndarray | None = None
        self._p: np.ndarray | None = None

    def engage(self, tcp_now) -> None:
        self.anchor = np.asarray(tcp_now, float).copy()
        self._neutral, self._p = None, None

    def __call__(self, pose: BodyPose) -> np.ndarray:
        rel = np.asarray(pose.wrist) - np.asarray(pose.shoulder)
        if self._neutral is None:
            self._neutral = rel
        p = self.anchor + self.scale * self.A @ (rel - self._neutral)
        self._p = p if self._p is None else (1 - self.alpha) * self._p + self.alpha * p
        return self._p.copy()

    def reset(self):
        self._neutral, self._p = None, None

"""Human wrist (relative to shoulder) -> robot TCP target, in RELATIVE ("clutch") mode.

On `engage(tcp_now)` ("follow me") the arm stays where it is; the first wrist sample after that becomes neutral:
    p_robot = tcp_at_engage + gains * (A @ ((wrist - shoulder) - neutral))
then a One-Euro filter (smooth when still, low lag when moving fast) and the safety layer's workspace clip.

Axes: MediaPipe world (x = image right, y = down, z = toward the camera is smaller) -> robot (x forward, y left, z up),
for a user FACING the webcam and watching the Isaac view from BEHIND the arm ("over the shoulder"):
    hand toward the screen -> gripper forward (+x);  hand to YOUR right -> gripper to the right on screen (-y);  up -> up.
Your right hand appears on the image LEFT (x decreases) in a raw, un-mirrored webcam frame, hence robot_y = +x_mp.
If your webcam mirrors the image, press M in the body client (it flips x before sending).
"""
from __future__ import annotations

import math

import numpy as np

from ..types import BodyPose


class OneEuro:
    """One-Euro filter (Casiez et al. 2012), vectorised over axes."""

    def __init__(self, min_cutoff=1.0, beta=0.7, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self._x is None or t <= self._t:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = np.array([self._alpha(c, dt) for c in np.atleast_1d(cutoff)])
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x.copy()

    def reset(self):
        self._x = self._dx = self._t = None


class Retargeter:
    def __init__(self, axes, scale, origin, alpha: float = 0.3, min_cutoff: float = 1.0, beta: float = 0.7,
                 ws_min=None, ws_max=None):
        self.ws_min = None if ws_min is None else np.asarray(ws_min, float)
        self.ws_max = None if ws_max is None else np.asarray(ws_max, float)
        self.clipped = False
        self.A = np.asarray(axes, float)
        self.gains = np.broadcast_to(np.asarray(scale, float), (3,)).copy()  # per robot axis (x, y, z)
        self.anchor = np.asarray(origin, float)
        self.filter = OneEuro(min_cutoff, beta)
        self._neutral: np.ndarray | None = None

    def engage(self, tcp_now, neutral=None) -> None:
        """neutral: calibrated wrist-shoulder vector (from the clutch); None = the next sample."""
        self.anchor = np.asarray(tcp_now, float).copy()
        self._neutral = None if neutral is None else np.asarray(neutral, float)
        self.filter.reset()
        self.clipped = False

    def __call__(self, pose: BodyPose) -> np.ndarray:
        rel = np.asarray(pose.wrist) - np.asarray(pose.shoulder)
        if self._neutral is None:
            self._neutral = rel
        p = self.filter(self.anchor + self.gains * (self.A @ (rel - self._neutral)), pose.t)
        if self.ws_min is not None:
            c = np.clip(p, self.ws_min, self.ws_max)
            self.clipped = bool(np.max(np.abs(c - p)) > 0.005)  # > 5 mm outside the box = really out of reach
            p = c
        return p

    def reset(self):
        self._neutral = None
        self.filter.reset()

"""Shared One-Euro filter for body teleop.

This used to also hold the legacy "position" retargeter (wrist-rel-shoulder -> gripper offset); that mode was removed
when the body teleop was cut down to PLANE (default) and ARM_MIRROR (fallback). Only the One-Euro filter remains, used
by body/plane.py and body/arm_mirror.py.
"""
from __future__ import annotations

import math

import numpy as np


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

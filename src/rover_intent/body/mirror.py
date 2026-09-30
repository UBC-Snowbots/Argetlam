"""Gesture mirroring: your arm's three natural motions drive the rover arm's three natural motions.

    you                                   robot (cylindrical coords around its base yaw axis)
    swing your hand left / right      ->  base rotates left / right          (yaw)
    raise / lower your hand           ->  gripper up / down                  (height)
    straighten / bend your elbow      ->  gripper reaches farther / closer   (radius)

Why not copy joint angles 1:1? Measured: every top-down gripper pose this arm can reach has its upper arm ~74 deg
up (it reaches with elbow + forearm), while a human upper arm hangs down; mirroring the upper-arm angle made
"raise your arm" push the gripper DOWN. So we mirror motions, not angles.

Robustness: lateral and vertical hand motion are measured in the torso frame from mostly image-plane components;
reach comes from the elbow angle (regularised by the pose model). Monocular depth is never used directly.
Relative ("clutch"): on engage the arm stays put; afterwards targets move by gain * (your change since engage).
Out-of-reach targets are clamped to the reachable band and flagged (`clipped`) so the ghost marker turns red.
"""
from __future__ import annotations

import numpy as np

from ..control.kinematics import ArmModel
from ..types import BodyPose
from .retarget import OneEuro


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def human_features(p: BodyPose):
    """-> np.array([lateral_m, up_m, extension]) or None. lateral + = toward your LEFT, up + = up (metres, wrist
    relative to shoulder, torso frame); extension = |shoulder->wrist| / (upper arm + forearm), 1 = straight arm."""
    if p.elbow is None or p.shoulder_other is None:
        return None
    S, E, W, O = (np.asarray(x, float) for x in (p.shoulder, p.elbow, p.wrist, p.shoulder_other))
    up = _unit((S + O) / 2 - np.asarray(p.hips, float)) if p.hips is not None else np.array([0.0, -1.0, 0.0])
    medial = O - S if p.side == "right" else S - O
    left = _unit(medial - np.dot(medial, up) * up)
    r = W - S
    ext = np.linalg.norm(r) / (np.linalg.norm(E - S) + np.linalg.norm(W - E) + 1e-9)
    return np.array([np.dot(r, left), np.dot(r, up), ext])


class GestureMirror:
    def __init__(self, arm: ArmModel, gains=(2.5, 1.0, 0.6), radius=(0.30, 0.75), height=(0.03, 0.55),
                 min_cutoff=1.0, beta=0.5):
        """gains: (rad of base yaw per m of lateral hand motion, m of height per m of hand height,
        m of reach per unit of arm extension)."""
        self.arm = arm
        self.gains = np.asarray(gains, float)
        self.r_lim, self.z_lim = radius, height
        _, o, _ = arm.fk(np.zeros(arm.dof), with_frames=True)
        self.axis_xy = o[0][:2]                     # base yaw axis
        self.filter = OneEuro(min_cutoff, beta)
        self._h0 = self._c0 = None
        self.clipped = False
        self.tcp_target = None

    def _cyl(self, p):
        d = p[:2] - self.axis_xy
        return np.array([np.arctan2(d[1], d[0]), np.hypot(*d), p[2]])  # yaw, radius, height

    def _cart(self, c):
        return np.array([self.axis_xy[0] + c[1] * np.cos(c[0]), self.axis_xy[1] + c[1] * np.sin(c[0]), c[2]])

    def engage(self, tcp_now, neutral=None):
        self._c0 = self._cyl(np.asarray(tcp_now, float))
        self._h0 = None
        self.filter.reset()
        self.clipped = False
        self.tcp_target = np.asarray(tcp_now, float).copy()

    def reset(self):
        self._c0 = self._h0 = None
        self.filter.reset()
        self.clipped = False
        self.tcp_target = None

    def __call__(self, pose: BodyPose):
        """-> TCP target (3,) or None if the pose is unusable."""
        if self._c0 is None:
            return None
        h = human_features(pose)
        if h is None:
            return None
        h = self.filter(h, pose.t)
        if self._h0 is None:
            self._h0 = h
        dh = h - self._h0
        c = self._c0 + np.array([self.gains[0] * dh[0], self.gains[2] * dh[2], self.gains[1] * dh[1]])
        c_clamped = c.copy()
        c_clamped[1] = np.clip(c[1], *self.r_lim)
        c_clamped[2] = np.clip(c[2], *self.z_lim)
        self.clipped = not np.allclose(c, c_clamped)
        self.tcp_target = self._cart(c_clamped)
        return self.tcp_target.copy()

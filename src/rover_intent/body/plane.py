"""Plane teleop: drive the TCP in the plane PARALLEL to the camera image, ignoring MediaPipe depth (z) entirely.

Why: the first real teleop run showed that motion TOWARD/AWAY from the camera (monocular depth, MediaPipe z) is the
noisy axis: ~28 mm std while holding still, versus ~3 mm in the image plane. So every control here is read from
IMAGE-PLANE landmarks (x, y) only, never z, and normalized by the shoulder width (in image units) with the shoulder
midpoint as the origin, so the mapping is independent of how far the operator stands from the camera.

Mapping (relative / clutched; the hand positions and the current TCP are captured on engage so there is no jump):
    robot y (lateral)  <- RIGHT hand horizontal   hand to the operator's RIGHT -> gripper to their right (robot -y)
    robot z (vertical) <- RIGHT hand vertical      raise -> up
    robot x (reach)    <- LEFT hand height         raise -> forward / away from the base, lower -> back
    target = tcp0 + ramp * gain * (hand_now - hand0)      (gain in metres per shoulder-width)
then a One-Euro filter on the normalized hand coordinates, a small deadband, a 2 s engage ramp, and the SafetyLayer
workspace box (targets outside the box are clamped and `.clipped` is flagged, so the ghost shows amber). The clamped
target still runs through the normal IK path in the app (orient from the profile's teleop_orient).

The image points come from `BodyPose.skel2d` (already mirror-corrected in the body client, so pressing M / --mirror
keeps this consistent). skel2d order is the body client's SKEL list:
    [0]=L shoulder [1]=R shoulder [2]=L elbow [3]=R elbow [4]=L wrist [5]=R wrist [6]=L hip [7]=R hip [8]=nose.
In that mirror-corrected frame the operator's RIGHT side sits at the SMALLER x (verified on the 2026-10-23 session:
R shoulder x ~0.41 < L shoulder x ~0.74), so moving the right hand to the operator's right DECREASES x; with a
positive lateral gain that drives robot -y, i.e. the gripper to the operator's right. See docs/PLANE_TELEOP.md.
"""
from __future__ import annotations

import numpy as np

from ..types import BodyPose
from .retarget import OneEuro

# skel2d (body client SKEL) indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, NOSE = range(9)

DEFAULTS = {
    "side": "right",                                  # the operator's CONTROL hand (lateral + vertical); other = reach
    "gain": {"lateral": 0.20, "vertical": 0.15, "reach": 0.15},  # metres of TCP per shoulder-width of hand motion
    "sign": {"lateral": 1.0, "vertical": 1.0, "reach": 1.0},     # flip one if a control drives the wrong way
    "one_euro": {"min_cutoff": 1.0, "beta": 0.05},    # on the NORMALIZED hand coords (steady when still, low lag)
    "deadband": 0.015,                                # shoulder-widths: hold a coord until it drifts this far
    "ramp_s": 2.0,                                    # ease the gain in from 0 over this long on engage (no jump)
    "drop_below_hip_s": 0.7,                          # right wrist below the hip line this long -> "right arm dropped"
    "conf_gate": True,                                # skip a pose that is not visible (the TCP holds)
}


def _cfg(cfg, key):
    d = DEFAULTS[key]
    v = (cfg or {}).get(key, d)
    if isinstance(d, dict) and isinstance(v, dict):
        out = dict(d)
        out.update(v)
        return out
    return v


def plane_coords(pose: BodyPose, side: str = "right"):
    """Normalized image-plane hand coordinates, or None if the pose is unusable. x, y only; z is never read.

    Returns dict:
      ctrl_x    control hand horizontal, shoulder-widths from the shoulder midpoint (+ = toward operator's LEFT).
      ctrl_up   control hand height, shoulder-widths above the shoulder midpoint (+ = up; image y is flipped).
      reach_up  reach (other) hand height, same units (drives forward/back reach).
      reach_ok  the reach hand is actually tracked (BodyPose.wrist_other visible), so reach_up is trustworthy.
      below_hip the control wrist is below the hip line (image y) -> used for the "right arm dropped" release.
    """
    sk = pose.skel2d
    if not sk or len(sk) < 8:
        return None
    pts = np.asarray(sk, float)
    l_sh, r_sh = pts[L_SH], pts[R_SH]
    sh_mid = (l_sh + r_sh) / 2.0
    sw = float(np.linalg.norm(l_sh - r_sh))
    if sw < 1e-6:
        return None
    ctrl_wr = pts[R_WR] if side == "right" else pts[L_WR]
    reach_wr = pts[L_WR] if side == "right" else pts[R_WR]
    hip_y = float((pts[L_HIP][1] + pts[R_HIP][1]) / 2.0)

    def nx(p):
        return float((p[0] - sh_mid[0]) / sw)

    def nup(p):
        return float(-(p[1] - sh_mid[1]) / sw)   # image y points DOWN; flip so raising the hand is +

    return {"ctrl_x": nx(ctrl_wr), "ctrl_up": nup(ctrl_wr), "reach_up": nup(reach_wr),
            "reach_ok": pose.wrist_other is not None, "below_hip": float(ctrl_wr[1]) > hip_y}


class PlaneMapper:
    """BodyPose -> a 3-vector TCP target in the arm base frame, for the app's IK path. Drop-in for `retarget`:
    `engage(tcp_now)` on clutch-engage, `pose -> p` each frame, `reset()`, and `.clipped` for the amber ghost."""

    def __init__(self, arm, home, cfg: dict | None = None, ws_min=None, ws_max=None):
        self.arm = arm
        self.home = np.asarray(home, float).copy()
        self.ws_min = None if ws_min is None else np.asarray(ws_min, float)
        self.ws_max = None if ws_max is None else np.asarray(ws_max, float)
        self.side = _cfg(cfg, "side")
        g = _cfg(cfg, "gain")
        s = _cfg(cfg, "sign")
        # per robot axis (x reach, y lateral, z vertical), already carrying the sign
        self.gain = np.array([g["reach"] * s["reach"], g["lateral"] * s["lateral"], g["vertical"] * s["vertical"]])
        oe = _cfg(cfg, "one_euro")
        self.filter = OneEuro(oe.get("min_cutoff", 1.0), oe.get("beta", 0.05))
        self.deadband = float(_cfg(cfg, "deadband"))
        self.ramp_s = float(_cfg(cfg, "ramp_s"))
        self.conf_gate = bool(_cfg(cfg, "conf_gate"))
        self.clipped = False
        self.reset()

    def reset(self):
        self._tcp0 = None
        self._n0 = None            # captured [ctrl_x, ctrl_up, reach_up] at engage
        self._held = None          # last emitted normalized coords (deadband)
        self._reach_hold = 0.0     # last trustworthy reach_up (held while the reach hand is not tracked)
        self._t_engage = None
        self.filter.reset()
        self.clipped = False

    def engage(self, tcp_now, neutral=None, *args, **kwargs):
        """Anchor at the current TCP. `neutral` / extra args are accepted and ignored (the app passes the clutch
        neutral and sometimes recentre=...); plane mode captures its own neutral from the first sample after engage,
        so engaging never jumps (target == tcp0 on the first frame)."""
        self._tcp0 = np.asarray(tcp_now, float)[:3].copy()
        self._n0 = None
        self._held = None
        self._reach_hold = 0.0
        self._t_engage = None
        self.filter.reset()
        self.clipped = False

    def __call__(self, pose: BodyPose):
        if self._tcp0 is None:
            return None
        if self.conf_gate and not pose.visible:
            return None
        c = plane_coords(pose, self.side)
        if c is None:
            return None
        reach = c["reach_up"] if c["reach_ok"] else self._reach_hold   # hold reach when the left hand isn't tracked
        if c["reach_ok"]:
            self._reach_hold = c["reach_up"]
        n = np.array([reach, c["ctrl_x"], c["ctrl_up"]], float)   # robot-axis order: x reach, y lateral, z vertical
        n = self.filter(n, pose.t)
        if self._held is None:
            self._held = n.copy()
        else:
            move = np.abs(n - self._held) >= self.deadband
            self._held = np.where(move, n, self._held)
        if self._n0 is None:
            self._n0 = self._held.copy()
            self._t_engage = pose.t
        s = 1.0 if self.ramp_s <= 0 else max(0.0, min(1.0, (pose.t - self._t_engage) / self.ramp_s))
        delta = self._held - self._n0
        p = self._tcp0 + s * self.gain * delta
        if self.ws_min is not None:
            cl = np.clip(p, self.ws_min, self.ws_max)
            self.clipped = bool(np.max(np.abs(cl - p)) > 0.005)   # > 5 mm outside the box: really out of reach
            p = cl
        return p


class DropWatch:
    """Latches the 'right arm dropped' release: the control wrist stays below the hip line for `hold_s`. Reused by
    the app in plane mode (the left-hand-raise clutch is disabled there, so a lowered control arm is the stow cue)."""

    def __init__(self, hold_s: float = 0.7):
        self.hold_s = float(hold_s)
        self._t_down = None

    def update(self, below_hip: bool, now: float) -> bool:
        if not below_hip:
            self._t_down = None
            return False
        if self._t_down is None:
            self._t_down = now
        return now - self._t_down >= self.hold_s

    def reset(self):
        self._t_down = None

"""Measured teleop calibration: sync YOUR frame of reference with the robot / the Isaac view.

1. Body frame (camera-independent), from MediaPipe world landmarks:
     x = right shoulder - left shoulder (your right), z = hips -> shoulders (up; fallback: camera up if no hips),
     y = z cross x (your FORWARD = toward the webcam when you face it)      [x cross z would point backward]
   hand = right wrist - right shoulder, expressed in that frame. Webcam position/tilt no longer matter.
2. Guided capture: NEUTRAL 2 s, then RIGHT / UP / FORWARD ("reach toward the camera") / LEFT / DOWN held 1.5 s each
   (BACK optional). A hold is rejected if the hand jitters > 2 cm, or moved < 8 cm from neutral.
3. Fit: rotation R (body -> command frame) by Kabsch on the unit directions; the command frame is the CURRENT Isaac
   camera's HEADING with gravity up: screen-right and forward are horizontal, up is +z. Per-direction scales map your
   comfortable reach onto the reachable box (ray from the box centre to the box wall along that axis). Neutral = the
   box centre. Mirroring (a proper rotation of the torso frame), camera yaw and arm length are absorbed.
   If the camera changes, the command frame is recomputed; no re-capture.
4. Verification: touch 4 ghost targets; per-target error (cm) + the fit's residual angle; accept if all < 3 cm.
5. Profile JSON (R, scales, neutral, residuals, date, camera, arm) reused next session; a 3 s re-centre updates neutral.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DIRS = {  # capture direction -> (command axis index, sign). Command axes: 0 = screen-right, 1 = forward, 2 = up
    "right": (0, +1), "left": (0, -1), "forward": (1, +1), "back": (1, -1), "up": (2, +1), "down": (2, -1)}
PROMPTS = {
    "neutral": "Relax: hand in front of your chest, hold STILL",
    "right": "Move your hand to your RIGHT and hold",
    "left": "Move your hand to your LEFT and hold",
    "up": "Raise your hand UP and hold",
    "down": "Lower your hand DOWN and hold",
    "forward": "Reach toward the CAMERA and hold",
    "back": "Pull your hand BACK toward your body and hold",
}
DEFAULT_STEPS = ["neutral", "right", "up", "forward", "left", "down"]


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def torso_frame(pose):
    """-> (R_cam_to_body 3x3, origin=right shoulder) or None. pose: BodyPose (shoulder = tracked RIGHT shoulder)."""
    if pose.shoulder_other is None:
        return None
    rs, ls = np.asarray(pose.shoulder, float), np.asarray(pose.shoulder_other, float)
    x = _unit(rs - ls) if pose.side == "right" else _unit(ls - rs)        # your right
    if pose.hips is not None:
        up = (rs + ls) / 2 - np.asarray(pose.hips, float)
    else:
        up = np.array([0.0, -1.0, 0.0])                                   # MediaPipe y points down
    z = _unit(up - np.dot(up, x) * x)
    y = np.cross(z, x)                                                    # forward (toward the camera)
    return np.stack([x, y, z]), rs


def hand_in_body(pose):
    tf = torso_frame(pose)
    if tf is None:
        return None
    R, origin = tf
    return R @ (np.asarray(pose.wrist, float) - origin)


def command_frame(cam_eye, cam_target):
    """Columns: screen-right, forward, up, in robot coords; from the camera HEADING only (gravity stays up)."""
    f = np.asarray(cam_target, float) - np.asarray(cam_eye, float)
    f[2] = 0.0
    fwd = _unit(f)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    return np.stack([right, fwd, up], axis=1)


def kabsch(src, dst, w=None):
    """Rotation R minimising sum w |R src_i - dst_i|^2 (proper rotation, det +1)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    H = (src * w[:, None]).T @ dst
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def ray_to_box(center, direction, lo, hi):
    """Distance from `center` along unit `direction` to the box wall."""
    t = np.inf
    for k in range(3):
        if abs(direction[k]) > 1e-9:
            wall = hi[k] if direction[k] > 0 else lo[k]
            t = min(t, (wall - center[k]) / direction[k])
    return float(t)


@dataclass
class Profile:
    R: list                      # body -> command (3x3)
    scale: dict                  # direction -> robot m per human m
    neutral: list                # hand-in-body at neutral (m)
    reach: dict                  # direction -> your measured reach (m)
    residual_deg: float
    per_dir_deg: dict
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    camera: dict | None = None
    arm_profile: str | None = None
    verify: dict | None = None

    def map(self, hand_body, cmd_frame, lo, hi, gain_boost: float = 1.0):
        """hand (body frame) -> robot target, clamped to the box. Returns (target, clamped)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        center = (lo + hi) / 2
        c = np.asarray(self.R) @ (np.asarray(hand_body, float) - np.asarray(self.neutral))   # command-frame metres
        out = np.zeros(3)
        for k, (pos, neg) in enumerate([("right", "left"), ("forward", "back"), ("up", "down")]):
            key = pos if c[k] >= 0 else neg
            s = self.scale.get(key) or self.scale.get(pos if key == neg else neg)
            out[k] = c[k] * s * gain_boost
        p = center + np.asarray(cmd_frame) @ out
        q = np.clip(p, lo, hi)
        return q, bool(np.max(np.abs(q - p)) > 0.005)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path):
        return Profile(**json.loads(Path(path).read_text()))


def fit(holds: dict, cmd_frame, lo, hi, arm_profile=None, camera=None) -> Profile:
    """holds: {"neutral": hand, "right": hand, ...} (hand-in-body means, metres)."""
    n = np.asarray(holds["neutral"], float)
    names = [k for k in holds if k in DIRS]
    src = np.array([_unit(np.asarray(holds[k]) - n) for k in names])
    dst = np.array([DIRS[k][1] * np.eye(3)[DIRS[k][0]] for k in names])
    R = kabsch(src, dst)
    per_dir = {k: float(np.degrees(np.arccos(np.clip(np.dot(R @ s, d), -1, 1)))) for k, s, d in zip(names, src, dst)}
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    center = (lo + hi) / 2
    reach, scale = {}, {}
    for k in names:
        ax, sign = DIRS[k]
        reach[k] = float(abs((R @ (np.asarray(holds[k]) - n))[ax]))
        wall = ray_to_box(center, sign * np.asarray(cmd_frame)[:, ax], lo, hi)
        scale[k] = float(wall / max(reach[k], 0.05))
    return Profile(R=R.tolist(), scale=scale, neutral=n.tolist(), reach=reach,
                   residual_deg=float(np.mean(list(per_dir.values()))), per_dir_deg=per_dir,
                   camera=camera, arm_profile=arm_profile)


@dataclass
class Capture:
    """Guided capture state machine. Feed hand-in-body samples; read .prompt / .progress / .done / .holds."""
    steps: list = field(default_factory=lambda: list(DEFAULT_STEPS))
    neutral_s: float = 2.0
    hold_s: float = 1.5
    jitter_m: float = 0.02
    min_move_m: float = 0.08
    step_timeout_s: float = 20.0
    i: int = 0
    holds: dict = field(default_factory=dict)
    note: str = ""
    _buf: list = field(default_factory=list)
    _t0: float | None = None
    _t_step: float | None = None

    @property
    def step(self):
        return self.steps[self.i] if self.i < len(self.steps) else None

    @property
    def done(self):
        return self.i >= len(self.steps)

    @property
    def prompt(self):
        return PROMPTS.get(self.step, "") if not self.done else "Calibration captured"

    @property
    def progress(self):
        if self._t0 is None or self.done:
            return 0.0
        need = self.neutral_s if self.step == "neutral" else self.hold_s
        return min(1.0, (self._buf[-1][0] - self._t0) / need) if self._buf else 0.0

    def update(self, now, hand):
        """-> event string or None ('held:<step>', 'rejected:<why>', 'done', 'timeout:<step>')."""
        if self.done or hand is None:
            return None
        if self._t_step is None:
            self._t_step = now
        hand = np.asarray(hand, float)
        k = self.step
        if k != "neutral":
            d = np.linalg.norm(hand - np.asarray(self.holds["neutral"]))
            if d < self.min_move_m:                 # not there yet: keep waiting (no hold timer)
                self._buf, self._t0 = [], None
                self.note = "move further"
                if now - self._t_step > self.step_timeout_s:
                    return self._skip(now, "timeout")
                return None
        self.note = ""
        self._buf.append((now, hand))
        if self._t0 is None:
            self._t0 = now
        pts = np.array([h for _, h in self._buf])
        if np.max(np.linalg.norm(pts - pts.mean(0), axis=1)) > self.jitter_m:
            self._buf, self._t0 = [(now, hand)], now            # too shaky: restart this hold
            self.note = "too shaky - hold still"
            return "rejected:jitter"
        need = self.neutral_s if k == "neutral" else self.hold_s
        if now - self._t0 >= need:
            self.holds[k] = pts.mean(0).tolist()
            self.i += 1
            self._buf, self._t0, self._t_step = [], None, None
            return "done" if self.done else f"held:{k}"
        return None

    def _skip(self, now, why):
        k = self.step
        self.i += 1
        self._buf, self._t0, self._t_step = [], None, None
        return f"{why}:{k}"

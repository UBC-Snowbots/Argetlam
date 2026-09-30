"""60-second guided warm-up: checks every axis responds the right way, and auto-detects a mirrored webcam.

Steps (each passes when the ROBOT target moves the expected way by `min_move` from where the step started;
a step that doesn't pass within `step_s` is marked failed and the warm-up moves on):
  mirror  "Raise your RIGHT hand above your shoulder" -> if the OTHER tracked hand rises instead, the webcam is
          mirrored: `mirror_detected` is set and the app tells the body client to flip left/right.
  calib   "Hold still" (the clutch calibrates; passes once following)
  right / left / up / down / reach : move the hand that way -> gripper target moves that way (robot -y/+y/+z/-z/+x)
  fist / open : gripper closes / opens
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STEPS = [
    ("mirror", "Raise your RIGHT hand above your shoulder", "↑R"),
    ("calib", "Hold still while it calibrates", "●"),
    ("right", "Move your hand to your RIGHT", "→"),
    ("left", "Now to your LEFT", "←"),
    ("up", "Raise your hand UP", "↑"),
    ("down", "Lower your hand DOWN", "↓"),
    ("reach", "REACH toward the screen", "⊙"),
    ("fist", "Make a FIST (gripper closes)", "✊"),
    ("open", "OPEN your hand (gripper opens)", "✋"),
]
AXIS = {"right": (1, -1), "left": (1, +1), "up": (2, +1), "down": (2, -1), "reach": (0, +1)}


@dataclass
class Coach:
    min_move: float = 0.06
    step_s: float = 12.0
    raise_m: float = 0.08
    active: bool = False
    i: int = 0
    results: dict = field(default_factory=dict)
    mirror_detected: bool | None = None
    _t0: float = 0.0
    _ref: np.ndarray | None = None
    _y0: tuple | None = None

    def start(self, now):
        self.active, self.i, self.results, self.mirror_detected = True, 0, {}, None
        self._begin(now)

    def stop(self):
        self.active = False

    @property
    def step(self):
        return STEPS[self.i] if self.active and self.i < len(STEPS) else None

    def view(self, now):
        """For the HUD / client: current prompt, arrow, seconds left, results so far."""
        if not self.active:
            return None
        k, text, arrow = STEPS[self.i]
        return {"step": k, "text": text, "arrow": arrow, "n": self.i + 1, "of": len(STEPS),
                "left_s": max(0.0, round(self.step_s - (now - self._t0), 1)), "results": dict(self.results),
                "mirror": self.mirror_detected}

    def update(self, now, pose=None, target=None, grip=None, clutch_state=None):
        """Feed every body sample. Returns an event dict when something happens (step passed/failed, mirror)."""
        if not self.active:
            return None
        k = STEPS[self.i][0]
        ev = None
        if k == "mirror" and pose is not None and pose.wrist_other is not None:
            if self._y0 is None:
                self._y0 = (pose.wrist[1], pose.wrist_other[1])
            up_tracked = self._y0[0] - pose.wrist[1]          # MediaPipe y points down: up = y decreases
            up_other = self._y0[1] - pose.wrist_other[1]
            if up_tracked > self.raise_m and up_tracked > up_other:
                self.mirror_detected, ev = False, self._next(now, True)
            elif up_other > self.raise_m and up_other > up_tracked:
                self.mirror_detected, ev = True, self._next(now, True)   # the "left" arm moved: image is mirrored
                ev["mirror"] = True
        elif k == "calib":
            if clutch_state == "following":
                ev = self._next(now, True)
        elif k in AXIS and target is not None:
            if self._ref is None:
                self._ref = np.asarray(target, float)
            ax, sign = AXIS[k]
            if sign * (target[ax] - self._ref[ax]) >= self.min_move:
                ev = self._next(now, True)
        elif k == "fist" and grip is not None and grip >= 0.5:
            ev = self._next(now, True)
        elif k == "open" and grip is not None and grip < 0.5:
            ev = self._next(now, True)
        if ev is None and now - self._t0 > self.step_s:
            ev = self._next(now, False)
        return ev

    def _begin(self, now):
        self._t0, self._ref, self._y0 = now, None, None

    def _next(self, now, ok):
        k = STEPS[self.i][0]
        self.results[k] = ok
        self.i += 1
        if self.i >= len(STEPS):
            self.active = False
            return {"step": k, "ok": ok, "done": True, "passed": sum(self.results.values()), "of": len(STEPS)}
        self._begin(now)
        return {"step": k, "ok": ok, "done": False}

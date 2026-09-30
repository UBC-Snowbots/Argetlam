"""Clutch + calibration for body teleop.

States: idle (arm frozen) -> calibrating (hold still `calib_s`) -> following -> idle ...
Engage (from idle):
  * F key in the body client (a "toggle" event; OpenCV can't see key-up, so F toggles)
  * open palm held UP (wrist above elbow) for `palm_s`  (a relaxed, hanging open hand doesn't count)
  * raise your LEFT hand above your left shoulder -> hold-to-follow: lowering it releases (dead-man)
Release (from calibrating/following): F toggle again, or lowering the left hand if it was the engage trigger.
Calibration: the wrist-relative-to-shoulder vector must stay within `still_m` of its running mean for `calib_s`;
if you move, the countdown restarts. The neutral pose = the mean over the countdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClutchEvent:
    kind: str                 # "engaged" (calibration done: start following), "released", "calibrating"
    neutral: np.ndarray | None = None


@dataclass
class Clutch:
    calib_s: float = 2.0
    still_m: float = 0.03
    palm_s: float = 1.0
    raise_m: float = 0.05
    state: str = "idle"                     # idle | calibrating | following
    trigger: str | None = None              # "toggle" | "palm" | "left_hand"
    release_reason: str | None = None       # why the last release happened (shown in the HUD)
    progress: float = 0.0                   # calibration 0..1
    _samples: list = field(default_factory=list)
    _t_calib: float | None = None
    _t_palm: float | None = None
    _t_start: float = 0.0
    raise_on_s: float = 0.3       # left hand must be up this long to engage (debounce: detection flickers)
    raise_off_s: float = 0.5      # ... and down this long to release (a flicker killed every hold, 2026-09-30)
    _t_up: float | None = None
    _t_down: float | None = None

    def toggle(self, now: float) -> ClutchEvent | None:
        if self.state == "idle":
            return self._start("toggle", now)
        return self._release()

    def update(self, rel: np.ndarray | None, palm_up_open: bool, left_raised: bool, now: float) -> ClutchEvent | None:
        """Call for every body sample. rel = wrist - shoulder (None if the arm isn't visible)."""
        # debounce the left-hand signal
        if left_raised:
            self._t_up = now if self._t_up is None else self._t_up
            self._t_down = None
        else:
            self._t_down = now if self._t_down is None else self._t_down
            self._t_up = None
        left_up = self._t_up is not None and now - self._t_up >= self.raise_on_s
        left_down = self._t_down is not None and now - self._t_down >= self.raise_off_s
        if self.state == "idle":
            if left_up:
                return self._start("left_hand", now)
            if palm_up_open:
                self._t_palm = now if self._t_palm is None else self._t_palm
                if now - self._t_palm >= self.palm_s:
                    return self._start("palm", now)
            else:
                self._t_palm = None
            return None
        if self.trigger == "left_hand" and left_down:
            return self._release("left hand lowered")
        if self.state == "calibrating":
            if rel is None:
                return None
            self._samples.append(np.asarray(rel, float))
            mean = np.mean(self._samples, axis=0)
            if np.linalg.norm(self._samples[-1] - mean) > self.still_m:   # moved: restart the countdown
                self._samples, self._t_calib = [np.asarray(rel, float)], now
            self.progress = min(1.0, (now - self._t_calib) / self.calib_s)
            if self.progress >= 1.0:
                self.state, self.progress = "following", 1.0
                return ClutchEvent("engaged", np.mean(self._samples, axis=0))
        return None

    def watchdog(self, now: float, last_packet_t: float, last_visible_t: float,
                 link_s: float = 0.5, track_s: float = 1.0) -> ClutchEvent | None:
        """Safety: auto-release (freeze the arm) if body packets stop or the arm isn't visible while engaged.
        Re-engaging recalibrates, so the arm never jumps when tracking comes back."""
        if self.state == "idle":
            return None
        if now - max(last_packet_t, self._t_start) > link_s:      # grace from the moment the clutch started
            return self._release("link lost")
        if now - max(last_visible_t, self._t_start) > track_s:
            return self._release("tracking lost")
        return None

    def _start(self, trigger, now):
        self.release_reason, self._t_start = None, now
        self.state, self.trigger, self.progress = "calibrating", trigger, 0.0
        self._samples, self._t_calib, self._t_palm = [], now, None
        return ClutchEvent("calibrating")

    def _release(self, reason: str | None = None):
        self.release_reason = reason
        self.state, self.trigger, self.progress = "idle", None, 0.0
        self._samples, self._t_calib, self._t_palm = [], None, None
        return ClutchEvent("released")

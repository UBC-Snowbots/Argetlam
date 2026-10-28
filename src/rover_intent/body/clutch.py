"""Clutch + calibration for body teleop.

States: idle (arm frozen) -> calibrating (hold still `calib_s`) -> following -> idle ...
A following session can also drop to a transient "paused" state when the arm leaves the camera frame (plane mode,
when `pause_on_lost` is on): the arm HOLDS its current target instead of releasing / gliding to base, and it resumes
the moment the arm is visible again for `resume_s`. A pause longer than `pause_max_s` falls through to the normal
"tracking lost" release (so arm.on_drop still stows the arm after a long absence).
Engage (from idle):
  * F key in the body client (a "toggle" event; OpenCV can't see key-up, so F toggles)
  * open palm held UP (wrist above elbow) for `palm_s`  (a relaxed, hanging open hand doesn't count)
  * raise your LEFT hand above your left shoulder -> hold-to-follow: lowering it releases (dead-man)
Release (from calibrating/following/paused): F toggle again, or lowering the left hand if it was the engage trigger.
Calibration: the wrist-relative-to-shoulder vector must stay within `still_m` of its running mean for `calib_s`;
if you move, the countdown restarts. The neutral pose = the mean over the countdown. Plane mode sets a short
`calib_s` and loose `still_m` (configs) so engaging is near-instant; arm_mirror keeps the tight still calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClutchEvent:
    # "engaged" (calibration done: start following), "released", "calibrating", "paused" (arm left view, hold the
    # target), "resumed" (arm back: re-anchor the mapper at the current TCP and keep following, no jump)
    kind: str
    neutral: np.ndarray | None = None


@dataclass
class Clutch:
    calib_s: float = 2.0
    still_m: float = 0.03
    palm_s: float = 1.0
    raise_m: float = 0.05
    state: str = "idle"                     # idle | calibrating | following | paused
    trigger: str | None = None              # "toggle" | "palm" | "left_hand"
    release_reason: str | None = None       # why the last release happened (shown in the HUD)
    progress: float = 0.0                   # calibration 0..1
    # Pause-on-tracking-loss (plane mode). When `pause_on_lost` is on, losing the arm while following parks the clutch
    # in "paused" (hold, no on_drop glide) instead of releasing; it resumes after `resume_s` of continuous visibility,
    # or releases with "tracking lost" after `pause_max_s`. Off by default so arm_mirror keeps its release behaviour.
    pause_on_lost: bool = False
    resume_s: float = 0.3
    pause_max_s: float = 8.0
    link_s: float = 0.5                     # packets stale this long -> "link lost" release (watchdog default)
    track_s: float = 1.0                    # arm not visible this long -> pause (or "tracking lost") (watchdog default)
    _samples: list = field(default_factory=list)
    _t_calib: float | None = None
    _t_palm: float | None = None
    _t_start: float = 0.0
    _t_paused: float | None = None          # when the current pause began
    _t_resume: float | None = None          # when the arm became continuously visible again during a pause
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
        if self.state == "paused":
            # the arm is back (rel not None = visible): resume after `resume_s` of continuous visibility. The app
            # re-anchors the plane mapper at the current TCP on "resumed", so the commanded target is continuous.
            if rel is not None:
                self._t_resume = now if self._t_resume is None else self._t_resume
                if now - self._t_resume >= self.resume_s:
                    self.state, self.progress = "following", 1.0
                    self._t_resume = self._t_paused = None
                    return ClutchEvent("resumed")
            else:
                self._t_resume = None
            return None
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
                 link_s: float | None = None, track_s: float | None = None) -> ClutchEvent | None:
        """Safety: while engaged, auto-react if body packets stop or the arm isn't visible.
        - packets stale > `link_s`: release immediately ("link lost"), from ANY state incl. paused (nobody watching).
        - arm not visible > `track_s` while following: if `pause_on_lost`, go to "paused" (hold, no on_drop glide),
          else release ("tracking lost", which drives arm.on_drop). Calibrating keeps releasing on a tracking loss.
        - paused longer than `pause_max_s`: fall through to the "tracking lost" release so on_drop still stows.
        Re-engaging recalibrates and resume re-anchors, so the arm never jumps when tracking comes back."""
        if self.state == "idle":
            return None
        link_s = self.link_s if link_s is None else link_s
        track_s = self.track_s if track_s is None else track_s
        if now - max(last_packet_t, self._t_start) > link_s:      # grace from the moment the clutch started
            return self._release("link lost")
        if self.state == "paused":
            if self._t_paused is not None and now - self._t_paused > self.pause_max_s:
                return self._release("tracking lost")
            return None
        if now - max(last_visible_t, self._t_start) > track_s:
            if self.pause_on_lost and self.state == "following":
                return self._pause(now)
            return self._release("tracking lost")
        return None

    def _start(self, trigger, now):
        self.release_reason, self._t_start = None, now
        self.state, self.trigger, self.progress = "calibrating", trigger, 0.0
        self._samples, self._t_calib, self._t_palm = [], now, None
        self._t_paused = self._t_resume = None
        return ClutchEvent("calibrating")

    def _pause(self, now):
        self.state, self.progress = "paused", 1.0
        self._t_paused, self._t_resume = now, None
        return ClutchEvent("paused")

    def _release(self, reason: str | None = None):
        self.release_reason = reason
        self.state, self.trigger, self.progress = "idle", None, 0.0
        self._samples, self._t_calib, self._t_palm = [], None, None
        self._t_paused = self._t_resume = None
        return ClutchEvent("released")

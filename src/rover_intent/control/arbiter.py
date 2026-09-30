"""Decides who drives the arm each tick.

Modes:
  FOLLOW  body mirroring drives the TCP; EEG (or voice) opens/closes the gripper.
          Shared autonomy: near a known object, the target is pulled toward its grasp point.
  AUTO    a skill's waypoint list drives the TCP (from a Nemotron-parsed voice command).
  HOLD    stay put (after 'stop', on start-up, or when tracking is lost).
Voice 'stop' always wins; 'follow' returns to FOLLOW.
"""
from __future__ import annotations

import time
from enum import Enum

import numpy as np

from ..planner.scene import Scene
from ..planner.skills import SkillError, Waypoint, plan
from ..types import Action, Intent


class Mode(str, Enum):
    FOLLOW = "follow"
    AUTO = "auto"
    HOLD = "hold"


class Arbiter:
    def __init__(self, scene: Scene, eeg_on: float = 0.7, eeg_off: float = 0.3, eeg_hold_s: float = 0.5,
                 assist_radius: float = 0.08, assist_gain: float = 0.6, waypoint_tol: float = 0.01,
                 waypoint_timeout: float = 8.0):
        self.scene = scene
        self.mode = Mode.HOLD
        self.gripper = 0.0
        self.held: str | None = None  # object believed to be in the gripper (set by skills)
        self.eeg_on, self.eeg_off, self.eeg_hold_s = eeg_on, eeg_off, eeg_hold_s
        self.assist_radius, self.assist_gain = assist_radius, assist_gain
        self.waypoint_tol = waypoint_tol
        self.waypoint_timeout = waypoint_timeout  # s: give up (HOLD) if a waypoint can't be reached, e.g. blocked
        self._wp_started: float | None = None
        self._plan: list[Waypoint] = []
        self._eeg_since: float | None = None
        self._wp_since: float | None = None
        self._carrot: np.ndarray | None = None   # straight-line interpolated target the IK chases
        self._carrot_t: float | None = None
        self.last_say = ""

    # --- inputs -------------------------------------------------------------------
    def on_intent(self, intent: Intent) -> None:
        self.last_say = intent.say
        a = intent.action
        if a == Action.stop:
            self.mode, self._plan, self._wp_since, self._wp_started = Mode.HOLD, [], None, None
        elif a == Action.reset:
            self.mode, self._plan, self._wp_since, self._wp_started = Mode.HOLD, [], None, None
            self.gripper, self.held, self._eeg_since, self._carrot = 0.0, None, None, None
        elif a == Action.follow:
            self.mode, self._plan = Mode.FOLLOW, []
        elif a == Action.grasp:
            self.gripper = 1.0
        elif a == Action.release:
            self.gripper, self.held = 0.0, None
        elif a in (Action.pick, Action.place, Action.move):
            try:
                self._plan = plan(intent, self.scene, held=self.held)
                self._wp_since = self._wp_started = None
                self.mode = Mode.AUTO
            except SkillError as e:
                self.last_say = f"Can't do that: {e}"

    def on_eeg(self, p_grasp: float, now: float | None = None) -> None:
        """Hysteresis + dwell time so one noisy window can't drop a cup."""
        if self.mode != Mode.FOLLOW:
            return
        now = time.time() if now is None else now
        if self.eeg_off < p_grasp < self.eeg_on:
            return  # dead band: no evidence either way, keep any pending change
        want = 1.0 if p_grasp >= self.eeg_on else 0.0
        if want == self.gripper:
            self._eeg_since = None
        elif self._eeg_since is None:
            self._eeg_since = now
        elif now - self._eeg_since >= self.eeg_hold_s:
            self.gripper, self._eeg_since = want, None

    def on_hand(self, hand_open: float, close_below: float = 0.35, open_above: float = 0.6) -> None:
        """Stand-in for EEG (config body.grip_from_hand): fist closes the gripper, open hand opens it."""
        if self.mode != Mode.FOLLOW:
            return
        if hand_open < close_below:
            self.gripper = 1.0
        elif hand_open > open_above:
            self.gripper = 0.0

    # --- output -------------------------------------------------------------------
    def target(self, tcp_now: np.ndarray, body_target: np.ndarray | None,
               now: float | None = None) -> tuple[np.ndarray | None, float]:
        """Returns (TCP position target or None = hold, gripper command)."""
        if self.mode == Mode.AUTO:
            if not self._plan:
                self.mode = Mode.HOLD
                return None, self.gripper
            now = time.time() if now is None else now
            wp = self._plan[0]
            if self._carrot is None:
                self._carrot, self._carrot_t = np.asarray(tcp_now, float).copy(), now
            step = wp.speed * max(0.0, min(0.1, now - self._carrot_t))
            self._carrot_t = now
            d = wp.pos - self._carrot
            n = np.linalg.norm(d)
            self._carrot = wp.pos.copy() if n <= step else self._carrot + d / n * step
            if self._wp_started is None:
                self._wp_started = now
            if self._wp_since is None and now - self._wp_started > self.waypoint_timeout:
                self.last_say = f"Couldn't reach {wp.label}; holding."
                self.mode, self._plan, self._wp_started = Mode.HOLD, [], None
                return None, self.gripper
            d = tcp_now - wp.pos
            reached = (np.linalg.norm(d) < self.waypoint_tol if wp.tol_xy is None
                       else np.linalg.norm(d[:2]) < wp.tol_xy and abs(d[2]) < self.waypoint_tol)
            if reached:
                self.gripper = wp.gripper
                if self._wp_since is None:
                    self._wp_since = now
                if now - self._wp_since >= wp.dwell:
                    if wp.label == "close":
                        self.held = wp.obj
                    elif wp.label == "open":
                        self.held = None
                    self._plan.pop(0)
                    self._wp_since = self._wp_started = None
            return self._carrot.copy(), self.gripper
        self._carrot = None
        if self.mode == Mode.FOLLOW and body_target is not None:
            return self._assist(body_target), self.gripper
        return None, self.gripper

    def _assist(self, p: np.ndarray) -> np.ndarray:
        if self.gripper > 0.5:
            return p  # holding something: pure teleop
        o = self.scene.nearest(p, self.assist_radius)
        if o is None:
            return p
        w = self.assist_gain * (1 - np.linalg.norm(o.pos - p) / self.assist_radius)
        return (1 - w) * p + w * o.pos

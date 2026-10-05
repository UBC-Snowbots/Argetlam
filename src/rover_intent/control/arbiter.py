"""Decides who drives the arm each tick.

Modes:
  FOLLOW  body mirroring drives the TCP; EEG (or voice) opens/closes the gripper.
          Shared autonomy: near a known object, the target is pulled toward its grasp point.
  AUTO    a skill's waypoint list drives the TCP (from a Nemotron-parsed voice command).
  HOLD    stay put (after 'stop', on start-up, or when tracking is lost).
Voice 'stop' always wins; 'follow' returns to FOLLOW.

Grab assist (FOLLOW + hand control): you steer roughly; a fist held HAND_HOLD_S with the gripper within GRAB_RADIUS
(sideways) of an object hands over to an AUTO "assist" plan (rise -> align over it -> straight descent -> close ->
short lift) that returns to FOLLOW when done. An open hand held HAND_HOLD_S while holding = open + small retreat.
"""
from __future__ import annotations

import time
from enum import Enum

import numpy as np

from ..planner.scene import Scene
from ..planner.skills import APPROACH, GRIP_DWELL, SkillError, Waypoint, plan
from ..types import Action, Intent

GRAB_RADIUS = 0.06       # m: sideways gripper->object distance at which a fist means "grab THAT"
GRAB_BELOW, GRAB_ABOVE = 0.03, 0.18   # m: gripper height window around the object's grasp point
HAND_HOLD_S = 0.4        # s: a fist / open hand must be held this long (MediaPipe openness flickers frame to frame)
ASSIST_LIFT = 0.06       # m: lift after an assisted grab
RELEASE_RETREAT = 0.05   # m: back off upward after letting go


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
        self.assist = False                 # the current AUTO plan is a grab/release assist -> back to FOLLOW after
        self.events: list[str] = []         # for the app: "assist_done" (re-engage the body mapping)
        self.z_max = np.inf                 # top of the workspace box (retreat targets stay reachable)
        self._hand_want: float | None = None
        self._hand_since: float | None = None

    # --- inputs -------------------------------------------------------------------
    def on_intent(self, intent: Intent) -> None:
        self.last_say = intent.say
        a = intent.action
        if a == Action.stop:
            self.mode, self._plan, self._wp_since, self._wp_started, self.assist = Mode.HOLD, [], None, None, False
        elif a == Action.reset:
            self.mode, self._plan, self._wp_since, self._wp_started, self.assist = Mode.HOLD, [], None, None, False
            self.gripper, self.held, self._eeg_since, self._carrot = 0.0, None, None, None
        elif a == Action.follow:
            self.mode, self._plan, self.assist = Mode.FOLLOW, [], False
        elif a == Action.grasp:
            self.gripper = 1.0
        elif a == Action.release:
            self.gripper, self.held = 0.0, None
        elif a in (Action.pick, Action.place, Action.move):
            try:
                self._plan = plan(intent, self.scene, held=self.held)
                self._wp_since = self._wp_started = None
                self.mode, self.assist = Mode.AUTO, False
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

    def on_hand(self, hand_open: float, now: float | None = None, tcp=None, close_below: float = 0.35,
                open_above: float = 0.6) -> None:
        """Stand-in for EEG (config body.grip_from_hand). A fist / open hand must be HELD for HAND_HOLD_S (hysteresis
        + dwell: openness flickers). Fist near an object = assisted grab; open hand while holding = assisted release."""
        if self.mode != Mode.FOLLOW:
            self._hand_want = self._hand_since = None
            return
        now = time.time() if now is None else now
        want = 1.0 if hand_open < close_below else 0.0 if hand_open > open_above else None
        if want is None:
            return                                   # dead band: keep any pending change
        if want == self.gripper:
            self._hand_want = self._hand_since = None
            return
        if want != self._hand_want:
            self._hand_want, self._hand_since = want, now
            return
        if now - self._hand_since < HAND_HOLD_S:
            return
        self._hand_want = self._hand_since = None
        tcp = None if tcp is None else np.asarray(tcp, float)
        if want == 1.0:
            o = self.grab_candidate(tcp) if tcp is not None else None
            if o is None:
                self.gripper, self.last_say = 1.0, "Gripper closed (nothing in reach to grab)."
                return
            pick = plan(Intent(action=Action.pick, object=o.name), self.scene, None)
            rise = np.array([tcp[0], tcp[1], max(tcp[2], o.pos[2] + APPROACH)])   # up first: never sweep sideways low
            wps = [Waypoint(rise, 0, "assist-rise")] + pick[1:4] + [Waypoint(o.pos + [0, 0, ASSIST_LIFT], 1, "lift")]
            self._start_assist(wps, f"Grabbing the {o.name}.")
        elif tcp is not None and self.held is not None:
            up = np.array([tcp[0], tcp[1], min(tcp[2] + RELEASE_RETREAT, self.z_max)])
            self._start_assist([Waypoint(tcp, 0, "open", self.held, GRIP_DWELL), Waypoint(up, 0, "retreat")],
                               f"Letting go{' of the ' + self.held if self.held else ''}.")
        else:
            self.gripper, self.held = 0.0, None

    def grab_candidate(self, tcp):
        """The object a fist would grab right now (gripper open, close enough sideways, in the height window)."""
        if self.held is not None or self.gripper > 0.5:
            return None
        tcp = np.asarray(tcp, float)
        best, best_d = None, GRAB_RADIUS
        for o in self.scene.objects.values():
            d = float(np.linalg.norm(tcp[:2] - o.pos[:2]))
            if d <= best_d and -GRAB_BELOW <= tcp[2] - o.pos[2] <= GRAB_ABOVE:
                best, best_d = o, d
        return best

    def _start_assist(self, wps, say):
        self._plan, self._wp_since, self._wp_started, self._carrot = wps, None, None, None
        self.mode, self.assist, self.last_say = Mode.AUTO, True, say

    def abort(self, say: str = "") -> None:
        """Clutch released / tracking lost: stop wherever we are (an assist must not keep running)."""
        self.mode, self._plan, self._wp_since, self._wp_started, self.assist = Mode.HOLD, [], None, None, False
        if say:
            self.last_say = say

    # --- output -------------------------------------------------------------------
    def target(self, tcp_now: np.ndarray, body_target: np.ndarray | None,
               now: float | None = None) -> tuple[np.ndarray | None, float]:
        """Returns (TCP position target or None = hold, gripper command)."""
        if self.mode == Mode.AUTO:
            if not self._plan:
                if self.assist:   # assisted grab/release finished: hand control back to the body mapping
                    self.mode, self.assist = Mode.FOLLOW, False
                    self.events.append("assist_done")
                    self.last_say = f"Got the {self.held}." if self.held else "Released."
                else:
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
                self.mode, self._plan, self._wp_started, self.assist = Mode.HOLD, [], None, False
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
        """Shared autonomy while steering: pull SIDEWAYS toward the object below, only when above its grasp point.
        (Pulling toward the grasp point in 3D dragged the fingers into objects from the side and tipped them.)"""
        if self.gripper > 0.5:
            return p  # holding something: pure teleop
        o = min(self.scene.objects.values(), key=lambda o: np.linalg.norm(o.pos[:2] - p[:2]), default=None)
        if o is None or p[2] < o.pos[2] + GRAB_BELOW:
            return p
        d = float(np.linalg.norm(o.pos[:2] - p[:2]))
        if d > self.assist_radius:
            return p
        w = self.assist_gain * (1 - d / self.assist_radius)
        q = p.copy()
        q[:2] = (1 - w) * p[:2] + w * o.pos[:2]
        return q

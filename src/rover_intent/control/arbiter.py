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
HAND_HOLD_S = 0.6        # s: a fist / open hand must be held this long (MediaPipe openness flickers frame to frame)
ASSIST_LIFT = 0.06       # m: lift after an assisted grab
RELEASE_RETREAT = 0.05   # m: back off upward after letting go


class Mode(str, Enum):
    FOLLOW = "follow"
    AUTO = "auto"
    HOLD = "hold"


class Arbiter:
    def __init__(self, scene: Scene, eeg_on: float = 0.7, eeg_off: float = 0.3, eeg_hold_s: float = 0.5,
                 assist_radius: float = 0.08, assist_gain: float = 0.6, waypoint_tol: float = 0.01,
                 waypoint_timeout: float = 8.0, grip_sources=("hand", "eeg"), coalesce_s: float = 0.6):
        self.scene = scene
        self.mode = Mode.HOLD
        self.gripper = 0.0
        # Dual-source gripper: both the hand (EDGE events: open->fist = CLOSE, fist->open = OPEN) and the Muse
        # (one-shot GripToggle = FLIP) feed one gripper state, LAST EVENT WINS. grip_source records who made the
        # last change ("hand" / "muse" / "eeg" / "voice" / "skill"); grip_sources restricts which inputs are live
        # (gripper.sources config; body.grip_from_hand is a backward-compatible alias for including "hand").
        self.grip_sources = set(grip_sources)
        self.grip_source: str | None = None
        self.coalesce_s = float(coalesce_s)        # gripper.coalesce_s: window in which hand + Muse are reconciled
        self.grip_clashes = 0                      # count of opposite-intent cross-source events inside the window
        self.grip_clash_log: list[tuple] = []      # (t, earlier_source, winner_source, direction)
        self._grip_event_t: float | None = None    # time of the last hand/Muse gripper event
        self._grip_event_dir: str | None = None     # "close" / "open" of that event
        self._grip_event_src: str | None = None     # "hand" / "muse" of that event
        self._grip_before = 0.0                     # gripper state just BEFORE the last event (for the coalescing rule)
        self._hand_state: str | None = None         # debounced hand posture: "fist" / "open" / None (unknown)
        self.grip_toggle_in_hold = False   # eeg.grip_toggle_in_hold: a clench also opens/closes the gripper in HOLD
        self.grip_auto_open_s = 0.0        # eeg.grip_auto_open_s: > 0 = a clench-closed gripper (HOLD) reopens after this
        self._auto_open_at: float | None = None
        self.held: str | None = None  # object believed to be in the gripper (set by skills)
        self.eeg_on, self.eeg_off, self.eeg_hold_s = eeg_on, eeg_off, eeg_hold_s
        self.assist_radius, self.assist_gain = assist_radius, assist_gain
        self.waypoint_tol = waypoint_tol
        self.stall_tol = 0.0        # m: > 0 lets a waypoint that stalls this close count as reached (skills.stall_tol)
        self._stall_ref = None
        self.waypoint_timeout = waypoint_timeout  # s: give up (HOLD) if a waypoint can't be reached, e.g. blocked
        self._wp_started: float | None = None
        self._plan: list[Waypoint] = []
        self._eeg_since: float | None = None
        self._wp_since: float | None = None
        self._carrot: np.ndarray | None = None   # straight-line interpolated target the IK chases
        self._carrot_t: float | None = None
        self.say_log: list[str] = []   # new say-messages, in order; the app drains and logs them once per tick (issue 9)
        self._last_say = ""
        self.assist = False                 # the current AUTO plan is a grab/release assist -> back to FOLLOW after
        self.events: list[str] = []         # for the app: "assist_done" (re-engage the body mapping)
        self.z_max = np.inf                 # top of the workspace box (retreat targets stay reachable)
        self._hand_want: float | None = None
        self._hand_since: float | None = None
        self._hand_ema: float | None = None   # smoothed hand openness (issue 5): cuts the frame-to-frame flicker

    @property
    def last_say(self) -> str:
        return self._last_say

    @last_say.setter
    def last_say(self, value: str) -> None:
        """Every distinct, non-empty say is pushed onto say_log so the app logs it once, when it happens (issue 9)."""
        if value and value != self._last_say:
            self.say_log.append(value)
        self._last_say = value

    # --- inputs -------------------------------------------------------------------
    def on_intent(self, intent: Intent) -> None:
        self.last_say = intent.say
        a = intent.action
        if a == Action.stop:
            self.mode, self._plan, self._wp_since, self._wp_started, self.assist = Mode.HOLD, [], None, None, False
        elif a == Action.reset:
            self.mode, self._plan, self._wp_since, self._wp_started, self.assist = Mode.HOLD, [], None, None, False
            self.gripper, self.held, self._eeg_since, self._carrot = 0.0, None, None, None
            self.grip_source, self._grip_event_t, self._grip_event_dir, self._grip_event_src = None, None, None, None
        elif a == Action.follow:
            self.mode, self._plan, self.assist = Mode.FOLLOW, [], False
        elif a == Action.grasp:
            self.gripper, self.grip_source = 1.0, "voice"
        elif a == Action.release:
            self.gripper, self.held, self.grip_source = 0.0, None, "voice"
        elif a in (Action.pick, Action.place, Action.move):
            try:
                self._plan = plan(intent, self.scene, held=self.held)
                self._wp_since = self._wp_started = None
                self.mode, self.assist = Mode.AUTO, False
            except SkillError as e:
                self.last_say = f"Can't do that: {e}"

    def on_eeg(self, p_grasp: float, now: float | None = None) -> None:
        """Hysteresis + dwell time so one noisy window can't drop a cup."""
        if "eeg" not in self.grip_sources or self.mode != Mode.FOLLOW:
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
            self.gripper, self._eeg_since, self.grip_source = want, None, "eeg"

    def on_hand(self, hand_open: float, now: float | None = None, tcp=None, close_below: float = 0.30,
                open_above: float = 0.65, ema_alpha: float = 0.5) -> None:
        """Hand gripper source (config gripper.sources includes "hand"; body.grip_from_hand is the alias). The hand
        feeds EDGE events: a debounced open->fist transition is a CLOSE event, fist->open is an OPEN event. A held
        posture no longer re-sends its level every frame, so a sustained fist cannot fight a Muse toggle that opened
        the gripper (it stays open until the hand opens and closes again). The debounce is unchanged: openness is
        EMA-smoothed (issue 5), the band is 0.30/0.65 so a half-open hand sits in the dead zone, and the new posture
        must be held HAND_HOLD_S before the edge fires. Fist near an object = assisted grab; open hand while holding =
        assisted release. The smoothed openness and the debounced posture reset whenever we are not FOLLOWing."""
        if "hand" not in self.grip_sources:
            return
        if self.mode != Mode.FOLLOW:
            self._hand_want = self._hand_since = self._hand_ema = self._hand_state = None
            return
        now = time.time() if now is None else now
        self._hand_ema = hand_open if self._hand_ema is None else ema_alpha * hand_open + (1 - ema_alpha) * self._hand_ema
        h = self._hand_ema
        posture = "fist" if h < close_below else "open" if h > open_above else None
        if posture is None:
            return                                   # dead band: keep any pending change
        if posture == self._hand_state:
            self._hand_want = self._hand_since = None
            return
        if posture != self._hand_want:              # a new candidate posture: start the dwell
            self._hand_want, self._hand_since = posture, now
            return
        if now - self._hand_since < HAND_HOLD_S:
            return
        self._hand_want = self._hand_since = None
        self._hand_state = posture                  # debounced edge confirmed
        self._emit_hand_edge(posture, tcp, now)

    def _emit_hand_edge(self, posture: str, tcp=None, now: float | None = None) -> None:
        """open->fist = CLOSE event, fist->open = OPEN event. The first stable posture after engaging is a baseline
        edge too: it sets the gripper to match the hand (a fist held at engage closes, as before)."""
        self._commit_grip(1.0 if posture == "fist" else 0.0, tcp, source="hand", now=now)

    def on_grip_toggle(self, tcp=None, now: float | None = None) -> None:
        """Muse one-shot toggle (double jaw clench, eeg/clench.py): FLIP the gripper. The detector already debounced
        it, so no dwell here. Near an object = assisted grab; holding = assisted release. CLASH POLICY: a toggle that
        lands within coalesce_s AFTER a hand edge is read relative to the gripper state BEFORE that hand edge, so a
        fist (CLOSE) plus a near-simultaneous clench is one CLOSE, not a CLOSE then a flip back to open."""
        if "eeg" not in self.grip_sources:
            return
        now = time.time() if now is None else now
        if self.mode == Mode.HOLD and self.grip_toggle_in_hold:
            # Muse-only test (eeg.grip_toggle_in_hold): plain open/close in place, no assisted grab, no arm motion
            self.gripper = 0.0 if self.gripper > 0.5 else 1.0
            self.grip_source = "muse"
            self.last_say = "Gripper closed." if self.gripper > 0.5 else "Gripper open."
            self._auto_open_at = (time.time() + self.grip_auto_open_s
                                  if self.gripper > 0.5 and self.grip_auto_open_s > 0 else None)
            return
        if self.mode != Mode.FOLLOW:
            return
        tcp_arr = None if tcp is None else np.asarray(tcp, float)
        if tcp_arr is not None and (self.held is not None or self.grab_candidate(tcp_arr) is not None):
            # an object in reach (or something held): assisted grab / release, unchanged (no coalescing, no clash)
            closed = self.gripper > 0.5 or self.held is not None
            self._commit_grip(0.0 if closed else 1.0, tcp, source="muse", now=now)
            return
        within_hand = (self._grip_event_t is not None and now - self._grip_event_t < self.coalesce_s
                       and self._grip_event_src == "hand")
        base = self._grip_before if within_hand else self.gripper
        self._commit_grip(0.0 if base > 0.5 else 1.0, tcp, source="muse", now=now)

    def _commit_grip(self, want: float, tcp=None, source: str = "hand", now: float | None = None) -> None:
        tcp = None if tcp is None else np.asarray(tcp, float)
        if want == 1.0:
            o = self.grab_candidate(tcp) if tcp is not None else None
            if o is None:
                self._grip_event(1.0, source, "close", now)
                self.last_say = "Gripper closed (nothing in reach to grab)."
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
            self.held = None
            self._grip_event(0.0, source, "open", now)

    def _grip_event(self, target: float, source: str, direction: str, now: float | None = None) -> None:
        """Apply one gripper event (hand edge or Muse toggle) with the coalescing clash policy. LAST EVENT WINS,
        except inside coalesce_s: same-direction events coalesce to one (idempotent, no double action), and
        opposite-direction cross-source events are a logged CLASH where the later event wins."""
        now = time.time() if now is None else now
        within = self._grip_event_t is not None and self._grip_event_dir is not None \
            and now - self._grip_event_t < self.coalesce_s
        if within and direction == self._grip_event_dir:
            self.gripper, self._grip_event_t = target, now   # same intent inside the window: no-op action, keep source
            return
        if within and direction != self._grip_event_dir:
            if source != self._grip_event_src:               # opposite intents, two different sources: a clash
                self.grip_clashes += 1
                self.grip_clash_log.append((round(now, 3), self._grip_event_src, source, direction))
                self.last_say = f"Grip clash: {self._grip_event_src} vs {source}; {source} wins ({direction})."
        self._grip_before = self.gripper
        self.gripper, self.grip_source = target, source
        self._grip_event_t, self._grip_event_dir, self._grip_event_src = now, direction, source

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
        # an assist leaves FOLLOW: drop the debounced hand posture so, on hand-back, the first posture is a fresh
        # baseline edge (a held fist/open after the assist re-arms the grab/release instead of being swallowed)
        self._hand_want = self._hand_since = self._hand_state = None

    def abort(self, say: str = "") -> None:
        """Clutch released / tracking lost: stop wherever we are (an assist must not keep running)."""
        self.mode, self._plan, self._wp_since, self._wp_started, self.assist = Mode.HOLD, [], None, None, False
        self._hand_want = self._hand_since = self._hand_ema = self._hand_state = None  # leaving FOLLOW: drop hand state
        if say:
            self.last_say = say

    # --- output -------------------------------------------------------------------
    def target(self, tcp_now: np.ndarray, body_target: np.ndarray | None,
               now: float | None = None) -> tuple[np.ndarray | None, float]:
        """Returns (TCP position target or None = hold, gripper command)."""
        if self._auto_open_at is not None and (time.time() if now is None else now) >= self._auto_open_at:
            self._auto_open_at = None   # clench-closed gripper: open again after eeg.grip_auto_open_s
            if self.mode == Mode.HOLD and self.gripper > 0.5:
                self.gripper, self.grip_source, self.last_say = 0.0, "muse", "Gripper open."
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
                if self.assist:
                    # a failed grab/release assist must NOT leave the arbiter in HOLD while the clutch still says
                    # "following" (that left the arm stuck). Hand control back to the body mapping instead.
                    self.last_say = f"Couldn't reach {wp.label}; back to following."
                    self.mode, self._plan, self._wp_since, self._wp_started, self.assist = \
                        Mode.FOLLOW, [], None, None, False
                    self.events.append("assist_done")
                else:
                    self.last_say = f"Couldn't reach {wp.label}; holding."
                    self.mode, self._plan, self._wp_started, self.assist = Mode.HOLD, [], None, False
                return None, self.gripper
            d = tcp_now - wp.pos
            reached = (np.linalg.norm(d) < self.waypoint_tol if wp.tol_xy is None
                       else np.linalg.norm(d[:2]) < wp.tol_xy and abs(d[2]) < self.waypoint_tol)
            # Stalled close enough (stall_tol > 0, the real arm): the sim gripper can rest on the sim object a cm or two
            # above the grip point while the real one is already around the real object (2026-10-28: the real arm got
            # to the block, the sim stalled 2 cm up and the pick timed out without closing). If the TCP has not moved
            # > 2 mm for 1.5 s within stall_tol of the waypoint, count it as reached.
            if not reached and self.stall_tol > 0 and np.linalg.norm(d) < self.stall_tol:
                tcp_a = np.asarray(tcp_now, float)
                if self._stall_ref is None or np.linalg.norm(tcp_a - self._stall_ref[0]) > 0.002:
                    self._stall_ref = (tcp_a.copy(), now)
                elif now - self._stall_ref[1] >= 1.5:
                    reached = True
                    self.say_log.append(f"{wp.label}: stalled {100 * np.linalg.norm(d):.1f} cm short, continuing")
            else:
                self._stall_ref = None
            if reached:
                if wp.gripper != self.gripper:
                    self.grip_source = "skill"
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

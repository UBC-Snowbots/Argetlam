"""Dual-source gripper: the hand (open->fist = CLOSE edge, fist->open = OPEN edge) and the Muse (GripToggle = FLIP)
feed one gripper state, LAST EVENT WINS, with a coalescing CLASH window (gripper.coalesce_s) so a fist plus a
near-simultaneous clench is one CLOSE, not a flip back to open. See src/rover_intent/control/arbiter.py."""
import numpy as np

from rover_intent.control.arbiter import Arbiter, HAND_HOLD_S, Mode
from rover_intent.planner.scene import Scene
from rover_intent.types import Action, Intent

SCENE = {"cup": {"pos": [0.55, 0.15, 0.10]}, "bottle": {"pos": [0.55, -0.15, 0.12], "height": 0.22}}
FAR = np.array([0.40, 0.0, 0.30])   # nothing within grab range -> plain gripper (no assist), so coalescing applies


def follow(**kw):
    a = Arbiter(Scene(SCENE), **kw)
    a.on_intent(Intent(action=Action.follow))
    return a


def _stream(a, value, t0, tcp, dur=HAND_HOLD_S + 0.3, n=14):
    """Stream a steady hand openness for `dur` seconds so the EMA settles and the posture is held past the dwell."""
    for i in range(n):
        a.on_hand(value, t0 + dur * i / (n - 1), tcp)


def fist_edge(a, t0, tcp=FAR):
    """Drive one debounced CLOSE edge: a fist posture held past HAND_HOLD_S. Edge fires near t0 + HAND_HOLD_S."""
    _stream(a, 0.02, t0, tcp)


def open_edge(a, t0, tcp=FAR):
    _stream(a, 0.98, t0, tcp)


# --- single source ------------------------------------------------------------------------------------------------
def test_hand_edges_only():
    a = follow()
    fist_edge(a, 0.0)
    assert a.gripper == 1.0 and a.grip_source == "hand"
    open_edge(a, 2.0)
    assert a.gripper == 0.0 and a.grip_source == "hand"
    fist_edge(a, 4.0)
    assert a.gripper == 1.0 and a.grip_source == "hand"
    assert a.grip_clashes == 0


def test_muse_toggles_only():
    a = follow()
    a.on_grip_toggle(FAR, now=0.0)
    assert a.gripper == 1.0 and a.grip_source == "muse"
    a.on_grip_toggle(FAR, now=1.0)            # well outside coalesce_s: a normal flip
    assert a.gripper == 0.0 and a.grip_source == "muse"
    a.on_grip_toggle(FAR, now=2.0)
    assert a.gripper == 1.0
    assert a.grip_clashes == 0


def test_hand_flicker_does_not_spam_and_fires_once():
    a = follow()
    # alternate saturated fist/open in blocks shorter than the dwell: the pending posture keeps flipping, so neither
    # is ever held HAND_HOLD_S and no edge fires
    t = 0.0
    for block in range(6):
        value = 0.02 if block % 2 == 0 else 0.98
        for _ in range(6):
            a.on_hand(value, t, FAR)
            t += 0.033
    assert a.gripper == 0.0 and a.grip_source is None
    # a steady fist fires exactly one CLOSE edge, then holding it sends nothing more
    seen, prev = 0, a.gripper
    for i in range(60):
        a.on_hand(0.02, t + i / 30.0, FAR)
        if a.gripper != prev:
            seen += 1
            prev = a.gripper
    assert seen == 1 and a.gripper == 1.0


# --- interleaved --------------------------------------------------------------------------------------------------
def test_muse_closes_then_hand_opens():
    a = follow()
    a.on_grip_toggle(FAR, now=0.0)
    assert a.gripper == 1.0 and a.grip_source == "muse"
    open_edge(a, 2.0)                          # hand opens, long after the window
    assert a.gripper == 0.0 and a.grip_source == "hand"


def test_held_fist_plus_muse_open_stays_open_until_next_hand_edge():
    a = follow()
    fist_edge(a, 0.0)
    assert a.gripper == 1.0 and a.grip_source == "hand"
    a.on_grip_toggle(FAR, now=3.0)             # Muse flips it open, outside the window
    assert a.gripper == 0.0 and a.grip_source == "muse"
    for i in range(40):                        # the hand is STILL a fist: must not re-close
        a.on_hand(0.05, 3.5 + i / 30.0, FAR)
    assert a.gripper == 0.0
    open_edge(a, 10.0)                          # hand opens (no change), then
    fist_edge(a, 12.0)                          # a fresh open->fist edge re-closes it
    assert a.gripper == 1.0 and a.grip_source == "hand"


# --- clash policy (coalesce window) -------------------------------------------------------------------------------
def test_fist_plus_simultaneous_clench_is_one_close_not_a_flip_back():
    """The bug: fist (hand CLOSE) + a near-simultaneous Muse clench (a FLIP) used to re-open. Now the toggle is read
    relative to the state BEFORE the hand edge, so fist + clench = one CLOSE and no clash."""
    a = follow()
    a.on_hand(0.05, 0.0, FAR)
    a.on_hand(0.05, 0.61, FAR)                 # hand CLOSE edge fires at 0.61
    assert a.gripper == 1.0 and a.grip_source == "hand"
    a.on_grip_toggle(FAR, now=0.71)            # clench 0.1 s later, inside the window
    assert a.gripper == 1.0                    # still CLOSED, not flipped back
    assert a.grip_clashes == 0


def test_muse_then_hand_same_state_is_a_no_op():
    a = follow()
    a.on_hand(0.05, 0.0, FAR)                  # fist posture starts
    a.on_grip_toggle(FAR, now=0.3)             # clench closes first
    assert a.gripper == 1.0 and a.grip_source == "muse"
    a.on_hand(0.05, 0.61, FAR)                 # hand CLOSE edge, same intent, inside the window
    assert a.gripper == 1.0 and a.grip_source == "muse"   # the edge was a no-op; Muse stays the author
    assert a.grip_clashes == 0


def test_opposite_intents_in_window_later_wins_and_is_a_clash():
    a = follow()
    a.on_grip_toggle(FAR, now=0.0)             # muse CLOSE
    a.on_grip_toggle(FAR, now=1.0)             # muse OPEN (outside the first window): clean flip, gripper 0
    assert a.gripper == 0.0 and a.grip_source == "muse"
    a.on_hand(0.05, 0.8, FAR)                  # a fist posture starts
    a.on_hand(0.05, 1.41, FAR)                 # hand CLOSE edge fires at 1.41, inside 0.6 s of the muse OPEN at 1.0
    assert a.gripper == 1.0 and a.grip_source == "hand"   # the later event (hand) wins
    assert a.grip_clashes == 1
    t, earlier, winner, direction = a.grip_clash_log[-1]
    assert earlier == "muse" and winner == "hand" and direction == "close"


def test_outside_window_everything_is_normal():
    a = follow(coalesce_s=0.6)
    a.on_grip_toggle(FAR, now=0.0)             # muse CLOSE
    open_edge(a, 2.0)                          # hand OPEN edge far outside the window: a normal set, no clash
    assert a.gripper == 0.0 and a.grip_source == "hand" and a.grip_clashes == 0


# --- source restriction + HOLD-mode toggle ------------------------------------------------------------------------
def test_source_restriction_hand_only():
    a = follow(grip_sources=["hand"])
    a.on_grip_toggle(FAR, now=0.0)
    assert a.gripper == 0.0                     # Muse disabled
    fist_edge(a, 0.0)
    assert a.gripper == 1.0 and a.grip_source == "hand"


def test_source_restriction_eeg_only():
    b = follow(grip_sources=["eeg"])
    fist_edge(b, 0.0)
    assert b.gripper == 0.0                     # hand disabled
    b.on_grip_toggle(FAR, now=0.0)
    assert b.gripper == 1.0 and b.grip_source == "muse"


def test_hold_mode_toggle_unchanged():
    a = Arbiter(Scene(SCENE))
    assert a.mode == Mode.HOLD
    a.on_grip_toggle(None)
    assert a.gripper == 0.0                     # default: HOLD ignores the clench
    a.grip_toggle_in_hold = True
    a.on_grip_toggle(None)
    assert a.gripper == 1.0 and a.mode == Mode.HOLD and a.grip_source == "muse"
    a.on_grip_toggle(None)
    assert a.gripper == 0.0


def test_assisted_grab_still_starts_from_a_muse_toggle():
    a = follow()
    cup = np.array(SCENE["cup"]["pos"])
    a.on_grip_toggle(cup + [0.03, 0.0, 0.06])   # object in reach: assisted grab, unchanged
    assert a.mode == Mode.AUTO and a.assist and "cup" in a.last_say


def test_config_resolution_default_and_alias():
    """gripper.sources wins; otherwise body.grip_from_hand is the backward-compatible alias."""
    from rover_intent.app import build
    from rover_intent.config import load_config
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(root / "configs/default.yaml"))
    assert build(cfg, "mock")[4].grip_sources == {"hand", "eeg"}   # default.yaml lists both
    c2 = {**cfg, "gripper": {}, "body": {**cfg["body"], "grip_from_hand": False}}
    assert build(c2, "mock")[4].grip_sources == {"eeg"}            # alias false -> Muse/eeg only
    c3 = {**cfg, "gripper": {"sources": ["hand"]}}
    assert build(c3, "mock")[4].grip_sources == {"hand"}           # explicit sources win

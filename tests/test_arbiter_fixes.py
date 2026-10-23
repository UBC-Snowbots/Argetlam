"""Arbiter body-teleop fixes: a failed grab assist returns to FOLLOW, not HOLD (issue 4); the hand->gripper
front end is EMA-smoothed with a wider band and a longer dwell (issue 5)."""
import numpy as np

from rover_intent.control.arbiter import Arbiter, HAND_HOLD_S, Mode
from rover_intent.planner.scene import Scene
from rover_intent.types import Action, Intent

SCENE = {"cup": {"pos": [0.55, 0.15, 0.10]}, "bottle": {"pos": [0.55, -0.15, 0.12], "height": 0.22}}


# ---- issue 4 -----------------------------------------------------------------------------------
def test_failed_assist_returns_to_follow_not_hold():
    a = Arbiter(Scene(SCENE), waypoint_timeout=8.0)
    a.on_intent(Intent(action=Action.follow))
    cup = np.array(SCENE["cup"]["pos"])
    tcp = cup + [0.04, -0.03, 0.06]
    a.on_hand(0.1, 0.0, tcp)
    a.on_hand(0.1, HAND_HOLD_S + 0.01, tcp)
    assert a.mode == Mode.AUTO and a.assist          # grab assist started
    far = np.array([0.0, 0.0, 1.0])                  # but the arm can never reach the waypoint
    assert a.target(far, None, now=100.0)[0] is not None
    p, _ = a.target(far, None, now=109.0)            # > waypoint_timeout later
    assert p is None
    assert a.mode == Mode.FOLLOW and not a.assist     # handed control BACK to the body, not stuck in HOLD
    assert a.events == ["assist_done"]
    assert "back to following" in a.last_say
    # body targets drive again immediately
    hand = cup + [0.1, 0.1, 0.1]
    assert a.target(far, hand, now=110.0)[0] is not None


def test_failed_plain_auto_still_holds():
    a = Arbiter(Scene(SCENE), waypoint_timeout=8.0)
    a.on_intent(Intent(action=Action.pick, object="cup"))   # a voice pick, not an assist
    far = np.array([0.0, 0.0, 1.0])
    a.target(far, None, now=0.0)
    p, _ = a.target(far, None, now=9.0)
    assert p is None and a.mode == Mode.HOLD and not a.assist and "holding" in a.last_say
    assert a.events == []


# ---- issue 5 -----------------------------------------------------------------------------------
def test_hand_band_widened_and_dwell_longer():
    far = np.array([0.40, 0.0, 0.30])
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.follow))
    # a steady half-open hand (0.5) sits in the dead band (0.30..0.65): never triggers
    for i in range(20):
        a.on_hand(0.5, 0.1 * i, far)
    assert a.gripper == 0.0
    # a clear fist streamed at 30 Hz: must be held ~HAND_HOLD_S (0.6 s) before it closes
    t0, closed_at = 5.0, None
    for i in range(40):
        t = t0 + i / 30
        a.on_hand(0.05, t, far)
        if a.gripper == 1.0 and closed_at is None:
            closed_at = t
    assert closed_at is not None and closed_at - t0 >= HAND_HOLD_S - 0.05
    assert HAND_HOLD_S == 0.6


def test_hand_ema_smooths_a_single_spike():
    far = np.array([0.40, 0.0, 0.30])
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.follow))
    # hand is open (high) but one frame spikes to a fist; the EMA keeps it above the close threshold
    a.on_hand(0.9, 0.0, far)
    a.on_hand(0.05, 0.03, far)                        # single-frame glitch toward a fist
    a.on_hand(0.9, 0.06, far)
    for i in range(5):
        a.on_hand(0.9, 0.1 + 0.03 * i, far)
    assert a.gripper == 0.0                           # one spike never closes the gripper


def test_hand_ema_resets_when_leaving_follow():
    far = np.array([0.40, 0.0, 0.30])
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.follow))
    a.on_hand(0.1, 0.0, far)                          # start a fist (ema low)
    assert a._hand_ema is not None
    a.abort()                                         # clutch release / tracking loss
    assert a._hand_ema is None and a._hand_want is None
    a.on_intent(Intent(action=Action.follow))
    a.on_hand(0.9, 1.0, far)                          # fresh open hand: ema starts here, no carry-over
    assert a._hand_ema == 0.9

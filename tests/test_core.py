from pathlib import Path

import numpy as np
import pytest

from rover_intent.control.arbiter import Arbiter, Mode
from rover_intent.control.kinematics import ArmModel, rot_err
from rover_intent.control.safety import SafetyLayer
from rover_intent.planner.nemotron import IntentParser, fallback_parse, urgent_stop
from rover_intent.planner.scene import Scene
from rover_intent.planner.skills import SkillError, plan
from rover_intent.types import Action, Intent

URDF = str(Path(__file__).resolve().parents[1] / "assets/rover2026/rover2026.urdf")
TOP_DOWN = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
SCENE = {"cup": {"pos": [0.55, 0.15, 0.10]}, "bottle": {"pos": [0.55, -0.15, 0.12], "height": 0.22}}


@pytest.fixture(scope="module")
def arm():
    return ArmModel(URDF, tcp_offset=(0.15, 0, 0))


def test_fk_matches_spec_sheet():
    # docs/rover2026_arm_specs.md: all-zero pose puts the EE holder at (-0.03, 0.15, 0.50)
    p = ArmModel(URDF).fk(np.zeros(6))[:3, 3]
    assert np.allclose(p, [-0.03, 0.15, 0.50], atol=0.005)


def test_jacobian_matches_finite_difference(arm):
    q = np.array([1.0, -1.2, 0.9, 0.3, 1.1, -0.4])
    J = arm.jacobian(q)
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6); dq[i] = eps
        dp = (arm.fk(q + dq)[:3, 3] - arm.fk(q - dq)[:3, 3]) / (2 * eps)
        assert np.allclose(J[:3, i], dp, atol=1e-5)


@pytest.mark.parametrize("p", [(0.55, 0.15, 0.10), (0.7, 0.0, 0.06), (0.45, 0.0, 0.45)])
def test_ik_top_down_reaches_scene_points(arm, p):
    home = np.array([3.98, -1.31, 0.86, 0.0, 2.12, 2.31])
    T = np.eye(4); T[:3, :3] = TOP_DOWN; T[:3, 3] = p
    q = arm.ik(home, T, iters=400)
    Tq = arm.fk(q)
    assert np.linalg.norm(Tq[:3, 3] - p) < 2e-3
    assert np.degrees(np.linalg.norm(rot_err(Tq[:3, :3], TOP_DOWN))) < 2
    assert np.all(q >= arm.lower) and np.all(q <= arm.upper)


def _safety(arm, **kw):
    return SafetyLayer(arm.lower, arm.upper, [0.8] * 3 + [1.5] * 3, [2.0] * 3 + [4.0] * 3,
                       [-0.2, -0.8, 0.05], [1.0, 0.8, 1.0], **kw)


def test_safety_respects_vel_and_acc(arm):
    s, dt = _safety(arm), 0.02
    q = np.array([3.98, -1.31, 0.86, 0.0, 2.12, 2.31])
    prev_v = np.zeros(6)
    for _ in range(200):
        nq = s.step(q, q + 1.0, dt, input_age_s=0.0)  # far target: saturate
        v = (nq - q) / dt
        assert np.all(np.abs(v) <= np.array(s.max_vel) + 1e-9)
        assert np.all(np.abs(v - prev_v) <= np.array(s.max_acc) * dt + 1e-9)
        q, prev_v = nq, v


def test_safety_estop_and_watchdog(arm):
    s, dt = _safety(arm), 0.02
    q = np.array([3.98, -1.31, 0.86, 0.0, 2.12, 2.31])
    for _ in range(20):
        q = s.step(q, q + 0.5, dt, 0.0)
    # stale input -> decelerates to a stop instead of continuing
    for _ in range(100):
        q = s.step(q, q + 0.5, dt, input_age_s=1.0)
    assert np.allclose(s._vel, 0, atol=1e-6)
    s.estop("test")
    assert np.array_equal(s.step(q, q + 1, dt, 0.0), q)
    s.reset()
    assert not s.estopped


def test_workspace_clip(arm):
    assert np.allclose(_safety(arm).clip_workspace(np.array([2.0, 0, -1])), [1.0, 0, 0.05])


@pytest.mark.parametrize("text,action,obj,rel,ref", [
    ("move the cup beside the bottle", Action.move, "cup", "beside", "bottle"),
    ("put the cup next to the bottle", Action.move, "cup", "beside", "bottle"),
    ("pick up the cup", Action.pick, "cup", None, None),
    ("let go", Action.release, None, None, None),
    ("follow me", Action.follow, None, None, None),
])
def test_fallback_parser(text, action, obj, rel, ref):
    i = fallback_parse(text, ["cup", "bottle"])
    assert (i.action, i.object, i.relation, i.reference) == (action, obj, rel, ref)


@pytest.mark.parametrize("text", ["stop!", "whoa whoa hold on", "WAIT", "emergency", "freeze"])
def test_stop_words_never_reach_the_llm(text, monkeypatch):
    assert urgent_stop(text).action == Action.stop
    p = IntentParser("any", "http://unused")
    p.client = object()  # an 'online' parser whose client would crash if called
    assert p.parse(text, ["cup"]).action == Action.stop


@pytest.mark.parametrize("text", ["reset", "Restart the table please", "let's start over"])
def test_reset_is_local(text):
    p = IntentParser("any", "http://unused")
    p.client = object()
    assert p.parse(text, ["cup"]).action == Action.reset


def test_reset_clears_arbiter_and_pick_while_holding_is_refused():
    a = Arbiter(Scene(SCENE))
    a.held, a.gripper = "block", 1.0
    a.on_intent(Intent(action=Action.pick, object="cup"))
    assert a.mode == Mode.HOLD and "already holding the block" in a.last_say  # no mid-air drop
    a.on_intent(Intent(action=Action.reset))
    assert (a.mode, a.held, a.gripper) == (Mode.HOLD, None, 0.0)


def test_object_not_in_scene_never_moves():
    # Nemotron may name anything; the skill layer only moves for objects it can see
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.pick, object="sender", say="Picking up the sender."))
    assert a.mode == Mode.HOLD and "unknown object 'sender'" in a.last_say


def test_clarify_does_not_move():
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.clarify, say="Which one, the cup or the block?"))
    assert a.mode == Mode.HOLD and a.last_say == "Which one, the cup or the block?"


def test_fallback_word_boundaries():
    # 'block' must not be found inside 'blocked', nor 'cup' inside 'cupboard'
    assert fallback_parse("the cupboard is blocked", ["cup", "block"]).action == Action.unknown


def test_skills_move_plan():
    wps = plan(Intent(action=Action.move, object="cup", relation="beside", reference="bottle"),
               Scene(SCENE), held=None)
    assert [w.label for w in wps][:5] == ["above-object", "pre-grasp", "grasp-pos", "close", "lift"]
    # every move between objects happens at the transit height, above the tallest object
    for w in wps:
        if w.label in ("above-object", "lift", "above-goal", "retreat"):
            assert w.pos[2] >= 0.30
    assert wps[-2].label == "open" and wps[-2].gripper == 0
    # released beside the bottle at the cup's own grasp height, not the bottle's
    assert np.isclose(wps[-2].pos[2], SCENE["cup"]["pos"][2])
    assert np.allclose(wps[-2].pos[:2], np.array(SCENE["bottle"]["pos"][:2]) + [0, 0.12])


def test_arbiter_dwells_and_tracks_held_object():
    a = Arbiter(Scene(SCENE), waypoint_tol=0.01)
    a.on_intent(Intent(action=Action.pick, object="cup"))
    cup = np.array(SCENE["cup"]["pos"])
    a.target(np.array([*cup[:2], 0.30]), None, now=0.0)  # above the object -> advances
    a.target(cup + [0, 0, 0.10], None, now=0.05)      # at pre-grasp -> advances
    a.target(cup, None, now=0.1)                      # at grasp-pos -> advances
    p, g = a.target(cup, None, now=0.2)               # at close: gripper closes, dwell starts
    assert g == 1.0 and a.held is None
    p, _ = a.target(cup, None, now=0.5)
    assert np.allclose(p, cup)                        # still waiting for the fingers
    a.target(cup, None, now=1.1)                      # dwell over
    assert a.held == "cup"
    p, _ = a.target(cup, None, now=1.2)
    assert p[2] > cup[2]                              # lifting
    with pytest.raises(SkillError):
        plan(Intent(action=Action.pick, object="banana"), Scene(SCENE), held=None)


def test_arbiter_eeg_hysteresis_and_dwell():
    a = Arbiter(Scene(SCENE), eeg_hold_s=0.5)
    a.on_intent(Intent(action=Action.follow))
    a.on_eeg(0.9, now=0.0); a.on_eeg(0.9, now=0.3)
    assert a.gripper == 0.0          # not held long enough
    a.on_eeg(0.5, now=0.4)           # in the dead band: keeps the pending change alive
    a.on_eeg(0.9, now=0.6)
    assert a.gripper == 1.0
    a.on_eeg(0.1, now=1.0); a.on_eeg(0.9, now=1.2); a.on_eeg(0.1, now=1.3)
    assert a.gripper == 1.0          # flicker does not release


def test_arbiter_waypoint_timeout_holds():
    a = Arbiter(Scene(SCENE), waypoint_timeout=8.0)
    a.on_intent(Intent(action=Action.pick, object="cup"))
    far = np.array([0.0, 0.0, 1.0])  # e.g. blocked by an obstacle, never gets there
    assert a.target(far, None, now=0.0)[0] is not None
    assert a.target(far, None, now=7.9)[0] is not None
    p, _ = a.target(far, None, now=8.1)
    assert p is None and a.mode == Mode.HOLD and "Couldn't reach" in a.last_say


def test_arbiter_stop_beats_auto():
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.pick, object="cup"))
    assert a.mode == Mode.AUTO
    a.on_intent(Intent(action=Action.stop))
    assert a.mode == Mode.HOLD and a.target(np.zeros(3), np.ones(3))[0] is None


def test_arbiter_shared_autonomy_pulls_toward_object():
    a = Arbiter(Scene(SCENE), assist_radius=0.08, assist_gain=0.6)
    a.on_intent(Intent(action=Action.follow))
    hand = np.array([0.55, 0.15, 0.14])  # 4 cm above the cup
    p, _ = a.target(np.zeros(3), hand)
    assert np.linalg.norm(p - SCENE["cup"]["pos"]) < np.linalg.norm(hand - SCENE["cup"]["pos"])

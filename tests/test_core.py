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


def test_retarget_is_relative_after_engage_and_directions_are_right():
    import yaml
    from rover_intent.body.retarget import Retargeter
    from rover_intent.types import BodyPose
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/default.yaml"))["body"]
    r = Retargeter(cfg["axes"], scale=1.0, origin=[0.45, 0, 0.45], min_cutoff=1e6)  # huge cutoff = no smoothing
    r.engage([0.5, 0.1, 0.3])
    pose = lambda w, t: BodyPose(t=t, wrist=w, shoulder=(0, 0, 0))
    assert np.allclose(r(pose((0.0, 0.5, 0.0), 0.0)), [0.5, 0.1, 0.3])     # first sample = neutral: no jump
    assert np.allclose(r(pose((0.0, 0.4, -0.1), 0.1)), [0.6, 0.1, 0.4])    # hand 10 cm up + toward the camera
    # user's RIGHT = image LEFT (x decreases) in a raw webcam frame -> gripper to the right in the over-shoulder view (-y)
    assert r(pose((-0.1, 0.5, 0.0), 0.2))[1] < 0.1


def test_one_euro_smooths_jitter_but_follows_fast_moves():
    from rover_intent.body.retarget import OneEuro
    f, rng = OneEuro(1.0, 0.7), np.random.default_rng(0)
    still = [f(np.array([0.0]) + rng.normal(0, 0.01, 1), i / 30)[0] for i in range(60)]
    assert np.std(still[30:]) < 0.004                      # 1 cm jitter -> < 4 mm
    f.reset()
    ramp = [f(np.array([i / 30 * 1.0]), i / 30)[0] for i in range(30)]  # 1 m/s move
    assert 1.0 * 29 / 30 - ramp[-1] < 0.06                  # lag under 6 cm at 1 m/s


def test_hand_not_found_is_not_open():
    from rover_intent.types import BodyPose
    assert BodyPose(t=0, wrist=(0, 0, 0), shoulder=(0, 0, 0)).hand_open is None


def test_hand_grip_only_in_follow_with_hysteresis():
    a = Arbiter(Scene(SCENE))
    a.on_hand(0.1)
    assert a.gripper == 0.0            # not following: ignored
    a.on_intent(Intent(action=Action.follow))
    a.on_hand(0.1); assert a.gripper == 1.0
    a.on_hand(0.5); assert a.gripper == 1.0   # dead band keeps it closed
    a.on_hand(0.8); assert a.gripper == 0.0


def test_app_and_laptop_scripts_compile():
    import py_compile
    root = Path(__file__).resolve().parents[1]
    import rover_intent.app  # noqa: F401
    for f in ["laptop/voice_client.py", "laptop/body_client.py", "sim/isaac_bridge.py", "src/rover_intent/eeg/live.py"]:
        py_compile.compile(str(root / f), doraise=True)


def _synthetic_arm(lateral=0.0, up=0.0, flex_deg=60.0, t=0.0):
    """Right arm of a person facing the camera, in MediaPipe world coords (x image-right, y down, z toward camera
    smaller). Their fwd = -z, left = +x, up = -y. Wrist placed via shoulder->elbow->wrist geometry."""
    from rover_intent.types import BodyPose
    fwd, left, upv = np.array([0, 0, -1.]), np.array([1., 0, 0]), np.array([0, -1., 0])
    S, O, H = -0.18 * left, 0.18 * left, -0.5 * upv
    u = _n(-0.5 * upv + 0.8 * fwd + lateral * left + up * upv)
    E = S + 0.30 * u
    side = _n(np.cross(u, upv)); perp = np.cross(side, u)
    f = np.cos(np.radians(flex_deg)) * u + np.sin(np.radians(flex_deg)) * perp
    return BodyPose(t=t, wrist=tuple(E + 0.27 * f), shoulder=tuple(S), elbow=tuple(E), shoulder_other=tuple(O),
                    hips=tuple(H))


def _n(v):
    return v / np.linalg.norm(v)


def test_gesture_mirror_directions():
    from rover_intent.body.mirror import GestureMirror
    arm = ArmModel(URDF, tcp_offset=(0.145, 0, 0))
    home_tcp = np.array([0.45, 0.0, 0.45])
    def run(start=home_tcp, **kw):
        m = GestureMirror(arm, min_cutoff=1e6)
        m.engage(start)
        first = m(_synthetic_arm(t=0.0))
        return first, m(_synthetic_arm(t=0.1, **kw)), m
    first, _, _ = run()
    assert np.allclose(first, home_tcp, atol=1e-6)                     # engage: no jump
    _, p, _ = run(lateral=0.3)                                          # swing hand to your LEFT
    assert p[1] > home_tcp[1] + 0.03                                    # -> gripper moves to robot left (+y)
    _, p, _ = run(up=0.4)                                               # raise hand
    assert p[2] > home_tcp[2] + 0.03                                    # -> gripper up
    _, p_ext, _ = run(flex_deg=20)                                      # straighten elbow (reach out)
    axis = GestureMirror(arm).axis_xy
    assert np.hypot(*(p_ext[:2] - axis)) > np.hypot(*(home_tcp[:2] - axis)) + 0.02   # -> reaches farther
    _, p, m = run(start=np.array([0.45, 0.0, 0.15]), up=-3.0)           # low gripper, hand dropped ~0.34 m
    assert m.clipped and abs(p[2] - 0.03) < 1e-9                        # clamped at the floor, flagged


@pytest.mark.parametrize("mode", ["gesture", "cartesian"])
def test_app_loop_end_to_end_mock(mode):
    """Runs the real control loop (mock arm) and drives it over UDP: reset, follow, gestures, stop."""
    import json, socket, threading, time, yaml
    from rover_intent import app
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/default.yaml"))
    port = 47190 + (mode == "cartesian")
    cfg["net"]["listen_port"], cfg["body"]["mode"] = port, mode
    cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"   # offline parser, no network
    th = threading.Thread(target=app.run, args=(cfg, "mock", 4.0), daemon=True)
    th.start(); time.sleep(0.5)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3)
    def say(t):
        s.sendto(json.dumps({"kind": "utterance", "t": time.time(), "text": t}).encode(), ("127.0.0.1", port))
        return json.loads(s.recvfrom(9999)[0])
    assert say("reset")["mode"] == "hold"
    assert say("follow me")["mode"] == "follow"
    statuses = []
    s.settimeout(0.001)  # non-blocking-ish drain while streaming poses
    for i in range(45):  # 1.5 s of swinging the hand to the left
        p = _synthetic_arm(lateral=min(0.25, i * 0.01), t=time.time()).model_dump()
        p.update(kind="body", visible=True)
        if mode == "cartesian":
            p["wrist"] = [p["wrist"][0] + min(0.25, i * 0.01), *p["wrist"][1:]]
        s.sendto(json.dumps(p).encode(), ("127.0.0.1", port)); time.sleep(1 / 30)
        try:
            while True:
                m = json.loads(s.recvfrom(9999)[0])
                if m.get("kind") == "status":
                    statuses.append(m)
        except (socket.timeout, BlockingIOError):
            pass
    s.settimeout(3)
    tcps = [m["tcp"] for m in statuses if m["mode"] == "follow"]
    assert tcps[-1][1] > tcps[0][1] + 0.02   # hand to your left -> gripper moved to robot left (+y)
    assert statuses and statuses[-1]["mode"] == "follow" and statuses[-1]["target"] is not None
    assert say("reset")["mode"] == "hold"
    th.join(timeout=6)
    assert not th.is_alive()

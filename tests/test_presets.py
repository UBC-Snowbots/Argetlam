"""Button-panel presets: plan checks, hardware home gate, message parsing, and the body-client panel logic."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rover_intent.config import load_config
from rover_intent.control.kinematics import ArmModel
from rover_intent.control.presets import (HOME_HW, RETURN, at_goal, outside, plan_preset, preset_names,
                                          request_hw_home)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(arm="dev_arm")
    a = cfg["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), base=a.get("base_link", "base_link"), tip=a["tip_link"],
                   tcp_offset=a.get("tcp_offset"))
    return cfg, arm, np.array(a["home"]), a.get("presets", {}), cfg["safety"]["workspace_min"], cfg["safety"]["workspace_max"]


def test_profile_has_presets_inside_joint_limits(setup):
    cfg, arm, home, presets, lo, hi = setup
    assert {"ready", "base"} <= set(presets)
    for q in presets.values():
        assert np.all(np.asarray(q) >= arm.lower) and np.all(np.asarray(q) <= arm.upper)


def test_return_and_named_presets_plan(setup):
    cfg, arm, home, presets, lo, hi = setup
    goal, msg = plan_preset(RETURN, presets, home, home, "idle", arm, lo, hi)
    assert np.allclose(goal, home) and "return" in msg
    goal, msg = plan_preset("base", presets, home, home, "idle", arm, lo, hi)
    assert goal is not None, msg  # base is outside the default box, but the path never goes further out than it


def test_refusals(setup):
    cfg, arm, home, presets, lo, hi = setup
    assert plan_preset("ready", presets, home, home, "following", arm, lo, hi)[0] is None
    assert plan_preset("nope", presets, home, home, "idle", arm, lo, hi)[0] is None
    bad = dict(presets, wild=[9.0, 0, 0, 0, 0, 0])
    goal, msg = plan_preset("wild", bad, home, home, "idle", arm, lo, hi)
    assert goal is None and "joint limits" in msg
    assert plan_preset("short", dict(short=[0.1, 0.2]), home, home, "idle", arm, lo, hi)[0] is None


class ArcArm:
    """Stand-in arm: joint 1 in [0, 1] sweeps the TCP along an arc that bulges out of the box in the middle."""
    lower, upper = np.full(6, -10.0), np.full(6, 10.0)

    @staticmethod
    def fk(q):
        T = np.eye(4)
        T[:3, 3] = [0.5, 0.6 * np.sin(np.pi * q[0]), 0.1]
        return T


def test_path_that_wanders_out_of_the_box_is_refused():
    lo, hi = [0.4, -0.4, 0.03], [0.62, 0.4, 0.32]
    q0, q1 = np.zeros(6), np.r_[1.0, np.zeros(5)]
    assert outside(ArcArm.fk(q0)[:3, 3], lo, hi) == 0 and outside(ArcArm.fk(q1)[:3, 3], lo, hi) < 1e-9
    goal, msg = plan_preset("arc", dict(arc=q1.tolist()), q0, q0, "idle", ArcArm, lo, hi)
    assert goal is None and "leaves the workspace box" in msg
    # a small arc that stays inside the box is fine
    goal, msg = plan_preset("small", dict(small=[0.15, 0, 0, 0, 0, 0]), q0, q0, "idle", ArcArm, lo, hi)
    assert goal is not None, msg


def test_at_goal():
    g = np.zeros(6)
    assert at_goal(g, g + 0.005, g) and not at_goal(g + 0.01, g, g) and not at_goal(g, g + 0.05, g)
    assert at_goal(g + 0.0012, g + 0.0012, g)  # the SafetyLayer stops this far short on purpose


def test_safety_layer_settles_within_at_goal(setup):
    """Regression: drive the real SafetyLayer to a preset and check the move is declared finished."""
    from rover_intent.control.safety import SafetyLayer
    cfg, arm, home, presets, lo, hi = setup
    a, s = cfg["arm"], cfg["safety"]
    safety = SafetyLayer(arm.lower, arm.upper, a["max_vel"], a["max_acc"], s["workspace_min"], s["workspace_max"])
    q, goal = home.copy(), np.asarray(presets["base"])
    for _ in range(2000):
        q = safety.step(q, goal, 0.02, 0.0)
        if at_goal(q, q, goal):
            break
    assert at_goal(q, q, goal)


def test_hw_home_gate():
    assert request_hw_home({"enabled": False, "home_cmd": "x"}, True)[0] is False
    assert request_hw_home({"enabled": True, "home_cmd": "x"}, False)[0] is False
    assert request_hw_home({"enabled": True}, True)[0] is False
    ok, msg = request_hw_home({"enabled": True, "home_cmd": "echo hi"}, True, dry_run=True)
    assert ok and "would run" in msg


def test_preset_names_and_default_real_arm_off(setup):
    cfg, *_ = setup
    assert cfg["real_arm"]["enabled"] is False and "cmd_value: 170" in cfg["real_arm"]["home_cmd"]
    assert preset_names({"ready": [], "base": []}, False) == [RETURN, "ready", "base"]
    assert preset_names({}, True)[-1] == HOME_HW


def test_preset_message_parses():
    from rover_intent.transport.udp import _MSG
    from rover_intent.types import ArmPreset
    m = _MSG.validate_python(json.loads(json.dumps({"kind": "preset", "t": 1.0, "name": "home_hw", "confirm": True})))
    assert isinstance(m, ArmPreset) and m.confirm


# --- body-client panel (cv2 / mediapipe stubbed: only the button logic is tested) ------------------------------
@pytest.fixture(scope="module")
def body_client():
    stubs = {}
    cv2 = types.ModuleType("cv2")
    cv2.EVENT_LBUTTONDOWN, cv2.FONT_HERSHEY_SIMPLEX = 1, 0
    cv2.rectangle = cv2.putText = lambda *a, **k: None
    stubs["cv2"] = cv2
    mp = types.ModuleType("mediapipe")
    tasks, tpy = types.ModuleType("mediapipe.tasks"), types.ModuleType("mediapipe.tasks.python")
    tpy.BaseOptions, tpy.vision = object, types.SimpleNamespace()
    stubs.update({"mediapipe": mp, "mediapipe.tasks": tasks, "mediapipe.tasks.python": tpy})
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("body_client_t", ROOT / "laptop/body_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_panel_buttons_and_home_confirm(body_client):
    p = body_client.ButtonPanel()
    frame = np.zeros((480, 640, 3), np.uint8)
    p.draw(frame, {"presets": ["return", "ready", "base", "home_hw"]}, 0.0)
    names = [r[0] for r in p.rects]
    assert names == ["stop", "return", "ready", "base", "home_hw"]

    def click(name, t):
        _, x, y, w, h = next(r for r in p.rects if r[0] == name)
        p.on_mouse(1, x + w // 2, y + h // 2, 0, None)
        return p.take(t)

    assert click("ready", 1.0) == [{"kind": "preset", "t": 1.0, "name": "ready", "confirm": False}]
    assert click("stop", 2.0)[0]["text"] == "stop"
    assert click("home_hw", 3.0) == []                      # first click only arms it
    out = click("home_hw", 4.5)                             # second click within 3 s confirms
    assert out == [{"kind": "preset", "t": 4.5, "name": "home_hw", "confirm": True}]
    assert click("home_hw", 10.0) == []                     # a later single click starts over
    assert click("home_hw", 14.0) == []                     # ...and a second click after 3 s does not confirm
    p.on_mouse(1, 5, 5, 0, None)                            # a click on the video, not a button
    assert p.take(15.0) == []

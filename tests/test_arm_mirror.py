"""Full-arm body teleop (body/arm_mirror.py) and its app wiring (body.mode = arm_mirror).

Pure-function + mapper tests: the torso-frame angles move the right way, engage never jumps, the 2 s ramp eases in,
joint-limit clamping is flagged, J1 is held when the arm is near vertical, the confidence gate skips invisible poses,
the One-Euro + rate limit keep the output bounded, and the wrist bend is only driven when a hand landmark is sent.
App test: dev_arm_real in arm_mirror mode follows synthetic poses with the joints moving smoothly inside the speed
limits (no IK), and dropping the clutch (F toggle off) returns the arm to base.
"""
import json
import math
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from rover_intent import app
from rover_intent.body.arm_mirror import ArmMirror, arm_angles
from rover_intent.config import load_config
from rover_intent.control.kinematics import ArmModel
from rover_intent.types import BodyPose
from test_core import _synthetic_arm

ROOT = Path(__file__).resolve().parents[1]
URDF = str(ROOT / "assets/dev_arm/dev_arm_gripper.urdf")


def _dev_real_arm():
    cfg = load_config(arm="dev_arm_real")
    a = cfg["arm"]
    arm = ArmModel(URDF, base=a["base_link"], tip=a["tip_link"], tcp_offset=a["tcp_offset"], tool_axis=a["tool_axis"])
    arm.lower = np.array(a["joint_limits_lower"], float)
    arm.upper = np.array(a["joint_limits_upper"], float)
    return arm, np.array(a["home"], float), np.array(a["max_vel"], float), cfg.get("arm_mirror")


# --- torso-frame angles ---------------------------------------------------------

def test_arm_angles_directions():
    base = arm_angles(_synthetic_arm())
    assert base is not None
    # swing the upper arm toward the operator's LEFT -> azimuth increases
    assert arm_angles(_synthetic_arm(lateral=0.4))["az"] > base["az"] + 0.1
    # raise the arm -> elevation increases
    assert arm_angles(_synthetic_arm(up=0.4))["el"] > base["el"] + 0.1
    # bend the elbow more (bigger flex_deg) -> the included elbow angle grows
    assert arm_angles(_synthetic_arm(flex_deg=100))["elbow"] > arm_angles(_synthetic_arm(flex_deg=30))["elbow"] + 0.3


def test_arm_angles_none_without_landmarks():
    p = BodyPose(t=0.0, wrist=(0, 0, 0), shoulder=(0, 0, 0))   # no elbow / shoulder_other / hips
    assert arm_angles(p) is None


def test_wrist_bend_needs_hand():
    assert arm_angles(_synthetic_arm())["wrist"] is None
    p = _synthetic_arm()
    W = np.asarray(p.wrist); E = np.asarray(p.elbow)
    bent = W + 0.08 * np.cross(np.asarray(p.shoulder) - E, [0, 0, 1])   # a hand off the forearm line
    p2 = p.model_copy(update={"hand": tuple(bent)})
    assert arm_angles(p2)["wrist"] is not None and arm_angles(p2)["wrist"] > 0.1


# --- mapper behaviour -----------------------------------------------------------

def _mirror(**over):
    arm, home, mv, cfg = _dev_real_arm()
    cfg = dict(cfg or {})
    cfg.update(over)
    return ArmMirror(arm, home, mv, cfg), home


def test_engage_no_jump_and_ramp():
    m, home = _mirror()
    m.engage(home.copy())
    first = m(_synthetic_arm(t=0.0))
    assert np.allclose(first, home, atol=1e-6)                       # engage: output == current joints
    mid = m(_synthetic_arm(t=1.0, lateral=0.3))                      # half way through the 2 s ramp
    full = home.copy()
    # run well past the ramp; feed the same pose so it settles
    out = mid
    for k in range(1, 400):
        out = m(_synthetic_arm(t=1.0 + k * 0.03, lateral=0.3))
    assert not np.allclose(out, home, atol=1e-3)                     # it did move toward the mirrored pose
    # the mid-ramp sample is between the engage pose and the settled pose on the joints that moved
    moved = np.argmax(np.abs(out - home))
    lo, hi = sorted([home[moved], out[moved]])
    assert lo - 1e-6 <= mid[moved] <= hi + 1e-6


def test_confidence_gate():
    m, home = _mirror()
    m.engage(home.copy())
    p = _synthetic_arm(t=0.0)
    assert m(p.model_copy(update={"visible": False})) is None        # not visible -> skip (arm holds)


def test_limit_clamp_flagged():
    m, home = _mirror(gain={"j1": 6.0, "j2": 6.0, "j3": 6.0, "j5": 6.0}, ramp_s=0.0, deadband_deg=0.0)
    m.engage(home.copy())
    out = None
    for k in range(400):
        out = m(_synthetic_arm(t=k * 0.03, lateral=0.5, up=0.5))
    assert m.clipped                                                 # a joint target hit its travel limit
    assert np.all(out <= m.upper + 1e-9) and np.all(out >= m.lower - 1e-9)


def test_j1_held_near_vertical():
    m, home = _mirror(ramp_s=0.0, deadband_deg=0.0)
    m.engage(home.copy())
    # upper arm pointing almost straight DOWN (hanging): horizfrac tiny -> J1 must not track azimuth
    down = _synthetic_arm(t=0.0)
    S = np.asarray(down.shoulder)
    E = S + np.array([0.001, 0.30, 0.0])                             # straight down (+y is down)
    hang = down.model_copy(update={"elbow": tuple(E), "wrist": tuple(E + np.array([0.0, 0.27, 0.0]))})
    f = arm_angles(hang)
    assert f["horizfrac"] < math.sin(math.radians(20))               # it is near vertical
    j1_start = m(hang)[0]
    for k in range(1, 50):
        last = m(hang.model_copy(update={"t": k * 0.03}))
    assert abs(last[0] - j1_start) < 1e-6                            # J1 frozen while near vertical


def test_rate_limit_respected():
    arm, home, mv, cfg = _dev_real_arm()
    m = ArmMirror(arm, home, mv, dict(cfg or {}, ramp_s=0.0, deadband_deg=0.0,
                                      gain={"j1": 4.0, "j2": 4.0, "j3": 4.0, "j5": 4.0}))
    m.engage(home.copy())
    prev = m(_synthetic_arm(t=0.0))
    t = 0.0
    for k in range(1, 200):
        t_new = k * 0.03
        out = m(_synthetic_arm(t=t_new, lateral=0.5))
        step = np.abs(out - prev)
        assert np.all(step <= mv * (t_new - t) + 1e-9)               # never faster than the profile max_vel
        prev, t = out, t_new


# --- app wiring -----------------------------------------------------------------

class _App:
    def __init__(self, port, seconds):
        cfg = load_config(arm="dev_arm_real")
        cfg["body"]["mode"] = "arm_mirror"   # plane is the dev_arm_real default now; this test exercises arm_mirror
        cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
        cfg["body"]["clutch"] = {"calib_s": 0.5, "still_m": 0.05, "palm_s": 1.0, "raise_m": 0.05}
        cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"
        self.home = np.array(cfg["arm"]["home"], float)
        self.base = np.array(cfg["arm"]["presets"]["base"], float)
        self.th = threading.Thread(target=app.run, args=(cfg, "mock", seconds), kwargs={"record": False}, daemon=True)
        self.th.start(); time.sleep(0.5)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.s.settimeout(3)
        self.addr = ("127.0.0.1", port)

    def send(self, m):
        self.s.sendto(json.dumps(m).encode(), self.addr)

    def drain(self):
        self.s.settimeout(0.001)
        try:
            while True:
                self.s.recvfrom(65536)
        except (socket.timeout, BlockingIOError):
            pass
        self.s.settimeout(3)

    def query(self):
        self.send({"kind": "query", "t": time.time()})
        while True:
            m = json.loads(self.s.recvfrom(65536)[0])
            if m.get("kind") == "state":
                return m

    def join(self):
        self.th.join(timeout=12)


def test_app_arm_mirror_follows_and_returns_to_base():
    a = _App(47197, seconds=14.0)
    assert a.query()["mapping"] == "ArmMirror"
    a.send({"kind": "utterance", "t": time.time(), "text": "follow me"})
    last_q = None
    for i in range(140):                       # hold still to calibrate, then swing + raise the arm
        k = max(0, i - 24)
        p = _synthetic_arm(lateral=min(0.3, k * 0.006), up=min(0.3, k * 0.006), t=time.time()).model_dump()
        p.update(kind="body", visible=True)
        a.send(p); time.sleep(1 / 30)
        a.drain()
    moved = a.query()
    assert moved["clutch"] == "following" and moved["mode"] == "follow"
    off_home = float(np.max(np.abs(np.asarray(moved["q_meas"]) - a.home)))
    assert off_home > 0.05, f"the joints should have mirrored the arm, only {off_home:.3f} rad from home"
    # drop the clutch (F toggle off) -> arm_mirror returns to base (Pranav: "when I drop my arm return it to base")
    a.send({"kind": "clutch", "t": time.time(), "action": "toggle"}); a.drain()
    t_end = time.time() + 6.0
    while time.time() < t_end:
        time.sleep(1 / 30); a.drain()
    end = a.query()
    a.join()
    assert end["clutch"] == "idle"
    err = float(np.max(np.abs(np.asarray(end["q_meas"]) - a.base)))
    assert err < 0.03, f"arm should be back at base, still {err:.3f} rad away"


def test_app_arm_mirror_speed_within_limits():
    """Stream fast pose changes and confirm the commanded joints never exceed the profile speed limits (SafetyLayer)."""
    a = _App(47198, seconds=10.0)
    mv = np.array(load_config(arm="dev_arm_real")["arm"]["max_vel"], float)
    a.send({"kind": "utterance", "t": time.time(), "text": "follow me"})
    for i in range(24):                        # calibrate
        p = _synthetic_arm(t=time.time()).model_dump(); p.update(kind="body", visible=True)
        a.send(p); time.sleep(1 / 30); a.drain()
    prev = np.asarray(a.query()["q_meas"]); t_prev = time.time()
    worst = 0.0
    for i in range(120):                       # jerk the arm back and forth
        lat = 0.3 if (i // 10) % 2 == 0 else -0.1
        p = _synthetic_arm(lateral=lat, up=0.2, t=time.time()).model_dump(); p.update(kind="body", visible=True)
        a.send(p); time.sleep(1 / 30); a.drain()
        st = a.query(); now = time.time()
        q = np.asarray(st["q_meas"])
        v = np.abs(q - prev) / max(1e-3, now - t_prev)
        worst = max(worst, float(np.max(v - mv)))
        prev, t_prev = q, now
    a.join()
    assert worst < 0.02, f"a joint exceeded its max_vel by {worst:.3f} rad/s"

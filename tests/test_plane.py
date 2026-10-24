"""Plane teleop (body/plane.py) and its app wiring (body.mode = plane, the dev_arm_real default).

Image-plane only: every control is read from skel2d (x, y), normalized by shoulder width; MediaPipe depth (z) is
never used. Mapper tests: engage never jumps, the mapping signs/scales are right, the target is invariant to the
operator's distance (skeleton scale) and to depth (world z), reach is driven by the LEFT hand and held when it is
not tracked, and targets outside the box are clamped + flagged. App tests: dev_arm_real in plane mode follows
synthetic poses, the left-hand-raise clutch is DISABLED, dropping the control arm (wrist below the hip for >=0.7 s)
returns the arm to base, and the joints never exceed the profile speed limits. Plus a reachability check of the box.
"""
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np

from rover_intent import app
from rover_intent.body.plane import DropWatch, PlaneMapper, plane_coords
from rover_intent.config import load_config
from rover_intent.control.kinematics import ArmModel
from rover_intent.types import BodyPose

ROOT = Path(__file__).resolve().parents[1]
URDF = str(ROOT / "assets/dev_arm/dev_arm_gripper.urdf")


def _arm():
    cfg = load_config(arm="dev_arm_real")
    a = cfg["arm"]
    arm = ArmModel(URDF, base=a["base_link"], tip=a["tip_link"], tcp_offset=a["tcp_offset"], tool_axis=a["tool_axis"])
    arm.lower = np.array(a["joint_limits_lower"], float)
    arm.upper = np.array(a["joint_limits_upper"], float)
    return arm, np.array(a["home"], float), cfg


def _skel(rwx=0.35, rwy=0.50, lwy=0.50, hip_y=0.90, scale=1.0):
    """skel2d in body-client SKEL order, scaled about the image centre (0.5) to fake a nearer/farther operator.
    Right side at the SMALLER x (the mirror-corrected convention). order: Lsh Rsh Lel Rel Lwr Rwr Lhip Rhip nose."""
    def sc(x, y):
        return [0.5 + (x - 0.5) * scale, 0.5 + (y - 0.5) * scale]
    return [sc(0.65, 0.5), sc(0.35, 0.5), sc(0.70, 0.65), sc(0.30, 0.65), sc(0.68, lwy), sc(rwx, rwy),
            sc(0.60, hip_y), sc(0.40, hip_y), sc(0.50, 0.30)]


def _pose(t, rwx=0.35, rwy=0.50, lwy=0.50, hip_y=0.90, scale=1.0, visible=True, left_ok=True, zc=0.0):
    return BodyPose(t=t, wrist=(0.1, 0.0, zc), shoulder=(0.0, 0.0, zc), elbow=(0.05, 0.1, zc),
                    shoulder_other=(-0.2, 0.0, zc), hips=(0.0, 0.3, zc),
                    skel2d=_skel(rwx, rwy, lwy, hip_y, scale), visible=visible,
                    wrist_other=(-0.2, -0.1, zc) if left_ok else None)


def _mapper():
    arm, home, cfg = _arm()
    m = PlaneMapper(arm, home, cfg.get("plane"), cfg["safety"]["workspace_min"], cfg["safety"]["workspace_max"])
    return m, arm.fk(home)[:3, 3]


def _settle(m, tcp0, seq):
    """engage, feed a calibration-still prefix, then the given (kwargs) frames; return the last target."""
    m.reset(); m.engage(tcp0)
    p = None
    for k in range(80):
        p = m(_pose(k * 0.033))
    for i, kw in enumerate(seq):
        p = m(_pose(0.033 * (80 + i), **kw))
    return p


# --- mapper: engage + signs/scales ---------------------------------------------

def test_engage_no_jump():
    m, tcp0 = _mapper()
    m.engage(tcp0)
    first = m(_pose(0.0))
    assert np.allclose(first, tcp0, atol=1e-9)   # target == current TCP on the first frame after engage


def test_mapping_signs():
    m, tcp0 = _mapper()
    hold = [dict() for _ in range(80)]
    # right hand to the operator's RIGHT (smaller x) -> gripper to their right = robot -y
    pr = _settle(m, tcp0, [dict(rwx=0.25)] * 80)
    assert pr[1] < tcp0[1] - 0.02
    # right hand UP (smaller image y) -> robot +z
    pu = _settle(m, tcp0, [dict(rwy=0.35)] * 80)
    assert pu[2] > tcp0[2] + 0.02
    # LEFT hand UP -> reach forward = robot +x
    pf = _settle(m, tcp0, [dict(lwy=0.35)] * 80)
    assert pf[0] > tcp0[0] + 0.02
    # LEFT hand DOWN -> reach back = robot -x
    pb = _settle(m, tcp0, [dict(lwy=0.65)] * 80)
    assert pb[0] < tcp0[0] - 0.02


def test_distance_invariant():
    """Standing nearer (2x skeleton) or farther (0.5x) must not change the target: shoulder-width normalization."""
    targets = []
    for scale in (0.5, 1.0, 2.0):
        m, tcp0 = _mapper()
        p = _settle(m, tcp0, [dict(rwx=0.28, rwy=0.42, scale=scale)] * 120)
        targets.append(p)
    assert np.allclose(targets[0], targets[1], atol=2e-3) and np.allclose(targets[1], targets[2], atol=2e-3)


def test_depth_ignored():
    """Varying the world-landmark depth (z) wildly must not change the target (plane mode never reads z)."""
    m, tcp0 = _mapper()
    p0 = _settle(m, tcp0, [dict(rwx=0.28, rwy=0.42, zc=0.0)] * 120)
    m2, _ = _mapper()
    p1 = _settle(m2, tcp0, [dict(rwx=0.28, rwy=0.42, zc=-5.0)] * 120)
    assert np.allclose(p0, p1, atol=1e-9)


def test_reach_held_when_left_untracked():
    m, tcp0 = _mapper()
    m.reset(); m.engage(tcp0)
    for k in range(120):
        p = m(_pose(k * 0.033, lwy=0.1, left_ok=False))   # left wrist moved, but not tracked (wrist_other None)
    assert abs(p[0] - tcp0[0]) < 1e-6   # reach (x) holds


def test_confidence_gate():
    m, tcp0 = _mapper()
    m.engage(tcp0)
    assert m(_pose(0.0, visible=False)) is None


def test_box_clamp_flagged():
    m, tcp0 = _mapper()
    p = _settle(m, tcp0, [dict(rwx=-1.0, rwy=-1.0)] * 120)   # absurd reach -> outside the box
    assert m.clipped
    lo = m.ws_min; hi = m.ws_max
    assert np.all(p <= hi + 1e-9) and np.all(p >= lo - 1e-9)


def test_plane_coords_below_hip_and_none():
    assert plane_coords(BodyPose(t=0, wrist=(0, 0, 0), shoulder=(0, 0, 0), skel2d=None)) is None
    up = plane_coords(_pose(0.0, rwy=0.5, hip_y=0.9))
    assert up is not None and not up["below_hip"]
    down = plane_coords(_pose(0.0, rwy=1.2, hip_y=0.9))   # right wrist below the hip line
    assert down["below_hip"]


def test_dropwatch_hysteresis():
    d = DropWatch(0.7)
    assert not d.update(True, 0.0)       # just went below
    assert not d.update(True, 0.5)       # not long enough
    assert d.update(True, 0.8)           # >= 0.7 s below -> drop
    d2 = DropWatch(0.7)
    d2.update(True, 0.0)
    assert not d2.update(False, 0.5)     # came back up before the timer -> no drop
    assert not d2.update(True, 0.9)      # timer restarted


def test_box_reachability_frozen():
    """The dev_arm_real plane box is mostly reachable by the frozen-wrist IK that holds the base-pose orientation."""
    arm, home, cfg = _arm()
    lo = np.array(cfg["safety"]["workspace_min"], float); hi = np.array(cfg["safety"]["workspace_max"], float)
    R = arm.fk(home)[:3, :3]
    ok = tot = 0
    for x in np.linspace(lo[0], hi[0], 5):
        for y in np.linspace(lo[1], hi[1], 5):
            for z in np.linspace(lo[2], hi[2], 5):
                tot += 1
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [x, y, z]
                q = arm.ik(home, T, iters=150, orient="frozen")
                if np.linalg.norm(arm.fk(q)[:3, 3] - np.array([x, y, z])) < 0.004:
                    ok += 1
    assert ok / tot >= 0.9, f"only {100*ok/tot:.0f}% of the box reachable (frozen)"


# --- app wiring -----------------------------------------------------------------

class _App:
    def __init__(self, port, seconds):
        cfg = load_config(arm="dev_arm_real")
        assert cfg["body"]["mode"] == "plane"   # plane is the dev_arm_real default
        cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
        cfg["body"]["clutch"] = {"calib_s": 0.5, "still_m": 0.05, "palm_s": 1.0, "raise_m": 0.05}
        cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"
        self.home = np.array(cfg["arm"]["home"], float)
        self.base = np.array(cfg["arm"]["presets"]["base"], float)
        self.th = threading.Thread(target=app.run, args=(cfg, "mock", seconds), kwargs={"record": False}, daemon=True)
        self.th.start(); time.sleep(0.5)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.s.settimeout(3)
        self.addr = ("127.0.0.1", port)

    def send_pose(self, **kw):
        self.s.sendto(json.dumps(_pose(time.time(), **kw).model_dump()).encode(), self.addr)

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


def test_app_plane_mapping_and_drop_to_base():
    a = _App(47260, seconds=16.0)
    assert a.query()["mapping"] == "PlaneMapper"
    a.send({"kind": "utterance", "t": time.time(), "text": "follow me"})
    for i in range(140):                       # hold still to calibrate, then swing the control hand right + up
        k = max(0, i - 24)
        a.send_pose(rwx=0.35 - min(0.12, k * 0.003), rwy=0.50 - min(0.12, k * 0.003)); time.sleep(1 / 30); a.drain()
    moved = a.query()
    assert moved["clutch"] == "following" and moved["mode"] == "follow"
    off_home = float(np.max(np.abs(np.asarray(moved["q_meas"]) - a.home)))
    assert off_home > 0.03, f"the TCP should have tracked the hand, only {off_home:.3f} rad from home"
    # drop the control arm: the right wrist held below the hip line for >=0.7 s -> return to base
    t_end = time.time() + 6.0
    while time.time() < t_end:
        a.send_pose(rwy=1.3); time.sleep(1 / 30); a.drain()   # right wrist well below the hips
    end = a.query()
    a.join()
    assert end["clutch"] == "idle"
    err = float(np.max(np.abs(np.asarray(end["q_meas"]) - a.base)))
    assert err < 0.03, f"arm should be back at base, still {err:.3f} rad away"


def test_app_plane_left_hand_does_not_engage():
    """In plane mode the left hand drives REACH, so raising it must NOT engage the clutch (it does in arm_mirror)."""
    a = _App(47262, seconds=8.0)
    for i in range(120):                       # left hand raised above the left shoulder the whole time
        a.send_pose(left_ok=True); time.sleep(1 / 30); a.drain()
    st = a.query()
    a.join()
    assert st["clutch"] == "idle", f"left-hand raise must not engage in plane mode (clutch={st['clutch']})"


def test_app_plane_speed_within_limits():
    """Jerk the control hand back and forth; the commanded joints must stay within the profile speed limits."""
    a = _App(47264, seconds=10.0)
    mv = np.array(load_config(arm="dev_arm_real")["arm"]["max_vel"], float)
    a.send({"kind": "utterance", "t": time.time(), "text": "follow me"})
    for i in range(24):
        a.send_pose(); time.sleep(1 / 30); a.drain()
    prev = np.asarray(a.query()["q_meas"]); t_prev = time.time(); worst = 0.0
    for i in range(120):
        rwx = 0.22 if (i // 10) % 2 == 0 else 0.40
        a.send_pose(rwx=rwx, rwy=0.40); time.sleep(1 / 30); a.drain()
        st = a.query(); now = time.time()
        q = np.asarray(st["q_meas"])
        worst = max(worst, float(np.max(np.abs(q - prev) / max(1e-3, now - t_prev) - mv)))
        prev, t_prev = q, now
    a.join()
    assert worst < 0.02, f"a joint exceeded its max_vel by {worst:.3f} rad/s"

"""Plane teleop (body/plane.py) and its app wiring (body.mode = plane, the dev_arm_real default).

Image-plane only: every control is read from skel2d (x, y), normalized by shoulder width; MediaPipe depth (z) is
never used. Mapper tests: engage never jumps, lateral/vertical signs/scales are right, the target is invariant to the
operator's distance (skeleton scale) and to depth (world z), and targets outside the box are clamped + flagged.
Reach is a JOYSTICK RATE from the LEFT-hand THUMB (reach_mode "rate", the dev_arm_real default): thumb up reaches
forward at +max_speed, thumb down reaches back at -max_speed, anything else holds; a short dwell stops flicker, a
lost reach hand holds at once, and the box wall stops the rate without wind-up (reversing frees it). reach_mode
"position" keeps the old height->x position map (legacy). App tests: dev_arm_real in plane mode follows synthetic
poses and reports reach state, the
left-hand-raise clutch is DISABLED, dropping the control arm (wrist below the hip for >=0.7 s) returns the arm to
base, and the joints never exceed the profile speed limits. Plus a reachability check of the box.
"""
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from rover_intent import app
from rover_intent.body.plane import DropWatch, PlaneMapper, plane_coords
from rover_intent.config import load_config
from rover_intent.control.kinematics import ArmModel
from rover_intent.types import BodyPose

from test_thumb import thumb_hand

# thumb= kwarg for _pose -> the reach hand's 21 landmarks (reach_hand2d). None = reach hand not found (reach holds);
# "open" = an open hand / wave (present but NOT a thumbs-up); "fist" = closed fist; "side" = a sideways thumb.
_THUMB_HANDS = {"up": ("up", "curled"), "down": ("down", "curled"), "open": ("up", "extended"),
                "fist": ("tuck", "curled"), "side": ("side", "curled")}

ROOT = Path(__file__).resolve().parents[1]
URDF = str(ROOT / "assets/dev_arm/dev_arm_gripper.urdf")


def _arm():
    cfg = load_config(arm="dev_arm_real")
    a = cfg["arm"]
    arm = ArmModel(URDF, base=a["base_link"], tip=a["tip_link"], tcp_offset=a["tcp_offset"], tool_axis=a["tool_axis"])
    arm.lower = np.array(a["joint_limits_lower"], float)
    arm.upper = np.array(a["joint_limits_upper"], float)
    return arm, np.array(a["home"], float), cfg


def _skel(rwx=0.35, rwy=0.50, lwy=0.62, hip_y=0.90, scale=1.0):
    """skel2d in body-client SKEL order, scaled about the image centre (0.5) to fake a nearer/farther operator.
    Right side at the SMALLER x (the mirror-corrected convention). order: Lsh Rsh Lel Rel Lwr Rwr Lhip Rhip nose.
    lwy (left-wrist height) only matters for the legacy reach_mode "position"; rate reach is the LEFT-hand THUMB now
    (pass thumb=... to _pose), so skel2d alone (no reach_hand2d) means the reach holds."""
    def sc(x, y):
        return [0.5 + (x - 0.5) * scale, 0.5 + (y - 0.5) * scale]
    return [sc(0.65, 0.5), sc(0.35, 0.5), sc(0.70, 0.65), sc(0.30, 0.65), sc(0.68, lwy), sc(rwx, rwy),
            sc(0.60, hip_y), sc(0.40, hip_y), sc(0.50, 0.30)]


def _pose(t, rwx=0.35, rwy=0.50, lwy=0.62, hip_y=0.90, scale=1.0, visible=True, left_ok=True, zc=0.0, thumb=None):
    reach_hand2d = None if thumb in (None, "none") else thumb_hand(*_THUMB_HANDS[thumb])
    return BodyPose(t=t, wrist=(0.1, 0.0, zc), shoulder=(0.0, 0.0, zc), elbow=(0.05, 0.1, zc),
                    shoulder_other=(-0.2, 0.0, zc), hips=(0.0, 0.3, zc),
                    skel2d=_skel(rwx, rwy, lwy, hip_y, scale), reach_hand2d=reach_hand2d, visible=visible,
                    wrist_other=(-0.2, -0.1, zc) if left_ok else None)


def _mapper(reach_mode=None):
    arm, home, cfg = _arm()
    pcfg = dict(cfg.get("plane") or {})
    if reach_mode is not None:
        pcfg["reach_mode"] = reach_mode
    m = PlaneMapper(arm, home, pcfg, cfg["safety"]["workspace_min"], cfg["safety"]["workspace_max"])
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
    """Lateral + vertical are a POSITION map from the right hand (mode-independent). Reach is covered separately
    (rate tests below; the old position-reach signs in test_reach_position_mode)."""
    m, tcp0 = _mapper()
    # right hand to the operator's RIGHT (smaller x) -> gripper to their right = robot -y
    pr = _settle(m, tcp0, [dict(rwx=0.25)] * 80)
    assert pr[1] < tcp0[1] - 0.02
    # right hand UP (smaller image y) -> robot +z
    pu = _settle(m, tcp0, [dict(rwy=0.35)] * 80)
    assert pu[2] > tcp0[2] + 0.02


def test_reach_position_mode():
    """reach_mode: position keeps the old height->x position map (still selectable). LEFT hand up -> forward,
    down -> back, relative to the engage height."""
    m, tcp0 = _mapper(reach_mode="position")
    pf = _settle(m, tcp0, [dict(lwy=0.35)] * 80)   # left hand up
    assert pf[0] > tcp0[0] + 0.02
    m2, _ = _mapper(reach_mode="position")
    pb = _settle(m2, tcp0, [dict(lwy=0.85)] * 80)  # left hand down
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


def test_reach_held_when_reach_hand_untracked():
    m, tcp0 = _mapper()
    m.reset(); m.engage(tcp0)
    p = None
    for k in range(120):
        p = m(_pose(k * 0.033))   # no reach-hand landmarks (reach hand not found) -> reach holds
    assert abs(p[0] - tcp0[0]) < 1e-6 and m.reach_state == "hold"


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
    ok = tot = floor_ok = floor_tot = 0
    for x in np.linspace(lo[0], hi[0], 5):
        for y in np.linspace(lo[1], hi[1], 5):
            for z in np.linspace(lo[2], hi[2], 5):
                tot += 1
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [x, y, z]
                q = arm.ik(home, T, iters=150, orient="frozen")
                hit = np.linalg.norm(arm.fk(q)[:3, 3] - np.array([x, y, z])) < 0.004
                ok += hit
                if z == lo[2]:
                    floor_tot += 1
                    floor_ok += hit
    # Since the floor came down to table + 5 cm (2026-10-28) the box includes a real frozen-wrist pocket at its back
    # edge (x 0.47, z ~0.08-0.43: the elbow would have to fold; straight-line paths from home miss it too), so the
    # whole box is ~88%. The floor layer (where the table objects are) must be fully reachable.
    # board surface (2026-10-28, floor 0.045): the back-edge pocket reaches the floor at x 0.47 (4/25 points)
    assert floor_ok / floor_tot >= 0.80, f"only {floor_ok}/{floor_tot} of the floor layer reachable (frozen)"
    assert ok / tot >= 0.85, f"only {100*ok/tot:.0f}% of the box reachable (frozen)"


# --- rate reach (joystick reach from the LEFT-hand THUMB) ------------------------
# reach_mode "rate" (the dev_arm_real default) reads the reach hand's thumb (BodyPose.reach_hand2d, built by
# _pose(thumb=...)): thumb up -> +max_speed (forward), down -> -max_speed (back), else hold. A dwell stops flicker.

def test_reach_rate_no_thumb_holds():
    m, tcp0 = _mapper()                                   # rate is the dev_arm_real default
    p = _settle(m, tcp0, [dict()] * 120)                  # no reach hand sent -> reach holds
    assert abs(p[0] - tcp0[0]) < 1e-6
    assert m.reach_rate == 0.0 and m.reach_state == "hold" and m.thumb_state == "none"
    # a hand that is present but NOT a thumbs-up (open hand / wave) also holds
    m2, _ = _mapper()
    p2 = _settle(m2, tcp0, [dict(thumb="open")] * 120)
    assert abs(p2[0] - tcp0[0]) < 1e-6 and m2.reach_state == "hold"


def test_reach_rate_forward_and_back():
    m, tcp0 = _mapper()
    pf = _settle(m, tcp0, [dict(thumb="up")] * 120)       # thumb up -> reach forward
    assert pf[0] > tcp0[0] + 0.02 and m.reach_state == ">>" and m.reach_rate > 0 and m.thumb_state == "up"
    m2, _ = _mapper()
    pb = _settle(m2, tcp0, [dict(thumb="down")] * 120)    # thumb down -> reach back
    assert pb[0] < tcp0[0] - 0.02 and m2.reach_state == "<<" and m2.reach_rate < 0 and m2.thumb_state == "down"


def test_reach_rate_full_speed_not_scaled():
    """Thumb reach is bang-bang: once committed it runs at exactly max_speed (no distance scaling)."""
    max_speed = _arm()[2]["plane"]["reach_rate"]["max_speed"]
    m, tcp0 = _mapper()
    _settle(m, tcp0, [dict(thumb="up")] * 80)
    assert m.reach_rate == pytest.approx(max_speed, abs=1e-6)
    mb, _ = _mapper()
    _settle(mb, tcp0, [dict(thumb="down")] * 80)
    assert mb.reach_rate == pytest.approx(-max_speed, abs=1e-6)


def test_reach_rate_dwell_anti_flicker():
    """A direction must hold ~0.2 s to start, and a flick (then release) never jogs the arm."""
    m, tcp0 = _mapper()
    m.reset(); m.engage(tcp0)
    for k in range(80):                                   # still prefix, no reach hand
        m(_pose(k * 0.033))
    t = 80 * 0.033
    for _ in range(3):                                    # a brief flick of thumb up (~0.1 s)
        m(_pose(t, thumb="up")); t += 0.033
    assert m.reach_state == "hold"                        # dwell not met -> no reach
    for _ in range(3):                                    # released -> dwell resets
        m(_pose(t)); t += 0.033
    assert m.reach_state == "hold"
    for _ in range(20):                                   # now a sustained thumb up -> commits
        m(_pose(t, thumb="up")); t += 0.033
    assert m.reach_state == ">>"


def test_reach_rate_tracking_loss_holds():
    m, tcp0 = _mapper()
    _settle(m, tcp0, [dict(thumb="up")] * 60)             # reaching forward
    x_before = m(_pose(0.033 * 140, thumb="up"))[0]
    p = None
    for i in range(80):
        p = m(_pose(0.033 * (141 + i)))                   # reach hand lost (no reach_hand2d) -> hold at once
    assert abs(p[0] - x_before) < 1e-6 and m.reach_rate == 0.0 and m.reach_state == "hold"


def test_reach_rate_box_stop_no_windup():
    m, tcp0 = _mapper()
    p = _settle(m, tcp0, [dict(thumb="up")] * 400)        # pile into the forward wall for a long time
    assert m.clipped and p[0] == pytest.approx(m.ws_max[0], abs=1e-9)
    assert m._reach_off <= (m.ws_max[0] - tcp0[0]) + 1e-9  # offset pinned at the wall, no accumulation
    # reverse (thumb down): the arm must leave the wall promptly (not sit out an accumulated overshoot)
    pr = None
    for i in range(120):
        pr = m(_pose(0.033 * (480 + i), thumb="down"))
    assert pr[0] < m.ws_max[0] - 0.03 and not m.clipped


def test_reach_rate_distance_invariant():
    """Standing nearer/farther (skeleton scale) must not change the thumb reach (the classifier is palm-normalized)."""
    xs = []
    for scale in (0.5, 1.0, 2.0):
        m, tcp0 = _mapper()
        p = _settle(m, tcp0, [dict(thumb="up", scale=scale)] * 60)
        xs.append(p[0])
    assert np.allclose(xs[0], xs[1], atol=1e-6) and np.allclose(xs[1], xs[2], atol=1e-6)


def test_reach_rate_sign_flip():
    """sign.reach = -1 flips the reach direction (camera / preference): thumb up then drives BACK."""
    arm, home, cfg = _arm()
    pcfg = dict(cfg["plane"]); pcfg["sign"] = dict(pcfg["sign"]); pcfg["sign"]["reach"] = -1.0
    m = PlaneMapper(arm, home, pcfg, cfg["safety"]["workspace_min"], cfg["safety"]["workspace_max"])
    tcp0 = arm.fk(home)[:3, 3]
    m.reset(); m.engage(tcp0)
    p = None
    for k in range(80):
        p = m(_pose(k * 0.033))
    for i in range(120):
        p = m(_pose(0.033 * (80 + i), thumb="up"))        # thumb UP, but sign flipped -> reach goes BACK
    assert p[0] < tcp0[0] - 0.02 and m.reach_rate < 0


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
    # the plane rate-reach state is reported for the HUD; the left hand sat in the dead zone -> "hold"
    assert moved["reach_state"] == "hold" and isinstance(moved["reach_rate"], (int, float))
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

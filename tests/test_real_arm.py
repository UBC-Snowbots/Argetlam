"""dev_arm_real profile and the body-teleop fixes it needs: axis-orient IK (issue 2), joint-limit override
(issue 3), the Retargeter engage glide (issue 1)."""
import numpy as np
import pytest
from pathlib import Path

from rover_intent.config import load_config
from rover_intent.control.kinematics import ArmModel, axis_angle, axis_err, rot_err

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real():
    cfg = load_config(arm="dev_arm_real")
    a = cfg["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), base=a["base_link"], tip=a["tip_link"],
                   tcp_offset=a["tcp_offset"], tool_axis=a.get("tool_axis"))
    if a.get("joint_limits_lower") is not None:
        arm.lower = np.array(a["joint_limits_lower"], float)
        arm.upper = np.array(a["joint_limits_upper"], float)
    return cfg, a, arm


# ---- issue 2: axis-orient IK frees the wrist roll -----------------------------------------------
def test_axis_orient_ignores_roll_but_full_chases_it():
    a = load_config(arm="dev_arm_real")["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), base=a["base_link"], tip=a["tip_link"],
                   tcp_offset=a["tcp_offset"], tool_axis=a["tool_axis"])
    home = np.array(a["home"])
    T0 = arm.fk(home)
    point = T0[:3, :3] @ arm.tool_axis            # world tool-pointing direction
    T_roll = T0.copy()
    T_roll[:3, :3] = axis_angle(point, 1.0) @ T0[:3, :3]   # 1 rad roll about the pointing axis, same position
    # the pointing direction is unchanged by a roll about it -> axis error is ~0
    assert np.linalg.norm(axis_err(T0[:3, :3] @ arm.tool_axis, T_roll[:3, :3] @ arm.tool_axis)) < 1e-6
    q_axis = home.copy()
    for _ in range(150):
        q_axis = arm.ik_step(q_axis, T_roll, orient="axis")
    q_full = home.copy()
    for _ in range(150):
        q_full = arm.ik_step(q_full, T_roll, orient="full")
    assert np.max(np.abs(q_axis - home)) < 1e-3          # axis mode: barely moves (roll is free)
    assert np.max(np.abs(q_full - home)) > 0.2           # full mode: spins the wrist to chase the roll


def test_axis_orient_still_reaches_a_translated_target():
    a = load_config(arm="dev_arm_real")["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), base=a["base_link"], tip=a["tip_link"],
                   tcp_offset=a["tcp_offset"], tool_axis=a["tool_axis"])
    home = np.array(a["home"])
    T = arm.fk(home).copy()
    T[:3, 3] = T[:3, 3] + np.array([0.02, 0.03, -0.04])
    q = home.copy()
    for _ in range(300):
        q = arm.ik_step(q, T, orient="axis")
    assert np.linalg.norm(arm.fk(q)[:3, 3] - T[:3, 3]) < 2e-3
    pt = arm.fk(q)[:3, :3] @ arm.tool_axis
    assert np.linalg.norm(axis_err(pt, T[:3, :3] @ arm.tool_axis)) < 0.03   # still points the tool the right way


# ---- issue 3: joint-limit override, and the profile is self-consistent -------------------------
def test_profile_home_presets_box_and_limits(real):
    cfg, a, arm = real
    urdf_lo = np.array([-1.5, -1.9, -1.5, -1.5, -2.0, -10.0])
    urdf_hi = np.array([1.5, 1.0, 1.5, 1.5, 1.05, 10.0])
    assert np.all(arm.lower >= urdf_lo - 1e-9) and np.all(arm.upper <= urdf_hi + 1e-9)  # tighter than URDF
    home = np.array(a["home"])
    assert np.all(home >= arm.lower) and np.all(home <= arm.upper)
    for q in a["presets"].values():
        q = np.asarray(q, float)
        assert np.all(q >= arm.lower) and np.all(q <= arm.upper)
    lo, hi = np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"])
    htcp = arm.fk(home)[:3, 3]
    assert np.all(htcp >= lo) and np.all(htcp <= hi)            # home TCP inside the box (issue 1)
    assert htcp[2] > 0.45                                        # home is high (z ~0.52), above the sim box ceiling
    # the real home does NOT point straight down (that is why it left the sim box)
    assert not np.allclose(arm.fk(home)[:3, :3] @ arm.tool_axis, [0, 0, -1], atol=0.2)


def test_max_vel_is_a_fraction_of_firmware_and_matches_follower_default(real):
    cfg, a, arm = real
    fw = np.radians([80, 40, 60, 110, 90, 90])
    frac = np.array(a["max_vel"]) / fw
    assert np.allclose(frac, frac[0], atol=1e-3)               # one common fraction across joints
    assert abs(frac[0] - 0.2) < 1e-2                            # == the sim_to_real follower default --speed
    assert np.allclose(a["max_acc"], np.radians(100), atol=1e-2)


def test_joint_limit_override_flows_through_app_build():
    from rover_intent import app
    cfg = load_config(arm="dev_arm_real")
    arm, safety, *_ = app.build(cfg, "mock")
    assert np.allclose(arm.lower, cfg["arm"]["joint_limits_lower"])
    assert np.allclose(arm.upper, cfg["arm"]["joint_limits_upper"])
    assert np.allclose(safety.lower, cfg["arm"]["joint_limits_lower"])


# ---- issue 1: the plain Retargeter glides on engage (no jump) ----------------------------------
def test_retarget_engage_glides_instead_of_jumping():
    from rover_intent.body.retarget import Retargeter
    from rover_intent.types import BodyPose
    cfg = load_config()["body"]
    r = Retargeter(cfg["axes"], scale=1.0, origin=[0.45, 0, 0.45], min_cutoff=1e6, glide_mps=0.15)
    # engage with a neutral that is far from the hand, so the first mapped point is ~0.3 m from the arm
    r.engage([0.45, 0.0, 0.45], neutral=[0.0, 0.0, 0.0])
    pose = lambda w, t: BodyPose(t=t, wrist=w, shoulder=(0, 0, 0))
    far = (0.0, 0.3, 0.0)                     # maps ~0.3 m away (axes/scale 1.0)
    p0 = r(pose(far, 0.0))
    p1 = r(pose(far, 0.05))                   # 50 ms later -> at most glide_mps*dt = 7.5 mm further
    p2 = r(pose(far, 0.10))
    assert np.allclose(p0, [0.45, 0.0, 0.45], atol=1e-6)      # starts exactly at the arm, no jump
    assert np.linalg.norm(p1 - p0) <= 0.15 * 0.05 + 1e-6
    assert np.linalg.norm(p2 - p0) > np.linalg.norm(p1 - p0)  # keeps easing toward the target
    # eventually arrives and then tracks 1:1
    for i in range(200):
        last = r(pose(far, 0.15 + 0.03 * i))
    target = r.anchor + r.gains * (r.A @ np.array(far))
    assert np.linalg.norm(last - target) < 1e-3


def test_retarget_no_glide_when_first_sample_is_neutral():
    from rover_intent.body.retarget import Retargeter
    from rover_intent.types import BodyPose
    cfg = load_config()["body"]
    r = Retargeter(cfg["axes"], scale=1.0, origin=[0.5, 0.1, 0.3], min_cutoff=1e6)
    r.engage([0.5, 0.1, 0.3])                 # neutral=None -> first sample becomes neutral == the arm, glide ends at once
    pose = lambda w, t: BodyPose(t=t, wrist=w, shoulder=(0, 0, 0))
    assert np.allclose(r(pose((0.0, 0.5, 0.0), 0.0)), [0.5, 0.1, 0.3])
    assert np.allclose(r(pose((0.0, 0.4, -0.1), 0.1)), [0.6, 0.1, 0.4])   # tracks immediately, as before

"""dev_arm_real profile and the body-teleop fixes it needs: axis-orient IK (issue 2), joint-limit override
(issue 3)."""
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


def test_max_vel_matches_what_the_real_arm_reaches_at_the_follower_cap(real):
    cfg, a, arm = real
    from rover_intent.control.stepper_map import MEASURED_VEL_RATIO
    fw = np.radians([80, 40, 60, 110, 90, 90])
    # sim limit = what the real arm actually reaches at the follower's default 0.2 x firmware command (the firmware
    # runs 'V' at ~half speed, bench sweep 2026-10-23), so the sim can't outrun the arm
    assert np.allclose(np.array(a["max_vel"]), 0.2 * fw * np.array(MEASURED_VEL_RATIO), rtol=2e-3)
    assert np.all(np.array(a["max_vel"]) < 0.2 * fw)
    assert np.allclose(a["max_acc"], np.radians(100), atol=1e-2)


def test_joint_limit_override_flows_through_app_build():
    from rover_intent import app
    cfg = load_config(arm="dev_arm_real")
    arm, safety, *_ = app.build(cfg, "mock")
    assert np.allclose(arm.lower, cfg["arm"]["joint_limits_lower"])
    assert np.allclose(arm.upper, cfg["arm"]["joint_limits_upper"])
    assert np.allclose(safety.lower, cfg["arm"]["joint_limits_lower"])

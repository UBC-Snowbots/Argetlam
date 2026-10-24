"""Named arm poses for the body-client button panel: plan checks for joint-space preset moves, and the hardware
home request. The move itself runs through the app's normal SafetyLayer (vel/acc limits) in HOLD mode.

Presets come from the arm profile (`arm.presets: {name: [6 joint angles, rad]}`); "return" always means the
profile's start pose (`arm.home`). Hardware home is only offered when `real_arm.enabled` is true.
"""
from __future__ import annotations

import shlex
import subprocess

import numpy as np

RETURN = "return"
HOME_HW = "home_hw"
PATH_SAMPLES = 24
BOX_SLACK = 0.01  # m: a path may run this much further outside the workspace box than its worse end does


def preset_names(presets: dict, real_arm_enabled: bool) -> list[str]:
    names = [RETURN] + [n for n in presets if n != RETURN]
    return names + ([HOME_HW] if real_arm_enabled else [])


def outside(p, ws_min, ws_max) -> float:
    """Distance (m) of a point outside the workspace box (0 inside)."""
    p = np.asarray(p, float)
    return float(np.linalg.norm(p - np.clip(p, ws_min, ws_max)))


def check_joint_goal(goal, q_now, clutch_state: str, arm, ws_min, ws_max, label: str = "goal"):
    """Check an ARBITRARY joint-space goal (a preset, or a voice Cartesian/joint jog). Returns (goal or None, msg).

    This is the shared safety check behind both the button-panel presets and the voice motion commands
    (planner/fast_path.py): the clutch must be idle, the goal must be 6 finite angles inside the joint limits, and
    the straight joint-space path must not leave the workspace box further than its start or end does (so a move
    from outside the box back into it is allowed, but never one that wanders out)."""
    if clutch_state != "idle":
        return None, f"refused: clutch is {clutch_state} (release it first)"
    goal = np.asarray(goal, float)
    if goal.shape != (6,) or not np.all(np.isfinite(goal)):
        return None, f"refused: {label} must be 6 finite joint angles"
    lo, hi = np.asarray(arm.lower), np.asarray(arm.upper)
    if np.any(goal < lo - 1e-6) or np.any(goal > hi + 1e-6):
        bad = [i + 1 for i in range(6) if not lo[i] - 1e-6 <= goal[i] <= hi[i] + 1e-6]
        return None, f"refused: {label} is outside the joint limits on joints {bad}"
    q_now = np.asarray(q_now, float)
    d_start = outside(arm.fk(q_now)[:3, 3], ws_min, ws_max)
    d_goal = outside(arm.fk(goal)[:3, 3], ws_min, ws_max)
    allowed = max(d_start, d_goal) + BOX_SLACK
    for s in np.linspace(0.0, 1.0, PATH_SAMPLES + 1)[1:]:
        p = arm.fk(q_now + s * (goal - q_now))[:3, 3]
        if outside(p, ws_min, ws_max) > allowed:
            return None, (f"refused: the path to {label} leaves the workspace box "
                          f"(at {np.round(p, 3).tolist()}, {s:.0%} of the way)")
    return goal, f"moving to {label}"


def plan_preset(name: str, presets: dict, home, q_now, clutch_state: str, arm, ws_min, ws_max):
    """Check a joint-space preset move by name. Returns (goal or None, message).

    Resolves the name (RETURN = the profile start pose; otherwise a profile preset) to a goal, then runs the
    shared check_joint_goal checks. Refused if the clutch is engaged, the name is unknown, the goal is outside the
    joint limits, or the straight joint-space path would leave the workspace box further than its start or end does."""
    if clutch_state != "idle":
        return None, f"refused: clutch is {clutch_state} (release it first)"
    if name == RETURN:
        goal = np.asarray(home, float)
    elif name in presets:
        goal = np.asarray(presets[name], float)
    else:
        return None, f"refused: unknown preset {name!r}"
    return check_joint_goal(goal, q_now, clutch_state, arm, ws_min, ws_max, label=name)


def at_goal(q_cmd, q_meas, goal, tol: float = 0.01, tol_cmd: float = 0.003) -> bool:
    """tol_cmd > the SafetyLayer's deliberate stop-short margin (max_acc * dt^2, about 0.0012 rad at 3 rad/s^2, 20 ms)."""
    return bool(np.max(np.abs(np.asarray(q_cmd) - goal)) < tol_cmd and np.max(np.abs(np.asarray(q_meas) - goal)) < tol)


def request_hw_home(real_arm: dict, confirm: bool, dry_run: bool = False):
    """Send the real arm's home-all command (Rowan's stepper_arm_driver, via ROS). Returns (ok, message).
    Needs real_arm.enabled and an explicit confirm. Runs `real_arm.home_cmd` in the background."""
    if not real_arm.get("enabled", False):
        return False, "refused: no real arm configured (real_arm.enabled is false)"
    if not confirm:
        return False, "refused: home needs an explicit confirm"
    cmd = real_arm.get("home_cmd")
    if not cmd:
        return False, "refused: real_arm.home_cmd is not set"
    if dry_run:
        return True, f"would run: {cmd}"
    subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return True, "home sent to the real arm: every axis drives to its limit switch, then back to base"


# -w 1 waits for one matching subscription before publishing, so a one-shot command is not lost to the DDS
# discovery race (and is captured by a running record_real.sh recorder). The driver subscribes to /arm/command.
HOME_CMD_EXAMPLE = ("source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash "
                    "&& ros2 topic pub --once -w 1 /arm/command rover_msgs/msg/ArmCommand "
                    + shlex.quote("{cmd_type: 72, cmd_value: 170}"))

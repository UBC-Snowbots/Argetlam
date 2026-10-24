"""Pure-python logic for the bench characterization scripts (no ROS, no numpy), so it is unit-testable with fakes.

The scripts in this directory (axis_sweep, axis4_check, drift_check, end_check) are thin rclpy wrappers around these
functions. Every velocity the scripts command goes through clamp_vel here, which hard-caps at SPEED_CAP (0.2) of the
firmware max speed, so the cap cannot be bypassed by a wrong argument. Mapping/limits come from control.stepper_map.
"""
from __future__ import annotations

import os
import signal
import time

from rover_intent.control.stepper_map import DEFAULT_TABLE, in_travel

SPEED_CAP = 0.2   # hard fraction of firmware max speed the bench may command, ever


def cap_speed_frac(frac: float) -> float:
    return min(max(float(frac), 0.0), SPEED_CAP)


def vel_cap_deg_s(axis: int, frac: float, table=DEFAULT_TABLE) -> float:
    """Allowed |velocity| on an axis = min(frac, SPEED_CAP) * firmware max speed."""
    return table.fw_max_vel[axis] * cap_speed_frac(frac)


def clamp_vel(v: float, axis: int, frac: float, table=DEFAULT_TABLE) -> float:
    cap = vel_cap_deg_s(axis, frac, table)
    return max(-cap, min(cap, float(v)))


def start_pose_ok(fw, table=DEFAULT_TABLE, margin: float = 5.0):
    """Start pose must be inside firmware travel [margin, max-margin] on all 6 arm axes. Returns (ok, [bad axes])."""
    oks = in_travel(fw, table, margin)
    bad = [i + 1 for i, o in enumerate(oks) if not o]
    return (len(bad) == 0, bad)


def p_step(target_fw: float, cur_fw: float, axis: int, frac: float, kp: float = 3.0, deadband: float = 0.5,
           table=DEFAULT_TABLE) -> float:
    """One capped P-loop velocity command (deg/s) toward target_fw; 0 inside the deadband. Used by end_check."""
    e = target_fw - cur_fw
    if abs(e) <= deadband:
        return 0.0
    return clamp_vel(kp * e, axis, frac, table)


def sweep_plan(axis: int, steps_deg_s, hold_s: float, frac: float, settle_s: float = 0.5, table=DEFAULT_TABLE):
    """Symmetric velocity-step plan around the current pose for one axis: for each magnitude s in steps_deg_s a
    +s hold, a -s hold, then a 0 settle. Every velocity is clamped to the bench cap. Returns [(duration_s, vel), ...]."""
    plan = []
    for s in steps_deg_s:
        v = clamp_vel(s, axis, frac, table)
        plan.append((hold_s, v))
        plan.append((hold_s, -v))
        plan.append((settle_s, 0.0))
    return plan


def _deriv(ts, xs):
    """Backward-difference derivative sampled at segment midpoints: returns (tmid, dx/dt) lists."""
    tm, dv = [], []
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt <= 0:
            continue
        tm.append(0.5 * (ts[i] + ts[i - 1]))
        dv.append((xs[i] - xs[i - 1]) / dt)
    return tm, dv


def estimate_axis_response(ts, fw, v_cmd):
    """From a commanded-velocity step (|v_cmd| deg/s) and the feedback (ts seconds, fw degrees), estimate:
      max_vel   : max |d fw/dt| reached (deg/s)
      accel     : max |d2 fw/dt2| (deg/s^2) -- use to replace the max_acc placeholder in dev_arm_real
      lag_s     : time from ts[0] until |velocity| first reaches 50% of |v_cmd| (None if never)
    Returns a dict; robust to short arrays (returns zeros / None)."""
    if len(ts) < 3:
        return {"max_vel": 0.0, "accel": 0.0, "lag_s": None, "n": len(ts)}
    tm, vel = _deriv(ts, fw)
    max_vel = max((abs(v) for v in vel), default=0.0)
    _, acc = _deriv(tm, vel)
    accel = max((abs(a) for a in acc), default=0.0)
    lag_s = None
    thresh = 0.5 * abs(v_cmd)
    if thresh > 0:
        for t, v in zip(tm, vel):
            if abs(v) >= thresh:
                lag_s = t - ts[0]
                break
    return {"max_vel": max_vel, "accel": accel, "lag_s": lag_s, "n": len(ts)}


def axis4_correction(commanded_fw_delta: float, measured_real_delta: float):
    """Axis 4's gear ratio is suspect. We command +commanded_fw_delta firmware degrees; Pranav measures
    measured_real_delta real degrees on the physical axis. Returns a dict:
      ratio_real_per_fw = measured/commanded   (how many real degrees per commanded firmware degree)
      correction        = commanded/measured   (multiply a desired real move, in fw deg, by this to actually get it)
    e.g. command 30 fw deg, measure 20 real deg -> ratio 0.667, correction 1.5 (command 1.5x to hit the target)."""
    if commanded_fw_delta == 0:
        raise ValueError("commanded_fw_delta must be non-zero")
    ratio = measured_real_delta / commanded_fw_delta
    correction = (commanded_fw_delta / measured_real_delta) if measured_real_delta else float("inf")
    return {"ratio_real_per_fw": ratio, "correction": correction}


def per_axis_drift(ts, fw_rows):
    """Least-squares slope (deg/s) per axis over a no-command recording. fw_rows[i] is the 6/7-list at ts[i].
    Returns a list of slopes (one per axis present). Robust to constant / short data (slope 0)."""
    n = len(ts)
    if n < 2:
        return [0.0] * (len(fw_rows[0]) if fw_rows else 0)
    tbar = sum(ts) / n
    stt = sum((t - tbar) ** 2 for t in ts)
    naxes = len(fw_rows[0])
    slopes = []
    for a in range(naxes):
        if stt <= 0:
            slopes.append(0.0)
            continue
        xbar = sum(row[a] for row in fw_rows) / n
        sxt = sum((ts[i] - tbar) * (fw_rows[i][a] - xbar) for i in range(n))
        slopes.append(sxt / stt)
    return slopes


def stop_group(proc, waits=(5.0, 5.0, 2.0)):
    """Stop record_real.sh and EVERYTHING it started (ros2 bag + the type-pin node share its process group).
    `ros2 bag record` here ignores SIGINT (2026-10-23 bench: it kept running and held the caller's stdout open, so
    the sweep looked hung), so escalate SIGINT -> SIGTERM -> SIGKILL on the whole group."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, wait in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), waits):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        t_end = time.monotonic() + wait
        while time.monotonic() < t_end:
            proc.poll()                     # reap the bash leader so it doesn't linger as a zombie in the group
            if not group_alive(pgid):
                return
            time.sleep(0.1)


def group_alive(pgid: int) -> bool:
    """True if any process in group `pgid` is still running. Killed children can sit as zombies (state Z) until
    their new parent reaps them; they hold no files or pipes, so they don't count. Linux /proc, else killpg probe."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
    for e in entries:
        if not e.isdigit():
            continue
        try:
            with open(f"/proc/{e}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()      # after "pid (comm) ": state ppid pgrp ...
        if int(fields[2]) == pgid and fields[0] != "Z":
            return True
    return False

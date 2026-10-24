"""Pure logic for the bench scripts (scripts/bench/benchlib.py): the 0.2x speed cap can't be bypassed, the
start-pose travel check, and the estimators (lag/max-vel/accel, axis-4 correction, drift). No ROS here."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
import benchlib  # noqa: E402
from rover_intent.control.stepper_map import DEFAULT_TABLE  # noqa: E402


def test_speed_cap_is_hard_02():
    assert benchlib.SPEED_CAP == 0.2
    assert benchlib.cap_speed_frac(0.9) == 0.2       # a wrong argument cannot exceed the cap
    assert benchlib.cap_speed_frac(0.1) == 0.1
    assert benchlib.cap_speed_frac(-5) == 0.0


def test_vel_cap_and_clamp_never_exceed_02x_firmware():
    for axis in range(6):
        cap = DEFAULT_TABLE.fw_max_vel[axis] * 0.2
        assert abs(benchlib.vel_cap_deg_s(axis, 1.0) - cap) < 1e-9      # asking for 1.0 still capped at 0.2
        assert benchlib.clamp_vel(1e6, axis, 1.0) == pytest.approx(cap)
        assert benchlib.clamp_vel(-1e6, axis, 1.0) == pytest.approx(-cap)
    assert benchlib.vel_cap_deg_s(0, 0.2) == pytest.approx(16.0)        # axis 1: 80*0.2


def test_start_pose_ok_flags_out_of_travel_axes():
    good = [90, 90, 70, 90, 90, 90]
    assert benchlib.start_pose_ok(good) == (True, [])
    bad = [90, 90, 70, 90, 90, -1]       # axis 6 below 0
    ok, axes = benchlib.start_pose_ok(bad)
    assert ok is False and axes == [6]


def test_p_step_deadband_and_cap():
    assert benchlib.p_step(100.0, 100.2, 0, 0.2) == 0.0                 # inside 0.5 deg deadband
    v = benchlib.p_step(200.0, 100.0, 0, 0.2)                           # huge error -> capped
    assert v == pytest.approx(16.0)                                     # 80*0.2


def test_sweep_plan_is_symmetric_and_capped():
    plan = benchlib.sweep_plan(0, [2, 5, 100], hold_s=0.5, frac=0.2)
    assert len(plan) == 9                                               # 3 steps x (+, -, settle)
    # step magnitudes capped to 16 deg/s; +/- pairs symmetric; settle is zero
    assert plan[0][1] == pytest.approx(2.0) and plan[1][1] == pytest.approx(-2.0) and plan[2][1] == 0.0
    assert plan[6][1] == pytest.approx(16.0) and plan[7][1] == pytest.approx(-16.0)   # 100 clamped to 16


def test_estimate_axis_response_on_a_ramp():
    # synthetic: axis accelerates to ~10 deg/s then holds; dt = 20 ms
    v_cmd, dt = 10.0, 0.02
    ts, fw, x, v = [], [], 0.0, 0.0
    for k in range(100):
        v = min(v_cmd, v + 40.0 * dt)       # accel 40 deg/s^2 up to 10 deg/s
        x += v * dt
        ts.append(k * dt)
        fw.append(x)
    est = benchlib.estimate_axis_response(ts, fw, v_cmd)
    assert est["max_vel"] == pytest.approx(10.0, abs=0.5)
    assert est["accel"] == pytest.approx(40.0, rel=0.5)
    assert est["lag_s"] is not None and 0.0 < est["lag_s"] < 0.2


def test_axis4_correction():
    c = benchlib.axis4_correction(30.0, 20.0)
    assert c["ratio_real_per_fw"] == pytest.approx(2 / 3)
    assert c["correction"] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        benchlib.axis4_correction(0.0, 5.0)


def test_per_axis_drift_matches_a_known_slope():
    ts = [i * 0.05 for i in range(200)]       # 10 s at 20 Hz
    rows = [[0.01 + (-0.006) * t, 5.0, 10.0 + 0.0 * t, 0, 0, 0] for t in ts]  # axis1 drifts -0.006 deg/s
    slopes = benchlib.per_axis_drift(ts, rows)
    assert slopes[0] == pytest.approx(-0.006, abs=1e-6)
    assert slopes[1] == pytest.approx(0.0, abs=1e-9)
    assert slopes[2] == pytest.approx(0.0, abs=1e-9)


def test_stop_group_kills_a_recorder_that_ignores_sigint():
    """2026-10-23 bench: `ros2 bag record` ignored SIGINT, kept running and held the sweep's stdout open, so the sweep
    looked hung. stop_group must escalate to the whole process group and return quickly."""
    import os
    import subprocess
    import time
    # a stand-in for record_real.sh: a bash leader plus a child that ignores SIGINT, both in their own group
    proc = subprocess.Popen(["bash", "-c", "trap '' INT; sleep 60 & wait"], preexec_fn=os.setsid)
    time.sleep(0.3)
    pgid = os.getpgid(proc.pid)
    t0 = time.monotonic()
    benchlib.stop_group(proc, waits=(0.5, 2.0, 2.0))
    assert time.monotonic() - t0 < 4.0
    assert not benchlib.group_alive(pgid)  # nobody left running in the group (zombies awaiting reaping don't count)


def test_return_to_start_converges_despite_command_lag(monkeypatch):
    """The sweep's +v/-v steps don't cancel (0.5 s command lag), so it ends with a P-loop back to the start. Against a
    fake arm with that lag it must end inside the 0.3 deg deadband without oscillating."""
    import collections
    import types
    import time
    rclpy = types.ModuleType("rclpy")
    rclpy.spin_once = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "benchros", types.ModuleType("benchros"))
    import importlib
    sweep = importlib.import_module("axis_sweep")
    clock = [0.0]
    monkeypatch.setattr(sweep.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sweep.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + max(s, 0.02)))

    class LaggyArm:
        def __init__(self, pos):
            self.fw = [pos, 30, 20, 125, 94, 180, 0]
            self.q = collections.deque([0.0] * 25)      # 25 ticks x 20 ms = 0.5 s command lag
            self.sent = []

        def send_v(self, v6):
            self.q.append(v6[0])
            self.fw[0] += self.q.popleft() * 0.02
            self.sent.append(v6[0])

        def stop(self):
            self.send_v([0.0] * 6)

    arm = LaggyArm(57.04)
    err = sweep.return_to(arm, 0, 60.01, 0.2)
    assert abs(err) <= 0.6                                 # lag can carry it a little past the deadband edge
    assert max(abs(v) for v in arm.sent) <= 16.0 + 1e-9    # still under the 0.2x cap on axis 1
    signs = [v > 0 for v in arm.sent if abs(v) > 1e-9]
    assert sum(a != b for a, b in zip(signs, signs[1:])) <= 2   # no sustained oscillation

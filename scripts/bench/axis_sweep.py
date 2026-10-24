#!/usr/bin/env python3
"""BENCH ONLY -- never run against the live link. Per-axis velocity-step sweep to characterize lag, max velocity
reached, and acceleration, so dev_arm_real max_vel / max_acc (today a placeholder) can be set from real numbers.

Small SYMMETRIC velocity steps (default 2, 5, 10 deg/s, a short hold each) around the current pose, through
/arm/command 'V'. Every velocity is hard-capped at 0.2x the firmware max (benchlib.clamp_vel). Prints a table.

    source /opt/ros/jazzy/setup.bash && source .../RoverFlake2/install/setup.bash
    python3 axis_sweep.py --axis 1 --i-am-at-the-bench
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib  # noqa: E402
from rover_intent.control.stepper_map import DEFAULT_TABLE  # noqa: E402

import rclpy  # noqa: E402
import benchros  # noqa: E402


def run_segment(arm, axis0, v, dur, hz=50.0):
    ts, fw = [], []
    dt, t_end = 1.0 / hz, time.monotonic() + dur
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        rclpy.spin_once(arm, timeout_sec=0.0)
        arm.send_v([v if i == axis0 else 0.0 for i in range(6)])
        if arm.fw is not None:
            ts.append(time.monotonic())
            fw.append(arm.fw[axis0])
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    return ts, fw


def return_to(arm, axis0, target, frac, kp=1.0, deadband=0.15, settle_s=1.0, timeout=20.0, hz=50.0):
    """Slow capped P-loop back to `target` (fw deg) on one axis, then zero. Needed because +v/-v velocity steps don't
    cancel: the ~0.5 s command-to-motion lag measured on axis 1 (2026-10-23) left the arm 1-3 deg off its start.
    kp is low (1/s) so kp * lag stays well under pi/2 (no oscillation). Returns the final error in deg."""
    dt, t_end, t_in = 1.0 / hz, time.monotonic() + timeout, None
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        rclpy.spin_once(arm, timeout_sec=0.0)
        e = target - arm.fw[axis0]
        if abs(e) <= deadband:
            arm.send_v([0.0] * 6)
            t_in = t_in or time.monotonic()
            if time.monotonic() - t_in >= settle_s:
                break
        else:
            t_in = None
            v = benchlib.p_step(target, arm.fw[axis0], axis0, frac, kp=kp, deadband=deadband)
            arm.send_v([v if i == axis0 else 0.0 for i in range(6)])
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    arm.stop()
    rclpy.spin_once(arm, timeout_sec=0.1)
    return target - arm.fw[axis0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", type=int, required=True, help="1-based axis to sweep")
    ap.add_argument("--steps", default="2,5,10", help="comma list of velocity magnitudes deg/s (default 2,5,10)")
    ap.add_argument("--hold", type=float, default=0.6, help="seconds per step (default 0.6 -> a few degrees)")
    ap.add_argument("--speed", type=float, default=benchlib.SPEED_CAP, help="fraction of firmware max (hard cap 0.2)")
    ap.add_argument("--i-am-at-the-bench", action="store_true")
    a = ap.parse_args()
    axis0 = a.axis - 1
    steps = [float(x) for x in a.steps.split(",")]
    frac = benchlib.cap_speed_frac(a.speed)
    plan = benchlib.sweep_plan(axis0, steps, a.hold, frac)
    cap = benchlib.vel_cap_deg_s(axis0, frac)
    plan_txt = (f"AXIS SWEEP  axis {a.axis}  steps {steps} deg/s  hold {a.hold}s  cap {cap:.1f} deg/s (0.2x fw)\n"
                f"  symmetric +/- velocity steps around the CURRENT pose; sends all-zero 'V' on exit.\n"
                f"  {len(plan)} segments, ~{sum(d for d, _ in plan):.1f} s total.")
    if not benchros.confirm(plan_txt, a.i_am_at_the_bench):
        return
    rclpy.init()
    with benchros.BenchArm() as arm:
        if arm.wait_feedback() is None:
            print("no /arm/feedback: is arm-link up?")
            return
        ok, bad = benchlib.start_pose_ok(arm.fw)
        if not ok:
            print(f"refused: start pose outside firmware travel on axes {bad} (fw={[round(x, 1) for x in arm.fw]})")
            return
        start = arm.fw[axis0]
        with benchros.Recording(f"axis_sweep_a{a.axis}"):
            rows = []
            for s in steps:
                v = benchlib.clamp_vel(s, axis0, frac)
                ts, fw = run_segment(arm, axis0, v, a.hold)       # + half (measured)
                run_segment(arm, axis0, -v, a.hold)               # - half (return)
                run_segment(arm, axis0, 0.0, 0.5)                 # settle
                est = benchlib.estimate_axis_response(ts, fw, v)
                rows.append((s, v, est))
            arm.stop()
            err = return_to(arm, axis0, start, frac)
            print(f"back at the start: axis {a.axis} {arm.fw[axis0]:.2f} deg (start {start:.2f}, error {err:+.2f})")
    print("done: recorder stopped", flush=True)
    print(f"\naxis {a.axis} response (cap {cap:.1f} deg/s):")
    print(f"{'cmd deg/s':>10} {'capped':>8} {'max_vel':>9} {'lag s':>8} {'accel deg/s^2':>14}")
    for s, v, e in rows:
        lag = f"{e['lag_s']:.3f}" if e["lag_s"] is not None else "  n/a"
        print(f"{s:>10.1f} {v:>8.1f} {e['max_vel']:>9.1f} {lag:>8} {e['accel']:>14.1f}")
    max_acc = max((e["accel"] for _, _, e in rows), default=0.0)
    print(f"\nsuggested dev_arm_real axis {a.axis}: max_acc ~ {max_acc:.1f} deg/s^2 "
          f"(= {max_acc * 3.14159 / 180:.3f} rad/s^2). Firmware max speed on this axis: "
          f"{DEFAULT_TABLE.fw_max_vel[axis0]:.0f} deg/s.")


if __name__ == "__main__":
    main()

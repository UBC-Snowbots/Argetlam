#!/usr/bin/env python3
"""BENCH ONLY -- never run against the live link. Axis 4 gear-ratio check (Rowan is recalculating it).

Moves axis 4 by +N firmware degrees slowly (capped at 0.2x firmware), asks Pranav to type the REAL angle it moved
through, computes the correction factor (benchlib.axis4_correction), then returns axis 4 to the start pose. Do this
WITH Rowan so the new ratio can go into the firmware / stepper_map.

    source /opt/ros/jazzy/setup.bash && source .../RoverFlake2/install/setup.bash
    python3 axis4_check.py --i-am-at-the-bench            # +30 fw deg by default
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

AXIS0 = 3   # axis 4


def drive_to(arm, target, frac, timeout=30.0, hz=50.0):
    dt, t_end = 1.0 / hz, time.monotonic() + timeout
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        rclpy.spin_once(arm, timeout_sec=0.0)
        v = benchlib.p_step(target, arm.fw[AXIS0], AXIS0, frac)
        arm.send_v([v if i == AXIS0 else 0.0 for i in range(6)])
        if v == 0.0:
            break
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    arm.send_v([0.0] * 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta", type=float, default=30.0, help="firmware degrees to move axis 4 (default 30)")
    ap.add_argument("--speed", type=float, default=benchlib.SPEED_CAP, help="fraction of firmware max (hard cap 0.2)")
    ap.add_argument("--i-am-at-the-bench", action="store_true")
    a = ap.parse_args()
    frac = benchlib.cap_speed_frac(a.speed)
    cap = benchlib.vel_cap_deg_s(AXIS0, frac)
    plan_txt = (f"AXIS 4 CHECK  move +{a.delta} fw deg slowly (cap {cap:.1f} deg/s, 0.2x fw), read the real angle,\n"
                f"  compute the correction, then return to start. Sends all-zero 'V' on exit.")
    if not benchros.confirm(plan_txt, a.i_am_at_the_bench):
        return
    rclpy.init()
    with benchros.BenchArm() as arm:
        if arm.wait_feedback() is None:
            print("no /arm/feedback: is arm-link up?")
            return
        start = arm.fw[AXIS0]
        target = start + a.delta
        if not (5.0 <= target <= DEFAULT_TABLE.fw_max_deg[AXIS0] - 5.0):
            print(f"refused: target {target:.1f} fw deg is outside axis-4 travel [5, {DEFAULT_TABLE.fw_max_deg[AXIS0] - 5:.0f}]")
            return
        with benchros.Recording("axis4_check"):
            print(f"axis 4 start {start:.2f} fw deg -> target {target:.2f} fw deg")
            drive_to(arm, target, frac)
            reached = arm.fw[AXIS0]
            print(f"axis 4 now {reached:.2f} fw deg (commanded +{a.delta:.1f}, firmware moved {reached - start:.2f})")
            try:
                measured = float(input("type the REAL angle axis 4 moved through, in degrees: ").strip())
            except (EOFError, ValueError, KeyboardInterrupt):
                print("no measurement entered; returning to start without a correction.")
                measured = None
            if measured:
                corr = benchlib.axis4_correction(a.delta, measured)
                print(f"\naxis 4: commanded {a.delta:.1f} fw deg, real {measured:.1f} deg -> "
                      f"ratio {corr['ratio_real_per_fw']:.4f} real/fw, correction {corr['correction']:.4f} "
                      f"(multiply fw commands by this to hit a real target). Give this to Rowan.")
            print("returning axis 4 to start...")
            drive_to(arm, start, frac)
            arm.stop()


if __name__ == "__main__":
    main()

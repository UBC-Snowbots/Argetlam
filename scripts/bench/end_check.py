#!/usr/bin/env python3
"""BENCH ONLY -- never run against the live link. End-stop / base-pose check.

Drives the arm to the dev_arm_real 'base' preset with a capped P-loop (0.2x firmware, like the follower), then asks
Pranav to compare the arm against the physical base mark and logs his answer. Every velocity is hard-capped at 0.2x;
sends all-zero 'V' on exit.

    source /opt/ros/jazzy/setup.bash && source .../RoverFlake2/install/setup.bash
    python3 end_check.py --i-am-at-the-bench
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib  # noqa: E402
from rover_intent.control.stepper_map import DEFAULT_TABLE, q_to_fw  # noqa: E402

import rclpy  # noqa: E402
import benchros  # noqa: E402

# dev_arm_real 'base' preset (= home), in URDF radians. Keep in sync with configs/arms/dev_arm_real.yaml.
BASE_Q = [0.104, -1.415, -0.997, -0.229, -0.482, -3.142]


def drive_all(arm, target_fw, frac, kp=1.0, deadband=0.15, settle_s=1.0, timeout=60.0, hz=50.0):
    """Capped P-loop on all six axes to target_fw, done once every axis has stayed inside `deadband` for `settle_s`.
    kp is low (1/s): the firmware starts moving 0.17-0.31 s after a command (bench sweep 2026-10-23), so the old
    kp 3 with an instant exit overshot; kp * lag stays well under pi/2."""
    dt, t_end, t_in = 1.0 / hz, time.monotonic() + timeout, None
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        rclpy.spin_once(arm, timeout_sec=0.0)
        v = [benchlib.p_step(target_fw[i], arm.fw[i], i, frac, kp=kp, deadband=deadband) for i in range(6)]
        arm.send_v(v)
        if all(x == 0.0 for x in v):
            t_in = t_in or time.monotonic()
            if time.monotonic() - t_in >= settle_s:
                break
        else:
            t_in = None
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    arm.send_v([0.0] * 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=benchlib.SPEED_CAP, help="fraction of firmware max (hard cap 0.2)")
    ap.add_argument("--i-am-at-the-bench", action="store_true")
    a = ap.parse_args()
    frac = benchlib.cap_speed_frac(a.speed)
    target_fw = q_to_fw(BASE_Q, DEFAULT_TABLE)
    plan_txt = ("END CHECK  drive to the 'base' preset with a capped P-loop (0.2x fw), then compare to the base mark.\n"
                f"  base target (fw deg): {[round(x, 1) for x in target_fw]}. Sends all-zero 'V' on exit.")
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
        with benchros.Recording("end_check"):
            print(f"driving to base {[round(x, 1) for x in target_fw]} ...")
            drive_all(arm, target_fw, frac)
            arm.stop()
            reached = [round(arm.fw[i], 2) for i in range(6)]
            err = [round(arm.fw[i] - target_fw[i], 2) for i in range(6)]
            print(f"\nreached (fw deg): {reached}")
            print(f"residual vs base: {err}")
            try:
                note = input("Compare the arm to the physical base mark - type what you see (logged): ").strip()
            except (EOFError, KeyboardInterrupt):
                note = ""
            print(f"\nEND CHECK LOG: target={[round(x, 1) for x in target_fw]} reached={reached} "
                  f"residual={err} note={note!r}")


if __name__ == "__main__":
    main()

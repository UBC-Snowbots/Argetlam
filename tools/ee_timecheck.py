#!/usr/bin/env python3
"""BENCH ONLY -- confirm the faster gripper (EE / firmware axis 7) still does a FULL open in the new open time and a
CLEAN stall-close in the new close time. Run it on the VM, next to the arm, with the e-stop in hand. It drives ONLY
the EE (all 6 joint velocities 0); it never moves a joint. It needs nothing but arm-link up (no sim, no app).

This mirrors tools/sim_to_real.py TimedGripper exactly: +ee_speed OPENS, -ee_speed CLOSES, open for --ee-open-s,
close for --ee-close-s (longer than full travel, so it stalls closed). Defaults match the follower's new defaults
(speed 75, open 2.0 s, close 2.8 s). It is phase-by-phase: it waits for you to press Enter before each move, so you
can watch, and any Ctrl-C (or exit) sends an all-zero 'V'.

    source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash
    python3 ~/projects/ubc-rover-arm/tools/ee_timecheck.py --i-am-at-the-bench

Sequence:
  1. PRIME CLOSE  (-speed for close_s)  -> brings the gripper to the closed stop and stalls (known state)
  2. OPEN         (+speed for open_s)   -> should be FULLY open at the end; confirm by eye
  3. CLOSE        (-speed for close_s)  -> should close and stall cleanly (watch/listen: no hard bang, no skipped
                                           steps). Feedback is OPEN-LOOP, so a stall does NOT show in the numbers.

WARNING: a faster close hits the closed stop (or any grasped object) harder than before. Keep the e-stop in hand and
do the first run with nothing in the jaws.
"""
import argparse
import atexit
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from rover_msgs.msg import ArmCommand

V_CMD = ord("V")


class EEOnly(Node):
    """Drive /arm/command with the 6 joint velocities 0 and the EE velocity in end_effector; read /arm/feedback[6].
    Sends an all-zero 'V' on exit / signal / finally (atexit + signal handlers), like scripts/bench/ee_limits.py."""

    def __init__(self):
        super().__init__("ee_timecheck")
        self.pub = self.create_publisher(ArmCommand, "/arm/command", 10)
        self.fw = None          # full firmware vector, length 7 (index 6 = EE) once a length-7 sample arrives
        self.t_fw = 0.0
        self.create_subscription(ArmCommand, "/arm/feedback", self._on, 10)
        atexit.register(self.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._sig)

    def _on(self, m):
        if len(m.positions) >= 7:
            self.fw = [float(x) for x in m.positions[:7]]
            self.t_fw = time.monotonic()

    def _sig(self, *_):
        self.stop()
        raise KeyboardInterrupt

    def ee(self):
        return self.fw[6] if self.fw is not None else None

    def send_ee(self, v):
        m = ArmCommand()
        m.cmd_type = V_CMD
        m.velocities = [0.0] * 6
        m.end_effector = float(v)
        if rclpy.ok():
            self.pub.publish(m)

    def stop(self):
        try:
            for _ in range(3):
                if rclpy.ok():
                    self.send_ee(0.0)
                    time.sleep(0.02)
        except Exception:  # noqa: BLE001
            pass

    def wait_feedback(self, timeout=3.0):
        t_end = time.monotonic() + timeout
        while self.fw is None and time.monotonic() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.fw


def drive(arm, v, duration, hz=20.0):
    """Send end_effector=v for `duration` wall-clock seconds at `hz`, then zero. Returns (ee_start, ee_end, elapsed)."""
    dt = 1.0 / hz
    ee_start = arm.ee()
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        tick = time.monotonic()
        rclpy.spin_once(arm, timeout_sec=0.0)
        arm.send_ee(v)
        time.sleep(max(0.0, dt - (time.monotonic() - tick)))
    arm.stop()
    rclpy.spin_once(arm, timeout_sec=0.0)
    return ee_start, arm.ee(), time.monotonic() - t0


def phase(arm, name, v, duration):
    print(f"\n== {name}: end_effector = {v:+.1f} fw deg/s for {duration:.2f} s ==", flush=True)
    s, e, el = drive(arm, v, duration)
    ds = (e - s) if (s is not None and e is not None) else None
    print(f"   done in {el:.2f} s | EE fw {('%.2f -> %.2f (d %.2f)' % (s, e, ds)) if ds is not None else 'n/a'} "
          "(open-loop: a stall will NOT show here -- trust your eyes/ears)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-am-at-the-bench", action="store_true",
                    help="REQUIRED. You are standing next to the arm with the e-stop. Without it this refuses to run.")
    ap.add_argument("--ee-speed", type=float, default=75.0, help="EE velocity magnitude, fw units (follower default 75)")
    ap.add_argument("--ee-open-s", type=float, default=2.0, help="open time, s (follower default 2.0)")
    ap.add_argument("--ee-close-s", type=float, default=2.8, help="close time, s (follower default 2.8)")
    a = ap.parse_args()

    speed = min(abs(a.ee_speed), 90.0)   # same cap as the follower (firmware axis-7 max_velocity)
    if not a.i_am_at_the_bench:
        print("REFUSED: this drives the REAL gripper. Re-run with --i-am-at-the-bench only while you are next to the")
        print("arm with the e-stop in hand, nothing in the jaws. It will NOT run on its own.")
        return 2
    print("EE TIME-CHECK (BENCH): drives ONLY the EE (joints held at 0).")
    print(f"  speed {speed:.1f} fw deg/s, open {a.ee_open_s:.2f} s, close {a.ee_close_s:.2f} s. +opens, -closes.")
    print("  Ctrl-C at any time -> all-zero 'V'. Keep the e-stop in hand.")

    rclpy.init()
    arm = None
    try:
        arm = EEOnly()
        if arm.wait_feedback() is None:
            print("no /arm/feedback with a length-7 positions vector (needs the EE): is arm-link up? Aborting.")
            return 1
        input("\nPress Enter to PRIME CLOSE (drive to the closed stop and stall), or Ctrl-C to abort... ")
        phase(arm, "PRIME CLOSE", -speed, a.ee_close_s)
        input("\nPress Enter to OPEN (should end FULLY open), or Ctrl-C to abort... ")
        phase(arm, "OPEN", +speed, a.ee_open_s)
        print("   >>> Confirm BY EYE: is the gripper fully open? If it fell short, raise --ee-open-s a little")
        print("       (or lower --ee-speed); if it slammed past, lower --ee-open-s.")
        input("\nPress Enter to CLOSE (should stall cleanly), or Ctrl-C to abort... ")
        phase(arm, "CLOSE", -speed, a.ee_close_s)
        print("   >>> Confirm BY EAR/EYE: did it close and stall cleanly (no hard bang, no skipped steps)?")
        print("\nDone. If both looked right, the follower's new defaults (75 / 2.0 / 2.8) are good.")
    except KeyboardInterrupt:
        print("\naborted (Ctrl-C): sending all-zero 'V'.")
    finally:
        try:
            if arm is not None and rclpy.ok():
                for _ in range(3):
                    arm.send_ee(0.0)
                    time.sleep(0.02)
        except Exception:  # noqa: BLE001
            pass
        try:
            if arm is not None:
                arm.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

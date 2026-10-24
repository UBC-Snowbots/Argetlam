#!/usr/bin/env python3
"""READ ONLY -- publishes NOTHING, so it is safe on the live arm (but still don't run it yourself; it is Pranav's).
Records N minutes of /arm/feedback with no commands and reports the per-axis drift (deg/s).

The steppers are open-loop and creep ~-0.005..-0.007 deg/s at zero command BEFORE the arm is homed; after a home,
/arm/feedback is dead still (0 deg/s). Run this to confirm which state the arm is in before a data session.

    source /opt/ros/jazzy/setup.bash && source .../RoverFlake2/install/setup.bash
    python3 drift_check.py --minutes 2
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rover_msgs.msg import ArmCommand  # noqa: E402
import benchros  # noqa: E402


class FeedbackOnly(Node):
    """Pure subscriber: no /arm/command publisher at all, so drift_check can PUBLISH NOTHING even on the live domain."""

    def __init__(self):
        super().__init__("drift_check")
        self.fw = None
        self.create_subscription(ArmCommand, "/arm/feedback", self._on, 10)

    def _on(self, m):
        if len(m.positions) >= 6:
            self.fw = ([float(x) for x in m.positions] + [0.0] * 7)[:7]

    def wait_feedback(self, timeout=3.0):
        t_end = time.monotonic() + timeout
        while self.fw is None and time.monotonic() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.fw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=2.0, help="recording length (default 2)")
    a = ap.parse_args()
    plan_txt = (f"DRIFT CHECK (READ ONLY)  record {a.minutes:.1f} min of /arm/feedback, no commands sent.\n"
                f"  reports per-axis drift deg/s. Safe on the live arm (publishes nothing).")
    try:
        input(plan_txt + "\n\nPress Enter to start, Ctrl-C to abort: ")
    except (EOFError, KeyboardInterrupt):
        print("\naborted.")
        return
    rclpy.init()
    arm = FeedbackOnly()
    try:
        if arm.wait_feedback() is None:
            print("no /arm/feedback: is arm-link up?")
            return
        with benchros.Recording(f"drift_{int(a.minutes)}min"):
            ts, rows = [], []
            t_end = time.monotonic() + a.minutes * 60.0
            t0 = time.monotonic()
            last = 0.0
            while time.monotonic() < t_end:
                rclpy.spin_once(arm, timeout_sec=0.1)
                now = time.monotonic()
                if arm.fw is not None and now - last >= 0.05:   # ~20 Hz sampling for the fit
                    last = now
                    ts.append(now - t0)
                    rows.append(list(arm.fw[:6]))
            slopes = benchlib.per_axis_drift(ts, rows)
    finally:
        arm.destroy_node()
    print(f"\ndrift over {a.minutes:.1f} min, {len(ts)} samples:")
    for i, s in enumerate(slopes):
        print(f"  axis {i + 1}: {s:+.5f} deg/s")
    worst = max((abs(s) for s in slopes), default=0.0)
    print(f"max |drift| = {worst:.5f} deg/s  "
          f"({'looks HOMED (still)' if worst < 0.002 else 'creeping -> home before a data session'})")


if __name__ == "__main__":
    main()

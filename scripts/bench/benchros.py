"""Shared ROS harness for the bench scripts: a /arm/command publisher + /arm/feedback subscriber that ALWAYS sends
all-zero 'V' on exit / Ctrl-C / exception, a confirm gate, and self-recording via record_real.sh.

Imports rclpy + rover_msgs, so it is only loaded when actually running a bench (never by the unit tests, which use
benchlib + pure-python fakes). NEVER run a bench against the live link.
"""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time

import benchlib
import rclpy
from rclpy.node import Node
from rover_msgs.msg import ArmCommand

V_CMD = ord("V")
_HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_SH = os.path.join(_HERE, "..", "record_real.sh")


class BenchArm(Node):
    """Minimal bench arm interface. Use as a context manager so the finally/exit path always zeros velocity."""

    def __init__(self):
        super().__init__("bench")
        self.pub = self.create_publisher(ArmCommand, "/arm/command", 10)
        self.fw = None
        self.t_fw = 0.0
        self.create_subscription(ArmCommand, "/arm/feedback", self._on, 10)
        self._stopped = False
        atexit.register(self.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._sig)

    def _on(self, m):
        if len(m.positions) >= 6:
            self.fw = ([float(x) for x in m.positions] + [0.0] * 7)[:7]
            self.t_fw = time.monotonic()

    def _sig(self, *_):
        self.stop()
        raise KeyboardInterrupt

    def send_v(self, v6):
        msg = ArmCommand()
        msg.cmd_type = V_CMD
        msg.velocities = [float(x) for x in v6]
        msg.end_effector = 0.0
        if rclpy.ok():
            self.pub.publish(msg)

    def stop(self):
        """Send a few all-zero velocity commands. Safe to call repeatedly (exit, signal, finally)."""
        try:
            for _ in range(3):
                if rclpy.ok():
                    self.send_v([0.0] * 6)
                    time.sleep(0.02)
        except Exception:  # noqa: BLE001
            pass
        self._stopped = True

    def wait_feedback(self, timeout=3.0):
        t_end = time.monotonic() + timeout
        while self.fw is None and time.monotonic() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.fw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def confirm(plan_text: str, i_am_at_the_bench: bool) -> bool:
    """Print the plan and require --i-am-at-the-bench plus an explicit Enter. Returns False to abort."""
    print(plan_text)
    if not i_am_at_the_bench:
        print("\nRefusing: pass --i-am-at-the-bench (someone must be at the arm with a hand on the e-stop).")
        return False
    try:
        input("\nPress Enter to run this plan, or Ctrl-C to abort: ")
    except (EOFError, KeyboardInterrupt):
        print("\naborted.")
        return False
    return True


class Recording:
    """Record the session's own bag via record_real.sh for the duration of the maneuver."""

    def __init__(self, label: str):
        self.label = label
        self.proc = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["bash", RECORD_SH, self.label], preexec_fn=os.setsid)
            time.sleep(4)  # let the recorder subscribe before the maneuver starts
            print(f"recording this session ({self.label}) via record_real.sh")
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not start recorder ({e}); continuing without a bag", file=sys.stderr)
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            benchlib.stop_group(self.proc)


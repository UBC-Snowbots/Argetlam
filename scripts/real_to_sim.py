#!/usr/bin/env python3
"""Real -> sim mirror: drive the sim arm from the PHYSICAL arm's joints.

  real joints <- Rowan's stepper_arm_driver /arm/feedback (RAW firmware degrees, index 6 = EE)
  convert     <- control.stepper_map (single source of truth)
  send        -> the Argetlam app over UDP, message kind "real_joints" {t, fw[7], q[6], table}, capped at --hz

On start it sends a MirrorMsg{on:true} so the app enters its "mirror real" state (sim joint targets come straight
from q, no IK, body/voice motion refused), and then RE-ASSERTS mirror on every ~2 s while running; on exit it sends
MirrorMsg{on:false}. The re-assert is cheap UDP and matters because the app can lose the mirror flag on an app/stack
restart: a single start-only "mirror on" then silently stops mirroring with no error (seen 2026-10-23). The app
stamps each sample with its OWN VM receive time (the one shared clock) and records it in the SessionRecorder; if
samples go stale (>0.5 s) the app holds. READ-ONLY on the arm: this never publishes to /arm/command.

    source /opt/ros/jazzy/setup.bash && source .../RoverFlake2/install/setup.bash
    python3 real_to_sim.py                      # live app on 127.0.0.1:47100, 30 Hz
    python3 real_to_sim.py --port 47195 --hz 30 # a spare app instance (tests / sim trial)
    python3 real_to_sim.py --no-mirror          # just stream real_joints, don't toggle the app's mirror state
    python3 real_to_sim.py --mirror-interval 2  # seconds between mirror-on re-asserts (default 2)

Pure stdlib + rclpy (no numpy) so it runs on the system python with ROS sourced.
"""
import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rover_intent.control.stepper_map import DEFAULT_TABLE, fw_to_q  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rover_msgs.msg import ArmCommand  # noqa: E402


class RealToSim(Node):
    def __init__(self, host, port, hz, mirror, mirror_interval=2.0):
        super().__init__("real_to_sim")
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.min_dt = 1.0 / hz
        self.mirror = mirror
        self.mirror_interval = mirror_interval
        self.fw = None
        self.last_send = 0.0
        self.create_subscription(ArmCommand, "/arm/feedback", self.on_fb, 10)

    def send_json(self, payload):
        self.sock.sendto(json.dumps(payload).encode(), self.addr)

    def send_mirror(self, on):
        self.send_json({"kind": "mirror", "t": time.time(), "on": bool(on)})

    def reassert_mirror(self):
        """Timer callback: re-send 'mirror on' while mirroring. The app can silently drop the mirror flag on an
        app/stack restart, so a single start-only message stops mirroring with no error; re-asserting self-heals it."""
        if self.mirror:
            self.send_mirror(True)

    def on_fb(self, m):
        # RAW firmware degrees, padded/truncated to 7 (index 6 = EE). Convert the 6 arm joints only.
        self.fw = ([float(x) for x in m.positions] + [0.0] * 7)[:7]
        now = time.monotonic()
        if now - self.last_send < self.min_dt:   # rate cap
            return
        self.last_send = now
        q = fw_to_q(self.fw, DEFAULT_TABLE)
        self.send_json({"kind": "real_joints", "t": time.time(), "fw": self.fw, "q": q,
                        "table": DEFAULT_TABLE.version})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47100, help="app UDP listen port (47100 live; a spare for tests)")
    ap.add_argument("--hz", type=float, default=30.0, help="max real_joints send rate")
    ap.add_argument("--no-mirror", action="store_true", help="stream real_joints but do not toggle the app mirror")
    ap.add_argument("--mirror-interval", type=float, default=2.0,
                    help="seconds between mirror-on re-asserts (cheap UDP; keeps mirror set across an app restart)")
    a = ap.parse_args()
    rclpy.init()
    node = RealToSim(a.host, a.port, a.hz, mirror=not a.no_mirror, mirror_interval=a.mirror_interval)
    if node.mirror:
        node.send_mirror(True)
        node.create_timer(a.mirror_interval, node.reassert_mirror)   # re-assert mirror on every ~interval s
        node.get_logger().info(f"mirror on -> {a.host}:{a.port} (re-asserting every {a.mirror_interval:.0f} s)")
    node.get_logger().info(f"real_to_sim: /arm/feedback -> real_joints @ {a.hz:.0f} Hz to {a.host}:{a.port}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.mirror:
            try:
                node.send_mirror(False)
            except OSError:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

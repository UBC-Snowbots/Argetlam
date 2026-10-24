"""Sim -> real follower (first test): the real dev_arm follows the Isaac sim arm's joint angles.

  sim joints  <- app Query reply "q_meas" (UDP 127.0.0.1:47100, the running rover-stack app)
  real joints <- Rowan's stepper_arm_driver /arm/feedback (firmware degrees)
  command     -> Rowan's driver /arm/command  cmd_type 'V' (velocities, firmware deg/s)

Mapping = Rowan's ArmConstants::Stepper table (RoverFlake2 arm_constants.h):
  q_urdf = radians(fw_deg) * dir + zero_rad   =>   fw_deg = degrees((q_urdf - zero_rad) * dir)
Control: v = kp * (fw_target - fw_real), deadband, capped at --speed x firmware max (80/40/60/110/90/90 deg/s).

    source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash
    python3 sim_to_real.py --dry              # print, send nothing
    python3 sim_to_real.py --axes 1           # only axis 1 follows (others held at zero velocity)
    python3 sim_to_real.py --axes 1,2,3,5,6 --speed 0.2

Safety: refuses to start if an enabled axis is > --max-start-err deg from the sim target; zero velocity if real
feedback (> 0.3 s) or the sim (> 0.5 s) goes stale, while the app reports hw_busy > 0 (the real arm is homing),
and on exit / Ctrl+C. --speed defaults to 0.2 (the dev_arm_real profile's fraction of the firmware max speed).
A copy of the log is written under ~/projects/ubc-rover-arm/logs/. Laptop bridge watchdog = backstop.
Pure Python (no numpy) so it runs on the system python with ROS sourced.
"""
import argparse
import json
import logging
import math
import os
import socket
import sys
import time

# Mapping = the single source of truth (rover_intent.control.stepper_map), imported via the repo src on sys.path so
# this stays a pure-python no-numpy script on the system python. tools/sim_to_real.py (outside git) keeps a synced
# copy of these constants inline; both must match stepper_map.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rover_intent.control.stepper_map import DEFAULT_TABLE  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rover_msgs.msg import ArmCommand  # noqa: E402

ZERO_RAD = list(DEFAULT_TABLE.zero_rad)
DIR = list(DEFAULT_TABLE.dir)
FW_MAX_DEG = list(DEFAULT_TABLE.fw_max_deg)
FW_MAX_VEL = list(DEFAULT_TABLE.fw_max_vel)
V_CMD = ord("V")
LOGDIR = os.path.expanduser("~/projects/ubc-rover-arm/logs")


def to_fw(q, i):
    return math.degrees((q - ZERO_RAD[i]) * DIR[i])


def _file_logger():
    os.makedirs(LOGDIR, exist_ok=True)
    path = os.path.join(LOGDIR, "sim_to_real_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
    lg = logging.getLogger("sim_to_real")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg.addHandler(h)
    return lg, path


class SimQuery:
    """Polls the rover-intent app for the sim arm's measured joints and the hw_busy window."""

    def __init__(self, host="127.0.0.1", port=47100):
        self.addr = (host, port)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.settimeout(0.05)
        self.hw_busy = 0.0
        self.mirror = False   # True = the app is mirroring the real arm; the follower must not command it (one-way)

    def joints(self):
        self.s.sendto(json.dumps({"kind": "query", "t": time.time()}).encode(), self.addr)
        t_end = time.monotonic() + 0.1
        while time.monotonic() < t_end:
            try:
                data, _ = self.s.recvfrom(65536)
            except socket.timeout:
                continue
            msg = json.loads(data)
            if msg.get("kind") == "state" and "q_meas" in msg:
                self.hw_busy = float(msg.get("hw_busy", 0.0) or 0.0)
                self.mirror = bool(msg.get("mirror", False))
                return [float(x) for x in msg["q_meas"][:6]]
        return None


class Follower(Node):
    def __init__(self, dry):
        super().__init__("sim_to_real")
        self.dry, self.fw, self.t_fw = dry, None, 0.0
        self.create_subscription(ArmCommand, "/arm/feedback", self.on_fb, 10)
        self.pub = self.create_publisher(ArmCommand, "/arm/command", 10)

    def on_fb(self, m):
        if len(m.positions) >= 6:
            self.fw, self.t_fw = [float(x) for x in m.positions[:6]], time.monotonic()

    def send(self, v):
        m = ArmCommand()
        m.cmd_type = V_CMD
        m.velocities = [float(x) for x in v]
        m.end_effector = 0.0
        if not self.dry:
            self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="1", help="1-based axes that follow the sim, e.g. 1 or 1,2,3,5,6")
    ap.add_argument("--speed", type=float, default=0.2, help="fraction of firmware max speed (cap 0.5); dev_arm_real")
    ap.add_argument("--kp", type=float, default=3.0, help="1/s")
    ap.add_argument("--deadband", type=float, default=0.5, help="deg")
    ap.add_argument("--margin", type=float, default=5.0, help="deg inside the firmware [0, max] range")
    ap.add_argument("--max-start-err", type=float, default=30.0, help="deg; refuse to start beyond this")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--dry", action="store_true", help="print, send nothing")
    a = ap.parse_args()
    axes = [int(x) - 1 for x in a.axes.split(",")]
    speed = min(a.speed, 0.5)
    filelog, logpath = _file_logger()
    rclpy.init()
    node, sim = Follower(a.dry), SimQuery()
    rlog = node.get_logger()

    def info(m):
        rlog.info(m)
        filelog.info(m)

    def error(m):
        rlog.error(m)
        filelog.error(m)

    info(f"sim_to_real starting (log {logpath}); axes {[i + 1 for i in axes]}, speed {speed:.0%}, dry={a.dry}")
    zero = [0.0] * 6
    try:
        t_end = time.monotonic() + 3.0
        while node.fw is None and time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
        q = sim.joints()
        if node.fw is None or q is None:
            error(f"no {'real feedback (/arm/feedback)' if node.fw is None else 'sim (app query on 47100)'}: "
                  "is arm-link up / rover-stack on?")
            return
        tgt = [min(max(to_fw(q[i], i), a.margin), FW_MAX_DEG[i] - a.margin) for i in range(6)]
        err = [tgt[i] - node.fw[i] for i in range(6)]
        info("sim fw target: " + " ".join(f"{x:7.2f}" for x in tgt))
        info("real fw now:   " + " ".join(f"{x:7.2f}" for x in node.fw))
        info("error (deg):   " + " ".join(f"{x:7.2f}" for x in err))
        bad = [i + 1 for i in axes if abs(err[i]) > a.max_start_err]
        if bad:
            error(f"axes {bad} are more than {a.max_start_err} deg from the sim: move the sim closer, "
                  "or leave them out of --axes. Not starting.")
            return
        if a.dry:
            info("dry run: nothing sent")
            return
        info(f"following on axes {[i + 1 for i in axes]} at {speed:.0%} speed. Ctrl+C to stop.")
        t_sim, last_log, last_stale, last_busy, last_mirror, dt = time.monotonic(), 0.0, 0.0, 0.0, 0.0, 1.0 / a.hz
        while True:
            t0 = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            qs = sim.joints()
            if qs is not None:
                q, t_sim = qs, time.monotonic()
            now = time.monotonic()
            if now - node.t_fw > 0.3:
                node.send(zero)
                error("real feedback stale: zero velocity, stopping")
                break
            if sim.mirror:                           # the app is mirroring the real arm: the follower must NOT drive it
                node.send(zero)                      # (otherwise real -> sim -> real would be a loop). Hold, one-way.
                if now - last_mirror > 1.0:
                    last_mirror = now
                    info("app reports mirror: holding (zero velocity) so the real->sim->real loop stays open")
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
                continue
            if sim.hw_busy > 0.0:                    # the app is homing the real arm: hold (zero velocity)
                node.send(zero)
                if now - last_busy > 1.0:
                    last_busy = now
                    info(f"app reports hw_busy {sim.hw_busy:.0f} s: holding (zero velocity)")
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
                continue
            if now - t_sim > 0.5:
                node.send(zero)                      # hold until the sim answers again
                if now - last_stale > 1.0:
                    last_stale = now
                    error(f"sim stale ({now - t_sim:.1f} s, app query on 47100 not answering): holding")
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
                continue
            v = [0.0] * 6
            for i in axes:
                t = min(max(to_fw(q[i], i), a.margin), FW_MAX_DEG[i] - a.margin)
                e = t - node.fw[i]
                if abs(e) > a.deadband:
                    cap = FW_MAX_VEL[i] * speed
                    v[i] = max(-cap, min(cap, a.kp * e))
            node.send(v)
            if now - last_log > 1.0:
                last_log = now
                info("err " + " ".join(f"{(min(max(to_fw(q[i], i), a.margin), FW_MAX_DEG[i] - a.margin) - node.fw[i]):6.1f}"
                                       for i in range(6)) + " | v " + " ".join(f"{x:5.1f}" for x in v))
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        # send the final zero velocity BEFORE tearing the context down, guarded so a torn-down context can't raise
        # the old "RCLError: publisher's context is invalid" on Ctrl+C.
        try:
            if rclpy.ok():
                for _ in range(3):
                    node.send(zero)
                    time.sleep(0.02)
                info("stopped: zero velocity sent")
        except Exception as e:  # noqa: BLE001
            error(f"could not send the final zero velocity: {e}")
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

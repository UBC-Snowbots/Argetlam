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
    python3 sim_to_real.py --axes 1,2,3,5,6 --ee   # ALSO drive the gripper (open-loop by time)

Safety: refuses to start if an enabled axis is > --max-start-err deg from the sim target; zero velocity if real
feedback (> 0.3 s) or the sim (> 0.5 s) goes stale, while the app reports hw_busy > 0 (the real arm is homing),
and on exit / Ctrl+C. --speed defaults to 0.2 (the dev_arm_real profile's fraction of the firmware max speed).
A copy of the log is written under ~/projects/ubc-rover-arm/logs/. Laptop bridge watchdog = backstop.
Pure Python (no numpy) so it runs on the system python with ROS sourced.

GRIPPER (EE, firmware axis 7): OFF unless --ee is passed. It is OPEN-LOOP BY TIME (TimedGripper): the EE count is
arbitrary and the stepper stalls safely at both stops, so close = drive closed for --ee-close-s (2.8 s, longer than
full travel: it stalls closed) and open = drive open for --ee-open-s (2 s at 75 = full open). It follows the app grip
value (0 open .. 1 closed), starts with a close, and is zeroed on stale/mirror/busy/exit like the joints.
"""
import argparse
import json
import logging
import math
import os
import socket
import sys
import time

import rclpy
from rclpy.node import Node
from rover_msgs.msg import ArmCommand

ZERO_RAD = [-0.9608, -1.9390, -1.3460, -2.4108, 2.2060 - math.pi / 3, 0.0]
DIR = [1, 1, 1, 1, -1, -1]
FW_MAX_DEG = [180.0, 180.0, 140.0, 180.0, 180.0, 9999.0]
FW_MAX_VEL = [80.0, 40.0, 60.0, 110.0, 90.0, 90.0]
V_CMD = ord("V")
LOGDIR = os.path.expanduser("~/projects/ubc-rover-arm/logs")

# EE / gripper control (optional, GATED). The EE math + limits live in the single source of truth
# rover_intent.control.ee_limits (pure stdlib). Import it from the repo if importable; if not, EE control simply
# stays OFF (end_effector = 0), so this self-contained live follower still runs when the repo is not importable.
_EE_SRC = os.path.expanduser("~/projects/ubc-rover-arm/rover-intent/src")
try:
    if _EE_SRC not in sys.path:
        sys.path.insert(0, _EE_SRC)
    from rover_intent.control import ee_limits as eel
    HAVE_EE = True
except Exception:  # noqa: BLE001
    eel = None
    HAVE_EE = False

_EE_LIMITS_DEFAULT = (str(eel.DEFAULT_LIMITS_PATH) if HAVE_EE
                      else os.path.expanduser("~/projects/ubc-rover-arm/rover-intent/configs/arms/dev_arm_real_ee.yaml"))


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
        self.grip = 0.0       # app gripper command, 0 open .. 1 closed (used only when EE control is enabled)

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
                self.grip = float(msg.get("gripper", 0.0) or 0.0)
                return [float(x) for x in msg["q_meas"][:6]]
        return None


class FeedbackGuard:
    """Real-feedback staleness state machine (2026-10-26, Pranav: pause/resume instead of exiting on a short gap).
    update(now, t_last_feedback) ->
      "ok"       feedback fresh, follow normally
      "paused"   gap just started (> stale_s): the caller sends zero velocity (returned once per pause)
      "holding"  still in a gap or waiting for feedback to stay fresh for resume_s: keep sending zero
      "resumed"  feedback has been fresh for resume_s: follow again (returned once)
      "stop"     the gap has lasted > stop_s: a real link loss, exit as before"""

    def __init__(self, stale_s=0.3, resume_s=1.0, stop_s=3.0):
        self.stale_s, self.resume_s, self.stop_s = float(stale_s), float(resume_s), float(stop_s)
        self.paused, self.gap_start, self.fresh_since, self.pauses = False, None, None, 0

    def update(self, now, t_fb):
        stale = now - t_fb > self.stale_s
        if not self.paused:
            if not stale:
                return "ok"
            self.paused, self.gap_start, self.fresh_since = True, now, None
            self.pauses += 1
            return "paused"
        if stale:
            self.fresh_since = None
            gap_from = self.gap_start if self.gap_start is not None else now
            return "stop" if now - gap_from > self.stop_s else "holding"
        self.fresh_since = self.fresh_since or now
        if now - self.fresh_since >= self.resume_s:
            self.paused, self.gap_start, self.fresh_since = False, None, None
            return "resumed"
        return "holding"


class TimedGripper:
    """Open-loop gripper (EE, axis 7) driven BY TIME, not position: the firmware's EE count is arbitrary and the
    stepper stalls safely at both stops (Pranav, 2026-10-25). close = -speed for close_s (longer than full travel, so it
    stalls closed); open = +speed for open_s (2 s at 75 = full open). Follows the app's grip value (0 open .. 1 closed):
    a flip starts the matching move at once, even mid-move. Starts with a close so the state is known. A hold (mirror,
    homing, sim stale) interrupts the move; it restarts in full afterwards (extra closing/opening only stalls).

    Speed bumped 50 -> 75 fw deg/s (2026-10-28, "a bit faster", 1.5x). Travel is assumed linear in speed, so the move
    times scale as 50/speed to keep the SAME travel: open 3.0 -> 2.0 s (full open, not more), close 4.0 -> 2.8 s (a bit
    more than 4.0*50/75 = 2.67 s, so it still overshoots full travel and stalls closed). The cap is the firmware
    axis-7 max_velocity (RoverFlake2 arm_firmware_teensy41 main.cpp: axes[6].max_velocity = 90.0, which the firmware
    clamps target_velocity to); the driver passes end_effector through unclamped, so this cap is the only software one."""

    def __init__(self, speed=75.0, close_s=2.8, open_s=2.0, hold_speed=30.0):
        self.speed = min(abs(float(speed)), 90.0)   # firmware axis-7 max_velocity (fw deg/s); above this fw clamps
        self.close_s, self.open_s = float(close_s), float(open_s)
        # While CLOSED keep driving closed at hold_speed (2026-10-28, Pranav: "make it continuously close", objects
        # slipped): a stepper at zero velocity holds its position but stops pushing, so the grip went soft once the
        # close move ended. Stalling against the object keeps the squeeze (safe for this stepper, Pranav 2026-10-25);
        # a low speed gives more stall torque and less noise. 0 = old behaviour (stop after close_s).
        self.hold_speed = min(abs(float(hold_speed)), self.speed)
        self.closed = None          # state we are driving to / reached (None = unknown -> close first)
        self.until = 0.0            # end time of the current move (monotonic)
        self.direction = 0          # -1 closing, +1 opening, 0 idle
        self.interrupted = False
        self.priming = True         # the start-up close runs to completion before the sim grip is followed

    @property
    def state(self):
        word = ("closed+squeeze" if self.hold_speed else "closed") if self.closed else "open"
        return f"{'closing' if self.direction < 0 else 'opening'}" if self.direction else word

    def _start(self, now, close):
        self.closed = close
        self.direction = -1 if close else 1
        self.until = now + (self.close_s if close else self.open_s)
        self.interrupted = False

    def interrupt(self):
        if self.direction:
            self.interrupted = True

    def update(self, now, grip):
        if self.priming:            # start-up close: known state first, then follow the sim
            if self.closed is None or self.interrupted:
                self._start(now, True)
            if now >= self.until:
                self.direction, self.priming = 0, False
            return self.direction * self.speed
        want_closed = grip > 0.5
        if want_closed != self.closed or self.interrupted:
            self._start(now, want_closed)
        if self.direction and now >= self.until:
            self.direction = 0
        if self.direction == 0 and self.closed:
            return -self.hold_speed          # keep squeezing until told to open
        return self.direction * self.speed


class Follower(Node):
    def __init__(self, dry):
        super().__init__("sim_to_real")
        self.dry, self.fw, self.t_fw = dry, None, 0.0
        self.n_fb = 0
        self.ee_fw = None        # EE firmware deg (/arm/feedback positions[6]); None until a length-7 sample arrives
        self.create_subscription(ArmCommand, "/arm/feedback", self.on_fb, 10)
        self.pub = self.create_publisher(ArmCommand, "/arm/command", 10)

    def on_fb(self, m):
        if len(m.positions) >= 6:
            self.fw, self.t_fw = [float(x) for x in m.positions[:6]], time.monotonic()
            self.n_fb += 1
            self.ee_fw = float(m.positions[6]) if len(m.positions) >= 7 else None

    def send(self, v, ee=0.0):
        m = ArmCommand()
        m.cmd_type = V_CMD
        m.velocities = [float(x) for x in v]
        m.end_effector = float(ee)        # EE velocity (fw deg/s); 0 unless EE control is enabled
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
    ap.add_argument("--resume-s", type=float, default=1.0,
                    help="after a short feedback gap, resume once feedback has been steady this long")
    ap.add_argument("--stale-stop-s", type=float, default=3.0,
                    help="exit (zero velocity) if real feedback stays stale longer than this (a real link loss)")
    ap.add_argument("--dry", action="store_true", help="print, send nothing")
    ap.add_argument("--ee", action="store_true", help="ALSO drive the gripper (EE) from the app grip value, "
                    "OPEN-LOOP by time (the EE count is arbitrary; the stepper stalls safely at the stops)")
    ap.add_argument("--ee-speed", type=float, default=75.0,
                    help="EE velocity magnitude, fw units (default 75, cap 90 = firmware axis-7 max_velocity)")
    ap.add_argument("--ee-close-s", type=float, default=2.8,
                    help="close = drive closed this long (longer than full travel, so it stalls closed)")
    ap.add_argument("--ee-hold-speed", type=float, default=30.0,
                    help="while closed keep driving closed at this speed so the grip keeps squeezing (0 = stop)")
    ap.add_argument("--ee-open-s", type=float, default=2.0, help="open = drive open this long (2 s at 75 = full open)")
    a = ap.parse_args()
    axes = [int(x) - 1 for x in a.axes.split(",")]
    speed = min(a.speed, 0.5)
    ee_on = bool(a.ee)
    gripper = TimedGripper(a.ee_speed, a.ee_close_s, a.ee_open_s, a.ee_hold_speed) if ee_on else None
    filelog, logpath = _file_logger()
    # rclpy's own SIGINT handler shuts the context down BEFORE our finally block runs, so the final zero velocity
    # failed ("publisher's context is invalid", first real teleop run 2026-10-23). Keep the context alive and let
    # Ctrl+C / SIGTERM arrive as KeyboardInterrupt; finally sends the zero, then shuts rclpy down.
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    import signal as _signal
    _signal.signal(_signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    node, sim = Follower(a.dry), SimQuery()
    # /arm/feedback is received on its OWN thread: spinning once per 20 Hz control tick (which also blocks up
    # to 0.1 s on the app query) let the 100 Hz feedback callback starve for > 0.3 s right after start
    # (spurious 'real feedback stale' stops in 3 of 5 starts, 2026-10-23).
    import threading
    from rclpy.executors import SingleThreadedExecutor
    _exec = SingleThreadedExecutor()
    _exec.add_node(node)
    threading.Thread(target=_exec.spin, daemon=True).start()
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
            time.sleep(0.05)
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
        if ee_on:
            info(f"EE control ON (open-loop by time): close = -{gripper.speed:.0f} for {gripper.close_s:.1f} s "
                 f"(stalls closed), then keeps squeezing at -{gripper.hold_speed:.0f} while closed, open = +{gripper.speed:.0f} for "
                 f"{gripper.open_s:.1f} s. Starting with a close.")
        else:
            info("EE control OFF (end_effector stays 0); pass --ee to enable")
        if a.dry:
            info("dry run: nothing sent")
            return
        info(f"following on axes {[i + 1 for i in axes]} at {speed:.0%} speed. Ctrl+C to stop.")
        t_sim, last_log, last_stale, last_busy, last_mirror, dt = time.monotonic(), 0.0, 0.0, 0.0, 0.0, 1.0 / a.hz
        stale = FeedbackGuard(stale_s=0.3, resume_s=a.resume_s, stop_s=a.stale_stop_s)
        while True:
            t0 = time.monotonic()
            qs = sim.joints()
            if qs is not None:
                q, t_sim = qs, time.monotonic()
            now = time.monotonic()
            fb = stale.update(now, node.t_fw)
            if fb == "stop":
                node.send(zero)
                error(f"real feedback stale for > {stale.stop_s:.0f} s: zero velocity, stopping")
                break
            if fb != "ok":                           # short gap: hold at zero velocity, resume when steady again
                node.send(zero)
                if ee_on:
                    gripper.interrupt()
                if fb == "paused":
                    error(f"real feedback stale ({now - node.t_fw:.2f} s): holding at zero velocity "
                          f"(resumes after {stale.resume_s:.0f} s of steady feedback, stops after {stale.stop_s:.0f} s)")
                elif fb == "resumed":
                    info(f"real feedback steady again: resuming (pause #{stale.pauses})")
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
                continue
            if sim.mirror:                           # the app is mirroring the real arm: the follower must NOT drive it
                node.send(zero)                      # (otherwise real -> sim -> real would be a loop). Hold, one-way.
                if ee_on:
                    gripper.interrupt()
                if now - last_mirror > 1.0:
                    last_mirror = now
                    info("app reports mirror: holding (zero velocity) so the real->sim->real loop stays open")
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
                continue
            if sim.hw_busy > 0.0:                    # the app is homing the real arm: hold (zero velocity)
                node.send(zero)
                if ee_on:
                    gripper.interrupt()
                if now - last_busy > 1.0:
                    last_busy = now
                    info(f"app reports hw_busy {sim.hw_busy:.0f} s: holding (zero velocity)")
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
                continue
            if now - t_sim > 0.5:
                node.send(zero)                      # hold until the sim answers again
                if ee_on:
                    gripper.interrupt()
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
            ee_v = gripper.update(now, sim.grip) if ee_on else 0.0
            node.send(v, ee_v)
            if now - last_log > 1.0:
                last_log = now
                msg_log = ("err " + " ".join(f"{(min(max(to_fw(q[i], i), a.margin), FW_MAX_DEG[i] - a.margin) - node.fw[i]):6.1f}"
                                             for i in range(6)) + " | v " + " ".join(f"{x:5.1f}" for x in v))
                if ee_on:
                    msg_log += f" | grip {sim.grip:.2f} ee {gripper.state} eev {ee_v:5.1f}"
                info(msg_log)
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
            _exec.shutdown(timeout_sec=1.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

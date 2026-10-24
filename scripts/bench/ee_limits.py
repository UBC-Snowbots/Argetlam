#!/usr/bin/env python3
"""BENCH ONLY -- never run against anything but the arm you are standing next to. Find and save the real gripper
(EE / firmware axis 7, 'ee_prismatic') limits by step-jogging it slowly, so the sim_to_real follower can drive the
gripper safely.

The EE is a lead-screw LINEAR axis with NO limit switch; home-all skips it, so its zero is ARBITRARY after every
power-on and the limits saved here are RELATIVE TO THIS POWER-ON. Re-mark (or --re-reference) each session. One end
has a mechanical hard stop; the other can squeeze/stall. BOTH ends are dangerous until marked. Feedback is OPEN-LOOP
(commanded step counts), so a mechanical STALL will NOT show in the numbers -- watch and listen, and stop with any
non-move key. Runs in a real terminal on the VM (Moonlight, or `ssh -t rover-vm`).

    source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash
    python3 scripts/bench/ee_limits.py --i-am-at-the-bench              # mark open + closed, save
    python3 scripts/bench/ee_limits.py --i-am-at-the-bench --re-reference   # later session: shift open by eye

Keys:  o = open one step   c = close one step   (hold a key = OS auto-repeat = repeated steps)
       [ = halve step      ] = double step
       m = mark here        s = save        q / Ctrl-C = stop and quit
       any other key        = STOP (zero velocity); the arm stops.

Drives ONLY the EE: every tick sends a full 'V' command with the 6 joint velocities all 0 and the EE velocity in
end_effector. On arrival / no key / exit / any exception it sends an all-zero 'V'.
"""
import argparse
import atexit
import os
import select
import signal
import sys
import termios
import time
import tty

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rover_intent.control import ee_limits as eel  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rover_msgs.msg import ArmCommand  # noqa: E402
import benchros  # noqa: E402

V_CMD = ord("V")
STEP_MIN, STEP_MAX = 0.25, 32.0          # fw deg, bounds on the jog step
STALL_WINDOW_S = 0.3                     # commanded-but-not-advancing time that trips the (best-effort) stall note


class EEBenchArm(Node):
    """Minimal EE-only bench interface: drive /arm/command with joints 0 + an EE velocity, read /arm/feedback[6].
    ALWAYS sends an all-zero 'V' on exit / signal / finally (atexit + signal handlers)."""

    def __init__(self):
        super().__init__("ee_limits_bench")
        self.pub = self.create_publisher(ArmCommand, "/arm/command", 10)
        self.fw = None            # full firmware vector, length 7 (index 6 = EE)
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

    def send_ee(self, ee_vel):
        m = ArmCommand()
        m.cmd_type = V_CMD
        m.velocities = [0.0] * 6
        m.end_effector = float(ee_vel)
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


class KeyReader:
    """Read single keystrokes from a real TTY without Enter (cbreak), non-blocking via select. Restores the TTY
    on exit. Use get() for one char (or None), and prompt() to ask a line in cooked mode."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.saved = None

    def __enter__(self):
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False

    def get(self, timeout=0.0):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if r:
            ch = os.read(self.fd, 1)
            return ch.decode(errors="replace") if ch else None
        return None

    def prompt(self, text):
        """Ask a full line in normal (cooked) mode, then return to cbreak."""
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        try:
            return input(text)
        finally:
            tty.setcbreak(self.fd)


def _drive_step(arm, keys, target, ee_speed, deadband, hz=50.0):
    """Drive the EE toward `target` fw deg at <= ee_speed until within deadband OR a key arrives OR it stalls.
    Returns (reached, stalled, key_or_None). Sends zero when it stops."""
    dt = 1.0 / hz
    t_start = time.monotonic()
    fw_start = arm.ee()
    last_progress_fw, last_progress_t = fw_start, t_start
    stalled = False
    while True:
        t0 = time.monotonic()
        rclpy.spin_once(arm, timeout_sec=0.0)
        cur = arm.ee()
        if cur is None:
            arm.send_ee(0.0)
            return False, False, None
        e = target - cur
        if abs(e) <= deadband:
            arm.send_ee(0.0)
            return True, stalled, None
        v = max(-ee_speed, min(ee_speed, (3.0 * e)))     # gentle P toward the step target, capped at ee_speed
        arm.send_ee(v)
        # best-effort stall note: feedback is open-loop (commanded steps), so this rarely fires -- rely on eyes/ears
        if abs(cur - last_progress_fw) > 0.05:
            last_progress_fw, last_progress_t = cur, t0
        elif t0 - last_progress_t > STALL_WINDOW_S:
            stalled = True
        k = keys.get(timeout=0.0)
        if k is not None:
            arm.send_ee(0.0)
            return False, stalled, k
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))


def _calibrate_open_sign(arm, keys, step, ee_speed, deadband):
    """First-tap calibration: tap o once, drive one step in +fw, ask which way the fingers moved -> open_sign."""
    print("\nCALIBRATE OPEN DIRECTION")
    print("  Tap 'o' ONCE. The EE will nudge one step in the +fw direction; watch the fingers.")
    while True:
        k = keys.get(timeout=0.2)
        if k is None:
            continue
        if k in ("q", "\x03"):
            return None
        if k == "o":
            break
        print("  (tap 'o' to nudge, or q to quit)")
    cur = arm.ee()
    reached, stalled, _ = _drive_step(arm, keys, cur + step, ee_speed, deadband)
    if stalled:
        print("  STALLED during the nudge: likely already at a stop. Back off before continuing.")
    ans = keys.prompt("  Did the fingers OPEN or CLOSE with that +fw nudge? [o=open / c=close]: ").strip().lower()
    open_sign = 1 if ans.startswith("o") else -1
    print(f"  recorded: +fw = {'OPEN' if open_sign == 1 else 'CLOSE'}  ->  open_sign = {open_sign:+d}")
    return open_sign


def run_mark(arm, keys, args):
    """Mark mode: jog with o/c, mark open & closed, save."""
    step = args.step
    open_sign = _calibrate_open_sign(arm, keys, step, args.ee_speed, args.deadband)
    if open_sign is None:
        print("quit before calibration.")
        return
    open_fw = closed_fw = None
    pending_key = None
    print("\nJOG:  o=open  c=close  [=halve step  ]=double step  m=mark  s=save  q=quit")
    while True:
        if pending_key is not None:
            k, pending_key = pending_key, None    # a key captured mid-step: process it now
        else:
            print(f"  ee={arm.ee():.2f} fw deg | step={step:.2f} | open={open_fw} closed={closed_fw}", flush=True)
            k = keys.get(timeout=0.25)
        if k is None:
            arm.send_ee(0.0)                      # watchdog: no key, step finished -> hold
            continue
        if k in ("q", "\x03"):
            break
        elif k == "[":
            step = max(STEP_MIN, step / 2)
        elif k == "]":
            step = min(STEP_MAX, step * 2)
        elif k in ("o", "c"):
            sign = open_sign if k == "o" else -open_sign
            target = arm.ee() + sign * step
            reached, stalled, nextk = _drive_step(arm, keys, target, args.ee_speed, args.deadband)
            if stalled:
                print("  STALLED: likely at a stop, back off (note: open-loop feedback may not show it).")
            if nextk is not None:                 # a key pressed mid-step (motion already stopped) -> process next
                pending_key = nextk
        elif k == "m":
            arm.send_ee(0.0)
            ans = keys.prompt("  MARK: is this the OPEN stop or the CLOSED end? [o/c]: ").strip().lower()
            if ans.startswith("o"):
                open_fw = arm.ee()
                print(f"  open marked at {open_fw:.2f} fw deg")
            elif ans.startswith("c"):
                closed_fw = arm.ee()
                print(f"  closed marked at {closed_fw:.2f} fw deg")
            else:
                print("  not marked (answer o or c).")
        elif k == "s":
            if open_fw is None or closed_fw is None:
                print("  cannot save: mark BOTH open and closed first.")
                continue
            lim = eel.EELimits(open_fw=open_fw, closed_fw=closed_fw, open_sign=open_sign,
                               margin_fw=args.margin, marked_at=eel.now_stamp())
            path = eel.save(lim, args.limits)
            lo, hi = lim.soft_limits()
            print(f"  saved {path}\n  soft limits (fw deg the follower may command): [{lo:.2f}, {hi:.2f}]")
        else:
            arm.send_ee(0.0)
            print(f"  unexpected key {k!r} -> STOP")
    arm.stop()


def run_reref(arm, keys, args):
    """Re-reference mode: jog to the OPEN position by eye, press m -> shift the saved open/closed to match (span kept)."""
    if not eel.exists(args.limits):
        print(f"no limits file at {args.limits}: run without --re-reference first (mark open + closed).")
        return
    lim = eel.load(args.limits)
    step = args.step
    print(f"\nRE-REFERENCE  loaded open={lim.open_fw:.2f} closed={lim.closed_fw:.2f} sign={lim.open_sign:+d} "
          f"span={lim.span:.2f}")
    print("  Jog to the OPEN position by eye (o/c). A gentle touch of the hard stop ONLY if Rowan OKs stalling it.")
    print("  Keys: o=open  c=close  [=halve  ]=double  m=mark-open-here  s=save  q=quit")
    pending = None
    pending_key = None
    while True:
        if pending_key is not None:
            k, pending_key = pending_key, None
        else:
            print(f"  ee={arm.ee():.2f} fw deg | step={step:.2f}", flush=True)
            k = keys.get(timeout=0.25)
        if k is None:
            arm.send_ee(0.0)
            continue
        if k in ("q", "\x03"):
            break
        elif k == "[":
            step = max(STEP_MIN, step / 2)
        elif k == "]":
            step = min(STEP_MAX, step * 2)
        elif k in ("o", "c"):
            sign = lim.open_sign if k == "o" else -lim.open_sign
            _, _, nextk = _drive_step(arm, keys, arm.ee() + sign * step, args.ee_speed, args.deadband)
            if nextk is not None:
                pending_key = nextk
        elif k == "m":
            arm.send_ee(0.0)
            pending = lim.shifted_to_open(arm.ee(), marked_at=eel.now_stamp())
            print(f"  open re-referenced to {pending.open_fw:.2f} -> closed {pending.closed_fw:.2f} "
                  f"(span kept {pending.span:.2f}). Press s to save.")
        elif k == "s":
            if pending is None:
                print("  mark the open position with m first.")
                continue
            path = eel.save(pending, args.limits)
            lo, hi = pending.soft_limits()
            print(f"  saved {path}\n  soft limits (fw deg): [{lo:.2f}, {hi:.2f}]")
            lim = pending
        else:
            arm.send_ee(0.0)
            print(f"  unexpected key {k!r} -> STOP")
    arm.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-am-at-the-bench", action="store_true")
    ap.add_argument("--re-reference", action="store_true", help="shift a saved open/closed to a new open by eye")
    ap.add_argument("--step", type=float, default=2.0, help="jog step, fw deg (default 2)")
    ap.add_argument("--ee-speed", type=float, default=eel.EE_DEFAULT_SPEED_DEG_S,
                    help=f"EE jog speed, mis-scaled fw deg/s (default {eel.EE_DEFAULT_SPEED_DEG_S}, hard cap "
                         f"{eel.EE_SPEED_CAP_DEG_S})")
    ap.add_argument("--deadband", type=float, default=1.0, help="fw deg; a step ends within this of its target")
    ap.add_argument("--margin", type=float, default=eel.EE_DEFAULT_MARGIN_FW, help="fw deg kept clear of each stop")
    ap.add_argument("--limits", default=str(eel.DEFAULT_LIMITS_PATH), help="where to read/write the limits file")
    ap.add_argument("--no-record", action="store_true", help="do not wrap the session in record_real.sh")
    a = ap.parse_args()
    a.ee_speed = eel.cap_ee_speed(a.ee_speed)      # hard cap, cannot be argued past
    mode = "RE-REFERENCE (shift open by eye, keep span)" if a.re_reference else "MARK (find open + closed)"
    plan = (f"EE LIMITS  {mode}\n"
            f"  Drives ONLY the EE (gripper), joints held at 0, at <= {a.ee_speed:.1f} fw deg/s (mis-scaled unit).\n"
            f"  step {a.step} fw deg, margin {a.margin} fw deg, limits file {a.limits}.\n"
            "  Feedback is OPEN-LOOP: a mechanical stall will NOT show -- watch/listen and stop with any other key.\n"
            "  Sends all-zero 'V' on arrival / no key / exit.")
    if not benchros.confirm(plan, a.i_am_at_the_bench):
        return
    rclpy.init()
    rec = None
    arm = None
    try:
        arm = EEBenchArm()
        if arm.wait_feedback() is None:
            print("no /arm/feedback (need length-7 positions incl. the EE): is arm-link up?")
            return
        if not a.no_record:
            rec = benchros.Recording("ee_limits")
            rec.__enter__()
        if not sys.stdin.isatty():
            print("this tool needs a real terminal (TTY) for the single-key jog. Use Moonlight or `ssh -t`.")
            return
        with KeyReader() as keys:
            if a.re_reference:
                run_reref(arm, keys, a)
            else:
                run_mark(arm, keys, a)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok() and arm is not None:
                for _ in range(3):
                    arm.send_ee(0.0)
                    time.sleep(0.02)
        except Exception:  # noqa: BLE001
            pass
        if rec is not None:
            rec.__exit__(None, None, None)
        try:
            if arm is not None:
                arm.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

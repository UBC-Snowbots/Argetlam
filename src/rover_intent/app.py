"""Main control loop (runs on the VM, or wherever the arm backend is).

  laptop (body pose, transcripts) --UDP--> Receiver
  transcripts -> Nemotron -> Intent -> Arbiter
  EEG decoder p_grasp -> Arbiter;  Muse 2 double jaw clench (eeg/clench.py) -> GripToggle -> Arbiter
  Arbiter -> TCP target -> IK step -> SafetyLayer -> backend

Run:  python -m rover_intent.app --config configs/default.yaml [--backend mock|isaac] [--seconds N]
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from .body.arm_mirror import ArmMirror
from .body.plane import PlaneMapper, DropWatch, PlaneTuneCheck, plane_coords
from .control.arbiter import Arbiter, Mode
from .control.kinematics import ArmModel
from .control.safety import SafetyLayer
from .planner.nemotron import IntentParser, urgent_stop
from .planner.scene import Scene
from .robot.backends import IsaacArm, MockArm, RealArm
from .transport.udp import Receiver
from .planner.nemotron import fallback_parse, is_confirm
from .planner.fast_path import (CART, DEFAULT_ANGLE, DEFAULT_DIST_M, J_NAMES, J_SIGN, J_WORDS, make_cart_step,
                                make_joint_step, make_preset_step, make_speed_step, parse_fast, scale_speed)
from .body.clutch import Clutch
from .config import load_config
from .control.presets import HOME_HW, RETURN, at_goal, check_joint_goal, plan_preset, preset_names, request_hw_home
from .session import SessionRecorder
from .types import (Action, ArmPreset, BodyPose, CameraFrame, ClutchMsg, EEGState, GripToggle, Intent, MirrorMsg,
                    PlaneTune, Query, RealJoints, Utterance)

log = logging.getLogger("rover_intent")
ROOT = Path(__file__).resolve().parents[2]
WINDUP = 0.25  # rad: max gap between the commanded and the measured joint positions
DROP_REASONS = ("tracking lost",)  # clutch releases that count as "the arm dropped" for arm.on_drop


def build(cfg: dict, backend: str):
    a = cfg["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), base=a.get("base_link", "base_link"), tip=a["tip_link"],
                   tcp_offset=a["tcp_offset"], tool_axis=a.get("tool_axis"))
    if a.get("joint_limits_lower") is not None and a.get("joint_limits_upper") is not None:
        # profile override (e.g. dev_arm_real): real firmware + URDF joint travel, tighter than the URDF alone (issue 3)
        arm.lower = np.array(a["joint_limits_lower"], float)
        arm.upper = np.array(a["joint_limits_upper"], float)
    safety = SafetyLayer(arm.lower, arm.upper, a["max_vel"], a["max_acc"],
                         cfg["safety"]["workspace_min"], cfg["safety"]["workspace_max"],
                         cfg["safety"]["watchdog_s"])
    home = np.array(a["home"], float)
    if backend == "mock":
        robot = MockArm(home)
    elif backend == "isaac":
        n = cfg["net"]
        robot = IsaacArm(n["isaac_host"], n["isaac_cmd_port"], n["isaac_state_port"], home)
    else:
        robot = RealArm()
    scene = Scene(cfg["scene"], support_z=cfg.get("table", {}).get("top", 0.0))
    e, s = cfg["eeg"], cfg["assist"]
    arbiter = Arbiter(scene, e["p_on"], e["p_off"], e["hold_s"], s["radius"], s["gain"])
    arbiter.z_max = cfg["safety"]["workspace_max"][2] - 0.005
    b = cfg["body"]
    # Two body-teleop modes: PLANE (default everywhere) and ARM_MIRROR (fallback). The legacy position / calibrated /
    # gesture modes were removed. An unknown mode falls back to plane.
    if b.get("mode", "plane") == "arm_mirror":   # full-arm joint mirroring, no IK (body/arm_mirror.py)
        retarget = ArmMirror(arm, home, a["max_vel"], cfg.get("arm_mirror"))
    else:                                        # plane: image-plane TCP teleop, depth ignored, IK path (body/plane.py)
        retarget = PlaneMapper(arm, home, cfg.get("plane"), cfg["safety"]["workspace_min"],
                               cfg["safety"]["workspace_max"])
    n = cfg["nemotron"]
    parser = IntentParser(n["model"], n["base_url"], n["api_key_env"], timeout_s=float(n.get("timeout_s", 6.0)))
    return arm, safety, robot, scene, arbiter, retarget, parser


def run(cfg: dict, backend: str, seconds: float | None = None, record: bool = True):
    arm, safety, robot, scene, arbiter, retarget, parser = build(cfg, backend)
    rx = Receiver(cfg["net"]["listen_port"])
    recorder = SessionRecorder(ROOT / "logs" / "sessions", enabled=record)
    dt = 1.0 / cfg["loop"]["hz"]
    log.info("backend=%s nemotron=%s objects=%s", backend, "online" if parser.online else "OFFLINE fallback",
             scene.names())
    descriptions = {n: o.get("desc", "") for n, o in cfg["scene"].items()}
    home = np.array(cfg["arm"]["home"], float)
    grip_from_hand = cfg["body"].get("grip_from_hand", False)
    teleop_orient = cfg["arm"].get("teleop_orient", "full")  # "axis" frees the wrist roll while teleoperating (issue 2)
    arm_mirror_on = isinstance(retarget, ArmMirror)   # full-arm joint mirroring: joints straight in, no IK
    plane_on = isinstance(retarget, PlaneMapper)      # image-plane TCP teleop (IK path); left-hand clutch disabled
    plain_gripper = arm_mirror_on or plane_on         # teleop modes: plain open/close gripper, no AUTO grab-assist
    plane_drop = DropWatch(float((cfg.get("plane") or {}).get("drop_below_hip_s", 0.7)))
    # Live tuning of plane teleop: a saved override (data/profiles/plane_tune_<arm>.json, gitignored) loaded at start,
    # and a guided reach-threshold "tune check". Both no-ops unless plane mode is active.
    tune_check = PlaneTuneCheck()
    tune_path = ROOT / f"data/profiles/plane_tune_{cfg['arm_profile']}.json"
    body_target, body_t = None, 0.0
    body_joint_target = None   # arm_mirror: the 6-vector joint target (instead of body_target's TCP)
    q_cmd = robot.read_joints()  # commanded trajectory state (the drives track it; see WINDUP)
    T_des = arm.fk(q_cmd)
    last_grip = arbiter.gripper  # log every gripper open/close (issue 5): hand, EEG, clench, skills and resets
    pool, pending = ThreadPoolExecutor(max_workers=1), []
    c = cfg["body"].get("clutch", {})
    clutch = Clutch(calib_s=c.get("calib_s", 2.0), still_m=c.get("still_m", 0.03), palm_s=c.get("palm_s", 1.0),
                    raise_m=c.get("raise_m", 0.05))
    hud = (cfg["net"].get("hud_host", "127.0.0.1"), cfg["net"].get("hud_port", 47130))
    hud_video = (cfg["net"].get("hud_host", "127.0.0.1"), cfg["net"].get("hud_video_port", 47131))
    last_body, last_target, last_body_arrival, last_visible_t = None, None, 0.0, 0.0
    set_mirror = None
    presets = cfg["arm"].get("presets", {}) or {}
    on_drop = cfg["arm"].get("on_drop")  # preset to go to when the body drops out of view (None = freeze)
    if on_drop and on_drop not in presets:
        raise ValueError(f"arm.on_drop {on_drop!r} is not one of the arm's presets {list(presets)}")
    # which clutch releases send the arm to the on_drop preset. arm_mirror adds "disable" (F / left hand lowered),
    # plane adds "right arm dropped" (control wrist held below the hip) + "disable" (F off): Pranav "when I drop my
    # arm return it to base". Voice stop/STOP release with reason None -> freeze (unchanged).
    drop_reasons = DROP_REASONS + (("left hand lowered", "disabled") if arm_mirror_on and on_drop else ()) \
        + (("right arm dropped", "disabled") if plane_on and on_drop else ())
    real_arm = cfg.get("real_arm", {}) or {}
    joint_goal, goal_name, hw_busy_until = None, None, 0.0  # button-panel preset move / real-arm homing window
    goal_start = None       # q_cmd captured when the current preset/jog move began (drives the move-progress readout)
    # "follower seen" heuristic for the real_following light: scripts/sim_to_real.py polls the app with Query several
    # times a second, so a Query within the last second means the follower is live (it can't be known directly).
    last_query_t, follower_ever = 0.0, False
    BUSY_MSG = lambda: f"busy: {goal_name} still moving, press STOP first"   # the one-move-at-a-time refusal text
    is_busy = lambda why: isinstance(why, str) and why.startswith("busy:")   # tag a refusal reply as busy for the UI
    # "mirror real" state: the sim arm's joint targets come straight from the PHYSICAL arm (real_to_sim -> real_joints),
    # no IK and no body/voice motion. Entered/left by a MirrorMsg (or STOP leaves it). real_t is the VM receive time of
    # the last real sample (the one shared clock); if it goes stale (> 0.5 s) the sim holds.
    mirror_real, real_q, real_fw, real_t = False, None, None, 0.0
    MIRROR_STALE_S = 0.5
    conf = cfg.get("confirm", {}) or {}
    confirm_window_s = float(conf.get("window_s", 5.0))
    confirm_skills = bool(conf.get("skills", False))   # also gate the autonomous pick/place/move voice skills
    confirm_presets = bool(conf.get("presets", True))  # named poses ask "say yes" first (false: move at once)
    confirm_motion = bool(conf.get("motion", True))    # Cartesian/joint jogs ask first (false: move at once)
    confirm_pending = None   # a voice move waiting for a spoken yes: {"kind", ..., "deadline", "addr", "say"}
    base_max_vel = safety.max_vel.copy()   # the profile's velocity cap; voice "slower"/"faster" scales this
    voice_speed = 1.0                      # current voice velocity-cap factor (0.25 .. 1.0 of base_max_vel)
    voice_seq = []           # remaining steps of a running voice sequence; next starts when this one reaches at_goal
    # Short motion history (fix 1, 2026-10-23): the last few EXECUTED voice motion commands, so "again" / "the other
    # way" resolve locally and Nemotron gets a compact context. current_pose = the preset we are at (None once a jog
    # moves off it). STOP does not clear this: a new command after STOP still sees the history.
    motion_history = deque(maxlen=5)   # each: {"steps": [step, ...]} (preset steps carry their from_pose)
    current_pose = RETURN              # the sim starts at arm.home, which is the RETURN pose
    bubble = (cfg["net"].get("bubble_host", "127.0.0.1"), cfg["net"].get("bubble_port", 47101))   # VM voice bubble (say replies)
    if arm_mirror_on:
        log.info("body mode: arm_mirror (full-arm joint mirroring, no IK; side=%s)", retarget.side)
    else:
        if tune_path.exists():   # load a saved live-tuning override on top of the profile knobs (if present)
            try:
                if retarget.load(tune_path):
                    log.info("plane tuning override loaded: %s | %s", tune_path.name, retarget.snapshot())
            except Exception as exc:
                log.warning("plane tuning override %s failed to load: %s", tune_path.name, exc)
        log.info("body mode: plane (image-plane TCP teleop, depth ignored, IK orient=%s; control hand=%s)",
                 teleop_orient, retarget.side)

    def apply_plane_tune(op, addr=None, text="plane_tune"):
        """One live-tuning op for plane teleop (typed PlaneTune message or a voice phrase). Handled ONLY in plane
        mode; each op replies a spoken "say" + a knob snapshot, is logged, and (via the snapshot in status) shows on
        the HUD / body client. save writes the gitignored override, reset restores the profile baseline, tune_check
        starts the guided reach-threshold check."""
        if not plane_on:
            if addr is not None:
                rx.reply(addr, {"kind": "reply", "text": text, "ok": False,
                                "say": "Tuning is only available in plane mode.", "mode": arbiter.mode.value})
            return
        if op == "tune_check":
            tune_check.start(time.time())
            say_txt = tune_check.prompt
        elif op == "save":
            retarget.save(tune_path)
            say_txt = f"Tuning saved to {tune_path.name}."
        elif op == "reset":
            retarget.reset_tuning()
            say_txt = "Tuning reset to the profile defaults."
        else:
            say_txt = retarget.adjust(op)
            if say_txt is None:
                if addr is not None:
                    rx.reply(addr, {"kind": "reply", "text": text, "ok": False,
                                    "say": f"Unknown tuning command {op!r}.", "mode": arbiter.mode.value})
                return
        snap = retarget.snapshot()
        arbiter.last_say = say_txt
        rx.reply(bubble, {"state": "reply", "text": say_txt})   # Pranav's voice bubble on the VM screen
        log.info("plane tune %s -> %s | %s", op, say_txt, snap)
        if addr is not None:
            rx.reply(addr, {"kind": "reply", "text": text, "ok": True, "say": say_txt,
                            "mode": arbiter.mode.value, "plane_tune": snap})

    seq_seen = []  # (t, seq) over the last second, for packet loss
    vis_seen = []  # (t, arm visible, in position, hand found) over the last second: tracking quality

    def on_clutch(ev):
        """Clutch events -> arm modes. Calibrating/released freeze the arm; engaged starts following.
        If the arm profile sets `on_drop: <preset>`, losing the body ("tracking lost": your arm dropped out of view)
        sends the arm to that preset instead of freezing it. F / left-hand releases (deliberate re-centring) and
        "link lost" (nobody watching) still freeze in place."""
        nonlocal body_target, last_target, retarget, joint_goal, goal_name, body_joint_target, goal_start
        if ev is None:
            return
        log.info("clutch: %s (trigger=%s%s)", ev.kind, clutch.trigger,
                 f", reason={clutch.release_reason}" if clutch.release_reason else "")
        if ev.kind in ("calibrating", "released"):
            if arbiter.mode == Mode.FOLLOW or arbiter.assist:   # a running grab assist stops too (dead-man)
                arbiter.abort()
            body_target = body_joint_target = None
        if ev.kind == "released" and on_drop and clutch.release_reason in drop_reasons:
            if time.time() < hw_busy_until:
                log.info("on_drop %s skipped: the real arm is homing", on_drop)
                return
            goal, why = plan_preset(on_drop, presets, home, q_cmd, clutch.state, arm, safety.ws_min, safety.ws_max)
            if goal is not None:
                arbiter.on_intent(Intent(action=Action.stop))  # HOLD: the preset move owns the arm
                joint_goal, goal_name, goal_start = goal, on_drop, q_cmd.copy()
            log.info("on_drop (%s): %s", clutch.release_reason, why)
        elif ev.kind == "engaged":
            if arm_mirror_on:
                retarget.engage(q_cmd.copy())   # anchor the 2 s ramp at the current joints (no jump)
            else:
                retarget.engage(arm.fk(q_cmd)[:3, 3], ev.neutral)
            arbiter.on_intent(Intent(action=Action.follow))
            body_target = last_target = body_joint_target = None

    def do_preset(name, now):
        """Start a named joint-space preset move, reusing the button-panel path (plan_preset -> joint_goal, no IK).
        Returns (ok, message). The SafetyLayer (vel/acc limits) runs it; the ghost cues clamping at the box edge."""
        nonlocal joint_goal, goal_name, goal_start, current_pose
        if joint_goal is not None:          # BUSY rule: one move at a time; STOP (which clears joint_goal) always wins
            return False, BUSY_MSG()
        if hw_busy_until and now < hw_busy_until:
            return False, f"refused: the real arm is homing ({hw_busy_until - now:.0f} s left)"
        goal, why = plan_preset(name, presets, home, q_cmd, clutch.state, arm, safety.ws_min, safety.ws_max)
        if goal is None:
            return False, why
        arbiter.on_intent(Intent(action=Action.stop))  # HOLD: the preset move owns the arm
        joint_goal, goal_name, goal_start = goal, name, q_cmd.copy()
        current_pose = name            # the named pose we are now at / heading to (for "undo")
        return True, why

    def run_confirmed(pend, addr):
        """A spoken yes arrived inside the window: execute what was queued (a preset, a fast-path motion sequence,
        or an autonomous skill)."""
        if pend["kind"] == "preset":
            prev_pose = current_pose
            ok, why = do_preset(pend["name"], time.time())
            record_command([make_preset_step(pend["name"])], prev_pose, ok)
            log.info("voice preset %s confirmed: %s", pend["name"], why)
            rep = {"kind": "reply", "text": "confirm", "ok": ok, "say": why, "mode": arbiter.mode.value}
            if not ok and is_busy(why):
                rep["busy"] = True
            rx.reply(addr, rep)
        elif pend["kind"] == "fast":   # a voice motion / sequence the operator confirmed
            start_sequence(pend["steps"], "confirm", addr, 0.0)
        else:  # an autonomous skill the operator confirmed
            arbiter.on_intent(pend["intent"])
            log.info("voice skill confirmed: %s -> %s", pend["intent"].action.value, arbiter.mode.value)
            rx.reply(addr, {"kind": "reply", "text": "confirm",
                            "intent": pend["intent"].model_dump(mode="json", exclude_none=True),
                            "say": arbiter.last_say or pend["intent"].say, "mode": arbiter.mode.value})

    def start_cart_jog(direction, dist_m, now):
        """Cartesian jog: solve IK for the current TCP + a base-frame delta, holding the current tool orientation
        (iterated to convergence, not one ik_step), then run it through the same joint_goal -> SafetyLayer path as
        the presets. Returns (ok, message); refuses with a spoken reason if it can't reach there or leaves the box."""
        nonlocal joint_goal, goal_name, goal_start, current_pose
        if joint_goal is not None:          # BUSY rule: one move at a time (same as a preset); STOP always wins
            return False, BUSY_MSG()
        if hw_busy_until and now < hw_busy_until:
            return False, f"refused: the real arm is homing ({hw_busy_until - now:.0f} s left)"
        axis, sign = CART[direction]
        T_des_jog = arm.fk(q_cmd).copy()
        T_des_jog[axis, 3] += sign * dist_m
        goal_q = arm.ik(np.array(q_cmd, float), T_des_jog, iters=200, tol=1e-4, orient="full")
        reached = float(np.linalg.norm(arm.fk(goal_q)[:3, 3] - T_des_jog[:3, 3]))
        if reached > 0.01:   # 1 cm: the IK could not get there (near a limit / out of the reachable shell)
            return False, f"Can't reach {dist_m * 100:g} centimetres {direction}; out of reach."
        goal, why = check_joint_goal(goal_q, q_cmd, clutch.state, arm, safety.ws_min, safety.ws_max,
                                     label=f"{dist_m * 100:g} cm {direction}")
        if goal is None:
            return False, why
        arbiter.on_intent(Intent(action=Action.stop))   # HOLD: the jog owns the arm, same as a preset
        joint_goal, goal_name, goal_start = goal, f"jog {direction}", q_cmd.copy()
        current_pose = None            # a jog moves us off any named pose
        return True, why

    def start_joint_jog(joint_idx, angle_rad, now):
        """Joint jog: current joints + a signed delta on one joint, CLAMPED to the joint limits, through the same
        checks/path as the presets. Returns (ok, message)."""
        nonlocal joint_goal, goal_name, goal_start, current_pose
        if joint_goal is not None:          # BUSY rule: one move at a time (same as a preset); STOP always wins
            return False, BUSY_MSG()
        if hw_busy_until and now < hw_busy_until:
            return False, f"refused: the real arm is homing ({hw_busy_until - now:.0f} s left)"
        goal_q = np.array(q_cmd, float).copy()
        goal_q[joint_idx - 1] += angle_rad
        goal_q = np.clip(goal_q, arm.lower, arm.upper)   # clamp into the joint limits (never refuse on the cap)
        goal, why = check_joint_goal(goal_q, q_cmd, clutch.state, arm, safety.ws_min, safety.ws_max,
                                     label=f"{J_NAMES[joint_idx]} jog")
        if goal is None:
            return False, why
        arbiter.on_intent(Intent(action=Action.stop))
        joint_goal, goal_name, goal_start = goal, f"jog {J_NAMES[joint_idx]}", q_cmd.copy()
        current_pose = None            # a jog moves us off any named pose
        return True, why

    def apply_speed(mode):
        """Scale the SafetyLayer velocity cap for voice moves between 0.25x and 1.0x of the profile (never above)."""
        nonlocal voice_speed
        voice_speed = scale_speed(voice_speed, mode)
        safety.max_vel = base_max_vel * voice_speed
        log.info("voice speed %s -> %.2fx of the profile cap", mode, voice_speed)
        return True, ""

    def record_command(steps, prev_pose, ok):
        """Append an executed voice motion command to the history (fix 1), so "again" / "the other way" can replay
        or reverse it. Each preset step stores the pose we were at just before it (for a reversed = "undo" move).
        prev_pose is current_pose captured BEFORE the command ran (the executors update current_pose as it runs)."""
        if not ok:
            return
        rec, cur = [], prev_pose
        for s in steps:
            s2 = dict(s)
            if s2["kind"] == "preset":
                s2["from_pose"] = cur
                cur = s2["name"]
            elif s2["kind"] in ("cartjog", "jointjog"):
                cur = None
            rec.append(s2)
        if any(s["kind"] in ("preset", "cartjog", "jointjog") for s in rec):
            motion_history.append({"steps": rec})

    def context_str():
        """Compact recent context for the Nemotron prompt: the last 3 commands + the current named pose."""
        parts = [" ".join(s.get("say", "") for s in c["steps"]).strip() for c in list(motion_history)[-3:]]
        where = f"at the {current_pose} pose" if current_pose else "not at a named pose"
        return ("; ".join(p for p in parts if p) + f" [{where}]").strip()

    def _start_step(step, now):
        """Run one parsed step. Returns (ok, spoken text). A motion step sets joint_goal; speed is instant."""
        k = step["kind"]
        if k == "preset":
            ok, why = do_preset(step["name"], now)
        elif k == "cartjog":
            ok, why = start_cart_jog(step["direction"], step["dist_m"], now)
        elif k == "jointjog":
            ok, why = start_joint_jog(step["joint"], step["angle_rad"], now)
        elif k == "speed":
            ok, why = apply_speed(step["mode"])
        else:
            ok, why = False, "Unknown command."
        return ok, (step["say"] if ok else why)

    def _advance_sequence(now):
        """Start steps until one is a motion step now running (wait for at_goal) or the list empties. A refused
        step cancels the rest. Returns (ok, last spoken text)."""
        nonlocal voice_seq
        say = ""
        while voice_seq:
            step = voice_seq.pop(0)
            ok, say = _start_step(step, now)
            if not ok:
                voice_seq = []
                log.info("voice sequence stopped: %s", say)
                return False, say
            if step["kind"] != "speed":   # a motion step is running: the loop advances on at_goal
                return True, say
        return True, say

    def start_sequence(steps, text, addr, latency):
        """Execute a parsed fast-path command (one step or a 'then' sequence) with NO spoken confirm."""
        nonlocal voice_seq
        prev_pose = current_pose       # snapshot BEFORE the steps run (executors update current_pose)
        voice_seq = list(steps)
        ok, say = _advance_sequence(time.time())
        record_command(steps, prev_pose, ok)
        combined = " ".join(s["say"] for s in steps if s.get("say"))
        arbiter.last_say = combined if ok else say
        log.info("voice fast-path %r -> %s (%.3fs)", text, [s["say"] for s in steps], latency)
        rep = {"kind": "reply", "text": text, "latency_s": round(latency, 3), "ok": ok,
               "say": combined if ok else say, "mode": arbiter.mode.value}
        if not ok and is_busy(say):      # a voice motion refused because a move is already running
            rep["busy"] = True
        rx.reply(addr, rep)

    def execute_or_confirm(steps, text, addr, latency):
        """Fast-path dispatch: move at once unless confirm.presets / confirm.motion ask for a spoken yes first."""
        nonlocal confirm_pending
        need = (confirm_presets and any(s["kind"] == "preset" for s in steps)) or \
               (confirm_motion and any(s["kind"] in ("cartjog", "jointjog") for s in steps))
        combined = " ".join(s["say"] for s in steps if s.get("say"))
        if need:
            say = f"{combined} Say yes to confirm."
            confirm_pending = {"kind": "fast", "steps": steps, "deadline": time.time() + confirm_window_s,
                               "addr": addr, "say": say}
            log.info("voice motion (%d step%s): waiting for confirm (%.0fs)", len(steps),
                     "" if len(steps) == 1 else "s", confirm_window_s)
            rx.reply(addr, {"kind": "reply", "text": text, "latency_s": round(latency, 3), "confirm": True,
                            "say": say, "mode": arbiter.mode.value})
        else:
            start_sequence(steps, text, addr, latency)

    def _intent_to_steps(intent):
        """Map a Nemotron/fallback motion Intent (jog / joint_jog / speed) to fast-path steps, else None (poses and
        skills are handled elsewhere in finish())."""
        if intent.action == Action.jog and intent.direction in CART:
            d = intent.distance_cm
            dist_m = DEFAULT_DIST_M if d is None else max(0.0, float(d)) / 100.0
            return [make_cart_step(intent.direction, dist_m)]
        if intent.action == Action.joint_jog and intent.joint:
            jw = intent.joint.strip().lower()
            if jw not in J_WORDS:
                return None
            joint = J_WORDS[jw]
            ang = DEFAULT_ANGLE if intent.angle_deg is None else np.radians(abs(float(intent.angle_deg)))
            dw = (intent.direction or "").strip().lower()
            if dw in J_SIGN[joint]:
                return [make_joint_step(joint, J_SIGN[joint][dw], ang, dw)]
            if intent.angle_deg is not None and float(intent.angle_deg) != 0:
                return [make_joint_step(joint, 1 if float(intent.angle_deg) > 0 else -1, ang)]
            return None
        if intent.action == Action.speed and intent.speed in ("slower", "faster", "normal"):
            return [make_speed_step(intent.speed)]
        return None

    def finish(text, addr, intent, latency):
        nonlocal q_cmd, T_des, body_target, joint_goal, mirror_real, confirm_pending, voice_seq
        log.info("heard %r -> %s (%.1fs)", text, intent.model_dump(exclude_none=True), latency)
        if voice_seq:   # any fresh command (or stop) cancels the rest of a running voice sequence
            log.info("voice sequence (%d left) cancelled by %r", len(voice_seq), text)
            voice_seq = []
        if intent.action in (Action.stop, Action.reset) and mirror_real:
            mirror_real = False  # STOP always leaves the mirror-real state
            log.info("mirror real left by %s", intent.action.value)
        if intent.action in (Action.stop, Action.reset) and joint_goal is not None:
            log.info("preset %s cancelled by %s", goal_name, intent.action.value)
            joint_goal = None
        if intent.action in (Action.stop, Action.reset):  # a slow cloud result must not restart the arm
            if confirm_pending is not None:
                log.info("pending confirm (%s) cancelled by %s", confirm_pending.get("name") or "skill",
                         intent.action.value)
                confirm_pending = None
            for p_text, p_addr, _, fut in pending:
                fut.cancel()
                rx.reply(p_addr, {"kind": "reply", "text": p_text, "say": "Cancelled by stop.", "mode": "hold"})
            pending.clear()
        # a named pose (voice) or the autonomous skills (when confirm.skills is on): ask first, do not move yet.
        want = RETURN if intent.action == Action.home else intent.preset if intent.action == Action.preset else None
        if want is not None and not confirm_presets:   # "just push it" (dev_arm_real): move now, no spoken yes
            prev_pose = current_pose
            ok, why = do_preset(want, time.time())
            record_command([make_preset_step(want)], prev_pose, ok)
            log.info("voice preset %s (no confirm): %s", want, why)
            rep = {"kind": "reply", "text": text, "latency_s": round(latency, 2), "ok": ok,
                   "intent": intent.model_dump(mode="json", exclude_none=True),
                   "say": why, "mode": arbiter.mode.value}
            if not ok and is_busy(why):
                rep["busy"] = True
            rx.reply(addr, rep)
            return
        if want is not None:
            label = "the start pose" if want == RETURN else want
            say = f"Moving to {label}. Say yes to confirm."
            confirm_pending = {"kind": "preset", "name": want, "deadline": time.time() + confirm_window_s,
                               "addr": addr, "say": say}
            log.info("voice preset %s: waiting for confirm (%.0fs)", want, confirm_window_s)
            rx.reply(addr, {"kind": "reply", "text": text, "latency_s": round(latency, 2), "confirm": True,
                            "intent": intent.model_dump(mode="json", exclude_none=True),
                            "say": say, "mode": arbiter.mode.value})
            return
        msteps = _intent_to_steps(intent)   # free-phrased jog / joint_jog / speed from Nemotron -> fast-path steps
        if msteps is not None:
            execute_or_confirm(msteps, text, addr, latency)
            return
        if confirm_skills and intent.action in (Action.pick, Action.place, Action.move):
            say = f"{intent.say or 'Ready'} Say yes to confirm."
            confirm_pending = {"kind": "intent", "intent": intent, "deadline": time.time() + confirm_window_s,
                               "addr": addr, "say": say}
            log.info("voice skill %s: waiting for confirm (%.0fs)", intent.action.value, confirm_window_s)
            rx.reply(addr, {"kind": "reply", "text": text, "latency_s": round(latency, 2), "confirm": True,
                            "intent": intent.model_dump(mode="json", exclude_none=True),
                            "say": say, "mode": arbiter.mode.value})
            return
        if intent.action == Action.follow:  # "follow me" = start the clutch (hold still -> calibrate -> follow)
            if clutch.state == "idle":
                on_clutch(clutch.toggle(time.time()))
        else:
            if intent.action in (Action.stop, Action.reset) and clutch.state != "idle":
                on_clutch(clutch._release())
            arbiter.on_intent(intent)
        if intent.action == Action.reset:
            robot.reset(home)
            safety.reset()
            retarget.reset()
            q_cmd, T_des, body_target = home.copy(), arm.fk(home), None
        rx.reply(addr, {"kind": "reply", "text": text, "latency_s": round(latency, 2),
                        "intent": intent.model_dump(mode="json", exclude_none=True),
                        "say": arbiter.last_say or intent.say, "mode": arbiter.mode.value})

    t_end = None if seconds is None else time.time() + seconds
    next_status = next_body_status = 0.0
    body_addr, n_body = None, 0
    while t_end is None or time.time() < t_end:
        t0 = time.time()
        for t_recv, msg, addr in rx.poll():
            recorder.record(t_recv, msg)   # every incoming message except Query, for offline replay (issue 9)
            if isinstance(msg, BodyPose):
                n_body += 1
                body_addr, last_body, last_body_arrival = addr, msg, t0
                if msg.visible:
                    last_visible_t = t0
                seq_seen.append((t0, msg.seq))
                vis_seen.append((t0, bool(msg.visible), bool(msg.in_position), msg.hand_open is not None))
                if not mirror_real:  # mirroring the real arm: ignore the body for motion (still track it for the HUD)
                    rel = np.asarray(msg.wrist) - np.asarray(msg.shoulder) if msg.visible else None
                    palm_up_open = bool(msg.hand_open is not None and msg.hand_open > 0.6 and msg.elbow is not None
                                        and msg.wrist[1] < msg.elbow[1] - 0.02)          # MediaPipe y points down
                    left_raised = bool(msg.wrist_other is not None and msg.shoulder_other is not None
                                       and msg.wrist_other[1] < msg.shoulder_other[1] - clutch.raise_m)
                    if plane_on:
                        left_raised = False   # plane mode drives REACH from the left hand, so it can't be the clutch
                    on_clutch(clutch.update(rel, palm_up_open, left_raised, t0))
                    if plane_on and clutch.state == "following":   # control wrist held below the hip line -> stow
                        pc = plane_coords(msg, retarget.side) if msg.visible else None
                        if pc is not None and plane_drop.update(pc["below_hip"], t0):
                            plane_drop.reset()
                            on_clutch(clutch._release("right arm dropped"))
                    else:
                        plane_drop.reset()
                    if plane_on and tune_check.active:   # guided reach-threshold check: sample poses, fit on done
                        tune_check.update(t0, plane_coords(msg, retarget.side) if msg.visible else None)
                        if tune_check.active:
                            prm = tune_check.pop_prompt()
                            if prm:
                                arbiter.last_say = prm
                                rx.reply(bubble, {"state": "reply", "text": prm})
                        if tune_check.finished:
                            res = tune_check.result
                            tune_check.finished = False
                            retarget.set_reach_thresholds(res["forward_frac"], res["back_frac"])
                            done_say = (f"Tune check done. Forward {res['forward_frac']:.2f}, "
                                        f"back {res['back_frac']:.2f}. Say save tuning to keep it.")
                            arbiter.last_say = done_say
                            rx.reply(bubble, {"state": "reply", "text": done_say})
                            log.info("plane tune check: fwd=%.3f back=%.3f measured=%s | %s",
                                     res["forward_frac"], res["back_frac"], res["measured"], retarget.snapshot())
                    if msg.visible and clutch.state == "following" and arbiter.mode == Mode.FOLLOW:
                        tgt = retarget(msg)
                        if tgt is not None and arm_mirror_on:
                            body_joint_target, body_t = tgt, t0          # 6-vector joints, straight in (no IK)
                        elif tgt is not None:
                            body_target, body_t, last_target = tgt, t0, tgt
                        if grip_from_hand and msg.hand_open is not None:
                            # teleop modes keep the gripper a plain open/close (tcp=None -> no grab-assist AUTO plan)
                            arbiter.on_hand(msg.hand_open, t0, None if plain_gripper else arm.fk(robot.read_joints())[:3, 3])
            elif isinstance(msg, ClutchMsg) and mirror_real:   # clutch is motion: refused while mirroring
                rx.reply(addr, {"kind": "reply", "text": "clutch", "say": "Mirroring the real arm; say stop to leave.",
                                "mode": arbiter.mode.value, "mirror": True})
            elif isinstance(msg, ClutchMsg) and msg.action in ("warmup", "calibrate", "redo_axis"):
                # calibration was removed with the legacy position mode: ignore the old C / X / warm-up actions so an
                # older body client does not accidentally toggle the clutch.
                rx.reply(addr, {"kind": "reply", "text": "clutch",
                                "say": "Calibration was removed; use plane or arm-mirror teleop.",
                                "mode": arbiter.mode.value})
            elif isinstance(msg, ClutchMsg):
                ev = clutch.toggle(t0)
                if (arm_mirror_on or plane_on) and ev is not None and ev.kind == "released" \
                        and clutch.release_reason is None:
                    clutch.release_reason = "disabled"   # deliberate F toggle off in a teleop mode -> return to base
                on_clutch(ev)
                rx.reply(addr, {"kind": "reply", "text": "clutch", "say": f"clutch: {clutch.state}",
                                "mode": arbiter.mode.value})
            elif isinstance(msg, Utterance):
                stop = urgent_stop(msg.text)
                if stop is not None:  # handled in this tick; never queued behind a cloud call (STOP also leaves
                    finish(msg.text, addr, stop, 0.0)  # mirror; finish clears any armed confirm)
                elif mirror_real:  # any non-stop command is motion: refuse while mirroring the real arm
                    rx.reply(addr, {"kind": "reply", "text": msg.text, "mode": arbiter.mode.value, "mirror": True,
                                    "say": "Mirroring the real arm; say stop to leave."})
                elif confirm_pending is not None and is_confirm(msg.text):
                    pend, confirm_pending = confirm_pending, None   # a spoken yes inside the window: execute it
                    run_confirmed(pend, addr)
                else:
                    if confirm_pending is not None:   # anything else cancels the armed confirm, then parse afresh
                        log.info("pending confirm (%s) cancelled: %r",
                                 confirm_pending.get("name") or "skill", msg.text)
                        rx.reply(confirm_pending["addr"], {"kind": "reply", "text": "confirm", "ok": False,
                                                           "say": "Cancelled.", "mode": arbiter.mode.value})
                        confirm_pending = None
                    # one local grammar for the fast path: parse_fast wraps quick_pose for poses AND adds the motion
                    # grammar, so it resolves the "base" pose vs base-joint collision ("turn the base left 30 deg").
                    fp = parse_fast(msg.text, scene.names(), list(presets), history=list(motion_history))
                    if fp is not None and fp["steps"] and all(s["kind"] == "plane_tune" for s in fp["steps"]):
                        for s in fp["steps"]:   # spoken plane-tuning phrase (plane mode only; replies in-tick)
                            apply_plane_tune(s["op"], addr, msg.text)
                    elif fp is not None and len(fp["steps"]) == 1 and fp["steps"][0]["kind"] == "preset":
                        step = fp["steps"][0]   # a lone named pose: reuse the pose path (confirm.presets, reply+intent)
                        finish(msg.text, addr, Intent(action=Action.preset, preset=step["name"], say=step["say"]), 0.0)
                    elif fp is not None:    # jog / joint jog / speed / "then" sequence: also parsed in-tick, no cloud
                        execute_or_confirm(fp["steps"], msg.text, addr, time.time() - t_recv)
                    elif msg.offline:
                        finish(msg.text, addr, fallback_parse(msg.text, scene.names(), list(presets)), 0.0)
                    else:  # Nemotron takes seconds: run it off the control loop
                        pending.append((msg.text, addr, time.time(),
                                        pool.submit(parser.parse, msg.text, scene.names(), descriptions,
                                                    list(presets), context_str())))
            elif isinstance(msg, EEGState):
                arbiter.on_eeg(msg.p_grasp, t0)
            elif isinstance(msg, ArmPreset) and mirror_real:
                ok, why = False, "refused: mirroring the real arm (say stop to leave)"
                log.info("preset %s: %s", msg.name, why)
                rx.reply(addr, {"kind": "reply", "text": f"preset {msg.name}", "ok": ok, "say": why,
                                "mode": arbiter.mode.value, "mirror": True})
            elif isinstance(msg, ArmPreset) and joint_goal is not None:
                # a preset move is already running: do NOT chain-replace it (that cancelled the previous move and left
                # Pranav seeing nothing). Refuse and tell him to STOP first; STOP (an Utterance) always gets through.
                why = f"busy: {goal_name} still moving, press STOP first"
                log.info("preset %s refused: %s", msg.name, why)
                rx.reply(addr, {"kind": "reply", "text": f"preset {msg.name}", "ok": False, "say": why,
                                "mode": arbiter.mode.value, "busy": True})
            elif isinstance(msg, ArmPreset):
                if msg.name == HOME_HW:
                    if t0 < hw_busy_until:
                        ok, why = False, f"refused: the real arm is already homing ({hw_busy_until - t0:.0f} s left)"
                    else:
                        ok, why = request_hw_home(real_arm, msg.confirm)
                    if ok:
                        if clutch.state != "idle":
                            on_clutch(clutch._release())
                        arbiter.on_intent(Intent(action=Action.stop))
                        joint_goal, goal_name = None, None
                        hw_busy_until = t0 + float(real_arm.get("busy_s", 120.0))
                else:
                    # same path the voice confirm uses; do_preset refuses a second move as busy while one runs so a
                    # panel tap mid-move no longer chain-replaces the move (STOP, an Utterance, always gets through).
                    ok, why = do_preset(msg.name, t0)
                log.info("preset %s%s: %s", msg.name, " (confirmed)" if msg.confirm else "", why)
                rep = {"kind": "reply", "text": f"preset {msg.name}", "ok": ok, "say": why, "mode": arbiter.mode.value}
                if not ok and is_busy(why):
                    rep["busy"] = True
                rx.reply(addr, rep)
            elif isinstance(msg, GripToggle):
                arbiter.on_grip_toggle(None if plain_gripper else arm.fk(robot.read_joints())[:3, 3])
                log.info("grip toggle (%s) in %s: gripper=%.0f", msg.source, arbiter.mode.value, arbiter.gripper)
            elif isinstance(msg, RealJoints):
                # one sample of the PHYSICAL arm (real_to_sim). Stamp it with the VM receive time (the shared clock),
                # not the payload's t. Recorded above like every other inbound message.
                if len(msg.q) >= 6:
                    real_q, real_fw, real_t = np.array(msg.q[:6], float), list(msg.fw), t_recv
            elif isinstance(msg, MirrorMsg):
                if msg.on and not mirror_real:  # entering mirror: drop the clutch and HOLD so the body can't fight it
                    if clutch.state != "idle":
                        on_clutch(clutch._release())
                    arbiter.on_intent(Intent(action=Action.stop))
                    joint_goal, goal_name = None, None
                    body_target = None
                mirror_real = bool(msg.on)
                log.info("mirror real %s", "on" if mirror_real else "off")
                rx.reply(addr, {"kind": "reply", "text": "mirror", "mirror": mirror_real, "mode": arbiter.mode.value,
                                "say": "Mirroring the real arm." if mirror_real else "Mirror off."})
            elif isinstance(msg, PlaneTune):   # live tuning of plane teleop (body client keys); plane mode only
                apply_plane_tune(msg.op, addr, "plane_tune")
            elif isinstance(msg, CameraFrame):
                # live webcam stream for the panel under the HUD. Forward the frame straight to the HUD's video
                # channel; keep it off the pose path entirely (separate message, not recorded, latest-wins at the HUD).
                try:
                    rx.reply(hud_video, msg.model_dump(mode="json"))
                except OSError:
                    pass  # nothing listening on the video port yet: drop it, never block the loop
            elif isinstance(msg, Query):
                last_query_t, follower_ever = t0, True  # the follower polls here: drives the real_following light
                rx.reply(addr, {"kind": "state", "mode": arbiter.mode.value, "held": arbiter.held,
                                "clutch": clutch.state, "assist": arbiter.assist, "mapping": type(retarget).__name__,
                                "gripper": arbiter.gripper, "tcp": np.round(arm.fk(q_cmd)[:3, 3], 4).tolist(),
                                "tcp_meas": np.round(arm.fk(robot.read_joints())[:3, 3], 4).tolist(),
                                "q_meas": np.round(robot.read_joints(), 4).tolist(),
                                "plan_left": len(arbiter._plan), "table_top": scene.support_z,
                                "preset": goal_name if joint_goal is not None else None,
                                # plane rate-reach state so the body client / HUD can show "REACH >>" / "<<" / "hold"
                                "reach_rate": round(float(getattr(retarget, "reach_rate", 0.0)), 4) if plane_on else None,
                                "reach_state": getattr(retarget, "reach_state", None) if plane_on else None,
                                "hw_busy": round(max(0.0, hw_busy_until - t0), 1) if hw_busy_until else 0.0,
                                "mirror": mirror_real,
                                "mirror_age": round(t0 - real_t, 3) if (mirror_real and real_t) else None,
                                "objects": {n: {"pos": np.round(o.pos, 4).tolist(),
                                                "bottom_z": round(float(o.pos[2] - o.grasp_z), 4)}
                                            for n, o in scene.objects.items()}})
        if confirm_pending is not None and t0 > confirm_pending["deadline"]:   # no yes in time: cancel the move
            log.info("pending confirm (%s) timed out", confirm_pending.get("name") or "skill")
            rx.reply(confirm_pending["addr"], {"kind": "reply", "text": "confirm", "ok": False,
                                               "say": "Confirmation timed out, cancelled.", "mode": arbiter.mode.value})
            confirm_pending = None
        on_clutch(clutch.watchdog(t0, last_body_arrival, last_visible_t))  # auto-release on link/tracking loss
        for item in [x for x in pending if x[3].done()]:
            pending.remove(item)
            text, addr, t_sub, fut = item
            try:
                intent = fut.result()
            except Exception as e:  # network/API error: say so, don't move
                log.warning("nemotron failed: %s", e)
                intent = Intent(action=Action.unknown, say="Language service error.")
            finish(text, addr, intent, time.time() - t_sub)
        q = robot.read_joints()
        if hasattr(robot, "objects"):
            for name, bottom in robot.objects.items():
                scene.update_from_bottom(name, bottom)
        tcp = arm.fk(q)[:3, 3]
        p, grip = arbiter.target(tcp, body_target if t0 - body_t < 0.5 else None)
        for s in arbiter.say_log:   # drain the arbiter's say queue: each message logged once, when it happened
            log.info("arbiter: %s", s)
        arbiter.say_log.clear()
        if grip != last_grip:
            log.info("gripper %s (%.0f -> %.0f) in %s%s", "CLOSE" if grip > last_grip else "OPEN",
                     last_grip, grip, arbiter.mode.value, f", holding {arbiter.held}" if arbiter.held else "")
            last_grip = grip
        while arbiter.events:
            if arbiter.events.pop(0) == "assist_done" and clutch.state == "following":
                # hand control back without a jump: position mode re-anchors at the arm, calibrated mode glides,
                # arm_mirror re-ramps from the current joints (assist is off in arm_mirror, but keep this safe).
                if arm_mirror_on:
                    retarget.engage(q_cmd.copy())
                    body_joint_target = None
                else:
                    try:
                        retarget.engage(arm.fk(q_cmd)[:3, 3], None, recentre=False)
                    except TypeError:
                        retarget.engage(arm.fk(q_cmd)[:3, 3], None)
                body_target = last_target = None
        if p is not None:
            T_des[:3, 3] = safety.clip_workspace(p)
        else:  # HOLD = stop here: aim at the current commanded pose, the safety layer brakes at max decel
            T_des = arm.fk(q_cmd)
        input_age = 0.0 if arbiter.mode == Mode.AUTO or p is None else t0 - body_t
        if joint_goal is not None and (arbiter.mode != Mode.HOLD or clutch.state != "idle"):
            log.info("preset %s cancelled: %s took over", goal_name,
                     "the clutch" if clutch.state != "idle" else f"{arbiter.mode.value} mode")
            joint_goal = None
            if voice_seq:   # the running sequence is interrupted by the clutch / a mode change
                log.info("voice sequence (%d left) cancelled: %s took over", len(voice_seq), goal_name)
                voice_seq = []
        # mirror real: the sim's joint target is the PHYSICAL arm's joints straight (no IK). If the real samples go
        # stale (> 0.5 s) the sim holds. Button-panel preset: joint-space target straight into the SafetyLayer, no IK.
        # teleop (FOLLOW) can free/freeze the wrist (orient from the profile); AUTO grasps keep full orientation.
        mirror_age = (t0 - real_t) if real_t else None
        mirror_hold = mirror_real and (real_q is None or mirror_age is None or mirror_age > MIRROR_STALE_S)
        orient = teleop_orient if arbiter.mode == Mode.FOLLOW else "full"
        arm_mirror_fresh = (arm_mirror_on and arbiter.mode == Mode.FOLLOW and clutch.state == "following"
                            and body_joint_target is not None and t0 - body_t < 0.5)
        if mirror_real and not mirror_hold:
            q_des = real_q
        elif mirror_real:                                    # real samples stale: hold the sim where it is
            q_des = q_cmd
        elif joint_goal is not None:
            q_des = joint_goal
        elif arm_mirror_fresh:                               # full-arm mirror: joints straight into the SafetyLayer
            q_des = body_joint_target
        else:
            q_des = arm.ik_step(q_cmd, T_des, orient=orient)
        q_cmd = safety.step(q_cmd, q_des, dt, 0.0 if mirror_real else input_age)
        q_cmd = np.clip(q_cmd, q - WINDUP, q + WINDUP)  # anti-windup: never run far ahead of the real joints
        if joint_goal is not None and at_goal(q_cmd, q, joint_goal):
            log.info("preset %s reached", goal_name)
            joint_goal = None
            if voice_seq:   # a sequence step just finished: start the next one in the same tick
                ok, step_say = _advance_sequence(time.time())
                if ok and step_say:
                    arbiter.last_say = step_say
                    log.info("voice sequence -> %s", step_say)
        if hw_busy_until and t0 >= hw_busy_until:  # real-arm homing window over: put the sim at the start pose
            hw_busy_until = 0.0
            robot.reset(home)
            safety.reset()
            retarget.reset()
            q_cmd, T_des, body_target = home.copy(), arm.fk(home), None
            log.info("real-arm homing window over: sim reset to the start pose")
        # sync cues: ghost at the target. ok = green, clamped = amber, lost = red, calib = blue
        ghost, ghost_state = None, None
        if clutch.state == "calibrating":
            ghost, ghost_state = arm.fk(q_cmd)[:3, 3], "calib"
        elif clutch.state == "following" and arbiter.mode == Mode.FOLLOW and arm_mirror_on:
            if body_joint_target is not None and t0 - body_t <= 0.4:   # ghost at the mirrored gripper pose
                ghost, ghost_state = arm.fk(body_joint_target)[:3, 3], "clamped" if retarget.clipped else "ok"
            elif body_joint_target is not None:
                ghost, ghost_state = arm.fk(q_cmd)[:3, 3], "lost"
        elif clutch.state == "following" and arbiter.mode == Mode.FOLLOW:
            if last_target is not None and t0 - body_t > 0.4:
                ghost, ghost_state = last_target, "lost"
            elif p is not None:
                clamped = getattr(retarget, "clipped", False) or np.max(np.abs(safety.clip_workspace(p) - p)) > 0.005
                ghost, ghost_state = T_des[:3, 3], "clamped" if clamped else "ok"
        elif arbiter.mode == Mode.AUTO and p is not None:
            ghost, ghost_state = T_des[:3, 3], "ok"
        ghost_bad = ghost_state in ("clamped", "lost")
        cand = (arbiter.grab_candidate(tcp) if grip_from_hand and arbiter.mode == Mode.FOLLOW
                and clutch.state == "following" else None)
        vtarget = np.round(cand.pos, 4).tolist() if cand is not None else None
        robot.command(q_cmd, grip, ghost=ghost, ghost_bad=ghost_bad, ghost_state=ghost_state, vtarget=vtarget)
        if t0 >= next_body_status:  # 10 Hz: status to the body client (echo for RTT) + HUD overlay on the VM screen
            next_body_status = t0 + 0.1
            seq_seen[:] = [x for x in seq_seen if t0 - x[0] <= 1.0]
            vis_seen[:] = [x for x in vis_seen if t0 - x[0] <= 1.0]
            pct = lambda i: round(100 * sum(x[i] for x in vis_seen) / len(vis_seen)) if vis_seen else None
            span = (seq_seen[-1][1] - seq_seen[0][1] + 1) if len(seq_seen) > 1 else len(seq_seen)
            loss = 1.0 - len(seq_seen) / span if span > 0 else 0.0
            preset_progress = None
            if joint_goal is not None and goal_start is not None:
                span_q = float(np.linalg.norm(np.asarray(joint_goal) - goal_start))
                done = float(np.linalg.norm(q_cmd - goal_start))
                preset_progress = round(min(1.0, max(0.0, done / span_q)), 2) if span_q > 1e-6 else 1.0
            st = {"kind": "status", "mode": arbiter.mode.value, "tcp": np.round(tcp, 3).tolist(),
                  "target": None if ghost is None else np.round(ghost, 3).tolist(), "ghost": ghost_state,
                  "clipped": ghost_bad, "grip": grip, "clutch": clutch.state, "trigger": clutch.trigger, "release_reason": clutch.release_reason,
                  "clutch_progress": round(clutch.progress, 2), "ws_min": safety.ws_min.tolist(), "ws_max": safety.ws_max.tolist(),
                  "echo_seq": last_body.seq if last_body else None, "echo_t": last_body.t if last_body else None,
                  "echo_age": round(t0 - last_body_arrival, 4) if last_body else None,
                  "pkt_s": len(seq_seen), "loss": round(max(0.0, loss), 3),
                  "vis_pct": pct(1), "inpos_pct": pct(2), "hand_pct": pct(3),
                  "calib": None, "mapping": type(retarget).__name__, "set_mirror": set_mirror,
                  "vtarget": vtarget, "grab_ready": cand.name if cand is not None else None,
                  "assist": arbiter.assist, "held": arbiter.held, "tcp_meas": np.round(tcp, 3).tolist(),
                  "presets": preset_names(presets, bool(real_arm.get("enabled"))),
                  "preset": goal_name if joint_goal is not None else None,
                  "preset_progress": preset_progress,
                  "real_following": (None if not follower_ever else (t0 - last_query_t < 1.0)),
                  "hw_busy": round(max(0.0, hw_busy_until - t0), 1) if hw_busy_until else 0.0,
                  # plane live-tuning: knob snapshot (reach state + gains/thresholds) for the compact HUD line, and
                  # the guided tune-check prompt while it runs
                  "plane_tune": retarget.snapshot() if plane_on else None,
                  "tune": ({"active": True, "prompt": tune_check.prompt, "progress": round(tune_check.progress, 2)}
                           if plane_on and tune_check.active else None)}
            if body_addr is not None:
                rx.reply(body_addr, st)
                set_mirror = None if set_mirror else set_mirror  # sent once
            rx.reply(hud, st | {"skel2d": last_body.skel2d if last_body else None,
                                "rtt_ms": last_body.rtt_ms if last_body else None,
                                "body_age": round(t0 - body_t, 2) if body_t else None,
                                "hand_open": last_body.hand_open if last_body else None,
                                "say": arbiter.last_say})
        if t0 >= next_status:
            next_status = t0 + 1.0
            vs = [x for x in vis_seen if t0 - x[0] <= 1.0]
            q_ = (lambda i: round(100 * sum(x[i] for x in vs) / len(vs)) if vs else -1)
            log.info("mode=%s tcp=%s grip=%.0f body=%d pkt/s arm-visible=%d%% in-position=%d%% hand=%d%% clutch=%s%s%s",
                     arbiter.mode.value, np.round(tcp, 3), grip, n_body, q_(1), q_(2), q_(3), clutch.state,
                     f" from {body_addr[0]}" if body_addr else "", " ESTOP:" + safety.reason if safety.estopped else "")
            n_body = 0
        time.sleep(max(0.0, dt - (time.time() - t0)))
    recorder.close()


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); real environment variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--arm", default=None, help="arm profile (configs/arms/<name>.yaml); default: ROVER_ARM or config")
    ap.add_argument("--backend", default=None, choices=["mock", "isaac", "real"])
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--no-record", action="store_true", help="do not record this session to logs/sessions/")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.arm)
    log.info("arm profile: %s (%s)", cfg["arm_profile"], cfg["arm"]["urdf"])
    run(cfg, args.backend or cfg["loop"]["backend"], args.seconds, record=not args.no_record)


if __name__ == "__main__":
    main()

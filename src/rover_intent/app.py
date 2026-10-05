"""Main control loop (runs on the VM, or wherever the arm backend is).

  laptop (body pose, transcripts) --UDP--> Receiver
  transcripts -> Nemotron -> Intent -> Arbiter
  EEG decoder p_grasp -> Arbiter
  Arbiter -> TCP target -> IK step -> SafetyLayer -> backend

Run:  python -m rover_intent.app --config configs/default.yaml [--backend mock|isaac] [--seconds N]
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from .body.mirror import GestureMirror
from .body.retarget import Retargeter
from .control.arbiter import Arbiter, Mode
from .control.kinematics import ArmModel
from .control.safety import SafetyLayer
from .planner.nemotron import IntentParser, urgent_stop
from .planner.scene import Scene
from .robot.backends import IsaacArm, MockArm, RealArm
from .transport.udp import Receiver
from .planner.nemotron import fallback_parse
from .body.clutch import Clutch
from .config import load_config
from .body.calib_session import CalibratedMapper, CalibrationSession
from .body.calibration import Profile
from .types import Action, BodyPose, ClutchMsg, EEGState, Intent, Query, Utterance

log = logging.getLogger("rover_intent")
ROOT = Path(__file__).resolve().parents[2]
WINDUP = 0.25  # rad: max gap between the commanded and the measured joint positions


def build(cfg: dict, backend: str):
    a = cfg["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), base=a.get("base_link", "base_link"), tip=a["tip_link"],
                   tcp_offset=a["tcp_offset"])
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
    oe = b.get("one_euro", {})
    retarget = Retargeter(b["axes"], b["scale"], b["origin"], min_cutoff=oe.get("min_cutoff", 1.0),
                          beta=oe.get("beta", 0.7), ws_min=cfg["safety"]["workspace_min"],
                          ws_max=cfg["safety"]["workspace_max"])  # "position" mode (default)
    if b.get("mode", "position") == "gesture":
        retarget = GestureMirror(arm, gains=b.get("gesture_gains", (2.5, 1.0, 0.6)),
                                 height=(cfg["safety"]["workspace_min"][2], cfg["safety"]["workspace_max"][2]),
                                 min_cutoff=oe.get("min_cutoff", 1.0), beta=oe.get("beta", 0.5))
    n = cfg["nemotron"]
    parser = IntentParser(n["model"], n["base_url"], n["api_key_env"])
    return arm, safety, robot, scene, arbiter, retarget, parser


def run(cfg: dict, backend: str, seconds: float | None = None):
    arm, safety, robot, scene, arbiter, retarget, parser = build(cfg, backend)
    rx = Receiver(cfg["net"]["listen_port"])
    dt = 1.0 / cfg["loop"]["hz"]
    log.info("backend=%s nemotron=%s objects=%s", backend, "online" if parser.online else "OFFLINE fallback",
             scene.names())
    descriptions = {n: o.get("desc", "") for n, o in cfg["scene"].items()}
    home = np.array(cfg["arm"]["home"], float)
    grip_from_hand = cfg["body"].get("grip_from_hand", False)
    body_target, body_t = None, 0.0
    q_cmd = robot.read_joints()  # commanded trajectory state (the drives track it; see WINDUP)
    T_des = arm.fk(q_cmd)
    pool, pending = ThreadPoolExecutor(max_workers=1), []
    c = cfg["body"].get("clutch", {})
    clutch = Clutch(calib_s=c.get("calib_s", 2.0), still_m=c.get("still_m", 0.03), palm_s=c.get("palm_s", 1.0),
                    raise_m=c.get("raise_m", 0.05))
    hud = (cfg["net"].get("hud_host", "127.0.0.1"), cfg["net"].get("hud_port", 47130))
    last_body, last_target, last_body_arrival, last_visible_t = None, None, 0.0, 0.0
    set_mirror = None
    lo_ws, hi_ws = cfg["safety"]["workspace_min"], cfg["safety"]["workspace_max"]
    sim = cfg.get("sim", {})
    default_cam = sim.get("views", {}).get(sim.get("default_view", "behind"))
    camera = lambda: getattr(robot, "camera", None) or default_cam
    profile_path = ROOT / cfg["body"].get("profile", "data/profiles/{arm}_pranav.json").format(arm=cfg["arm_profile"])
    session = CalibrationSession(lo_ws, hi_ws, profile_path, cfg["arm_profile"], default_cam)
    bubble = (cfg["net"].get("bubble_host", "127.0.0.1"), cfg["net"].get("bubble_port", 47101))
    if profile_path.exists():
        prof = Profile.load(profile_path)
        retarget = CalibratedMapper(prof, lo_ws, hi_ws, camera)
        log.info("calibration profile loaded: %s (%s, residual %.1f deg, verify %s)", profile_path.name, prof.created,
                 prof.residual_deg, prof.verify)
    else:
        log.info("no calibration profile yet (%s): using %s mode until you calibrate (C)", profile_path.name,
                 cfg["body"].get("mode"))

    def say(text):
        arbiter.last_say = text
        rx.reply(bubble, {"state": "reply", "text": text})   # Pranav's voice bubble on the VM screen
        log.info("calibration: %s", text)

    def handle_session(effects):
        nonlocal retarget
        if not effects:
            return
        if effects.get("say"):
            say(effects["say"])
        if effects.get("release") and clutch.state != "idle":
            on_clutch(clutch._release("calibrating"))
        if effects.get("profile") is not None:
            retarget = CalibratedMapper(effects["profile"], lo_ws, hi_ws, camera)
        if effects.get("engage") and clutch.state == "idle":
            on_clutch(clutch.toggle(time.time()))
    seq_seen = []  # (t, seq) over the last second, for packet loss
    vis_seen = []  # (t, arm visible, in position, hand found) over the last second: tracking quality

    def on_clutch(ev):
        """Clutch events -> arm modes. Calibrating/released freeze the arm; engaged starts following."""
        nonlocal body_target, last_target, retarget
        if ev is None:
            return
        log.info("clutch: %s (trigger=%s%s)", ev.kind, clutch.trigger,
                 f", reason={clutch.release_reason}" if clutch.release_reason else "")
        if ev.kind in ("calibrating", "released"):
            if arbiter.mode == Mode.FOLLOW or arbiter.assist:   # a running grab assist stops too (dead-man)
                arbiter.abort()
            body_target = None
        elif ev.kind == "engaged":
            retarget.engage(arm.fk(q_cmd)[:3, 3], ev.neutral)
            arbiter.on_intent(Intent(action=Action.follow))
            body_target = last_target = None

    def finish(text, addr, intent, latency):
        nonlocal q_cmd, T_des, body_target
        log.info("heard %r -> %s (%.1fs)", text, intent.model_dump(exclude_none=True), latency)
        if intent.action in (Action.stop, Action.reset):  # a slow cloud result must not restart the arm
            for p_text, p_addr, _, fut in pending:
                fut.cancel()
                rx.reply(p_addr, {"kind": "reply", "text": p_text, "say": "Cancelled by stop.", "mode": "hold"})
            pending.clear()
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
        if arbiter.last_say and arbiter.last_say != intent.say:
            log.info("arbiter: %s", arbiter.last_say)
        rx.reply(addr, {"kind": "reply", "text": text, "latency_s": round(latency, 2),
                        "intent": intent.model_dump(mode="json", exclude_none=True),
                        "say": arbiter.last_say or intent.say, "mode": arbiter.mode.value})

    t_end = None if seconds is None else time.time() + seconds
    next_status = next_body_status = 0.0
    body_addr, n_body = None, 0
    while t_end is None or time.time() < t_end:
        t0 = time.time()
        for _, msg, addr in rx.poll():
            if isinstance(msg, BodyPose):
                n_body += 1
                body_addr, last_body, last_body_arrival = addr, msg, t0
                if msg.visible:
                    last_visible_t = t0
                seq_seen.append((t0, msg.seq))
                vis_seen.append((t0, bool(msg.visible), bool(msg.in_position), msg.hand_open is not None))
                rel = np.asarray(msg.wrist) - np.asarray(msg.shoulder) if msg.visible else None
                palm_up_open = bool(msg.hand_open is not None and msg.hand_open > 0.6 and msg.elbow is not None
                                    and msg.wrist[1] < msg.elbow[1] - 0.02)          # MediaPipe y points down
                left_raised = bool(msg.wrist_other is not None and msg.shoulder_other is not None
                                   and msg.wrist_other[1] < msg.shoulder_other[1] - clutch.raise_m)
                if session.state in ("positioning", "capturing"):
                    palm_up_open = left_raised = False   # calibration poses (hand up, open palm) must not engage the arm
                on_clutch(clutch.update(rel, palm_up_open, left_raised, t0))
                if msg.visible and clutch.state == "following" and arbiter.mode == Mode.FOLLOW:
                    tgt = retarget(msg)
                    if tgt is not None:
                        body_target, body_t, last_target = tgt, t0, tgt
                    if grip_from_hand and msg.hand_open is not None:
                        said = arbiter.last_say
                        arbiter.on_hand(msg.hand_open, t0, arm.fk(robot.read_joints())[:3, 3])
                        if arbiter.last_say != said:
                            log.info("arbiter: %s", arbiter.last_say)
                if session.active:
                    handle_session(session.update(t0, msg, bool(msg.in_position),
                                                  tcp_meas=arm.fk(robot.read_joints())[:3, 3],
                                                  camera=camera()))
            elif isinstance(msg, ClutchMsg) and msg.action in ("warmup", "calibrate"):
                handle_session(session.start(t0))
            elif isinstance(msg, ClutchMsg) and msg.action == "redo_axis":
                bad = (session.result or {}).get("bad_axis")
                handle_session(session.start(t0, redo_axis=bad))
            elif isinstance(msg, ClutchMsg):
                on_clutch(clutch.toggle(t0))
                rx.reply(addr, {"kind": "reply", "text": "clutch", "say": f"clutch: {clutch.state}",
                                "mode": arbiter.mode.value})
            elif isinstance(msg, Utterance) and ("calibrat" in msg.text.lower() or
                                                 ("warm" in msg.text.lower() and "up" in msg.text.lower())):
                handle_session(session.start(t0))
                rx.reply(addr, {"kind": "reply", "text": msg.text, "say": "Calibration started.",
                                "mode": arbiter.mode.value})
            elif isinstance(msg, Utterance):
                stop = urgent_stop(msg.text)
                if stop is not None:  # handled in this tick; never queued behind a cloud call
                    finish(msg.text, addr, stop, 0.0)
                elif msg.offline:
                    finish(msg.text, addr, fallback_parse(msg.text, scene.names()), 0.0)
                else:  # Nemotron takes seconds: run it off the control loop
                    pending.append((msg.text, addr, time.time(),
                                    pool.submit(parser.parse, msg.text, scene.names(), descriptions)))
            elif isinstance(msg, EEGState):
                arbiter.on_eeg(msg.p_grasp, t0)
            elif isinstance(msg, Query):
                rx.reply(addr, {"kind": "state", "mode": arbiter.mode.value, "held": arbiter.held,
                                "clutch": clutch.state, "assist": arbiter.assist, "mapping": type(retarget).__name__,
                                "gripper": arbiter.gripper, "tcp": np.round(arm.fk(q_cmd)[:3, 3], 4).tolist(),
                                "tcp_meas": np.round(arm.fk(robot.read_joints())[:3, 3], 4).tolist(),
                                "q_meas": np.round(robot.read_joints(), 4).tolist(),
                                "plan_left": len(arbiter._plan), "table_top": scene.support_z,
                                "objects": {n: {"pos": np.round(o.pos, 4).tolist(),
                                                "bottom_z": round(float(o.pos[2] - o.grasp_z), 4)}
                                            for n, o in scene.objects.items()}})
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
        said = arbiter.last_say
        p, grip = arbiter.target(tcp, body_target if t0 - body_t < 0.5 else None)
        if arbiter.last_say != said:
            log.info("arbiter: %s", arbiter.last_say)
        while arbiter.events:
            if arbiter.events.pop(0) == "assist_done" and clutch.state == "following":
                # hand control back without a jump: position mode re-anchors at the arm, calibrated mode glides
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
        q_des = arm.ik_step(q_cmd, T_des)
        q_cmd = safety.step(q_cmd, q_des, dt, input_age)
        q_cmd = np.clip(q_cmd, q - WINDUP, q + WINDUP)  # anti-windup: never run far ahead of the real joints
        # sync cues: ghost at the target. ok = green, clamped = amber, lost = red, calib = blue
        ghost, ghost_state = None, None
        if clutch.state == "calibrating":
            ghost, ghost_state = arm.fk(q_cmd)[:3, 3], "calib"
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
        vtarget = session.vtarget() or (np.round(cand.pos, 4).tolist() if cand is not None else None)
        robot.command(q_cmd, grip, ghost=ghost, ghost_bad=ghost_bad, ghost_state=ghost_state, vtarget=vtarget)
        if t0 >= next_body_status:  # 10 Hz: status to the body client (echo for RTT) + HUD overlay on the VM screen
            next_body_status = t0 + 0.1
            seq_seen[:] = [x for x in seq_seen if t0 - x[0] <= 1.0]
            vis_seen[:] = [x for x in vis_seen if t0 - x[0] <= 1.0]
            pct = lambda i: round(100 * sum(x[i] for x in vis_seen) / len(vis_seen)) if vis_seen else None
            span = (seq_seen[-1][1] - seq_seen[0][1] + 1) if len(seq_seen) > 1 else len(seq_seen)
            loss = 1.0 - len(seq_seen) / span if span > 0 else 0.0
            st = {"kind": "status", "mode": arbiter.mode.value, "tcp": np.round(tcp, 3).tolist(),
                  "target": None if ghost is None else np.round(ghost, 3).tolist(), "ghost": ghost_state,
                  "clipped": ghost_bad, "grip": grip, "clutch": clutch.state, "trigger": clutch.trigger, "release_reason": clutch.release_reason,
                  "clutch_progress": round(clutch.progress, 2), "ws_min": safety.ws_min.tolist(), "ws_max": safety.ws_max.tolist(),
                  "echo_seq": last_body.seq if last_body else None, "echo_t": last_body.t if last_body else None,
                  "echo_age": round(t0 - last_body_arrival, 4) if last_body else None,
                  "pkt_s": len(seq_seen), "loss": round(max(0.0, loss), 3),
                  "vis_pct": pct(1), "inpos_pct": pct(2), "hand_pct": pct(3),
                  "calib": session.view(), "mapping": type(retarget).__name__, "set_mirror": set_mirror,
                  "vtarget": vtarget, "grab_ready": cand.name if cand is not None else None,
                  "assist": arbiter.assist, "held": arbiter.held, "tcp_meas": np.round(tcp, 3).tolist()}
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
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.arm)
    log.info("arm profile: %s (%s)", cfg["arm_profile"], cfg["arm"]["urdf"])
    run(cfg, args.backend or cfg["loop"]["backend"], args.seconds)


if __name__ == "__main__":
    main()

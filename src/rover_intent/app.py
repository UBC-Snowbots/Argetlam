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
from .types import Action, BodyPose, EEGState, Intent, Utterance

log = logging.getLogger("rover_intent")
ROOT = Path(__file__).resolve().parents[2]
WINDUP = 0.25  # rad: max gap between the commanded and the measured joint positions


def build(cfg: dict, backend: str):
    a = cfg["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), tip=a["tip_link"], tcp_offset=a["tcp_offset"])
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
    scene = Scene(cfg["scene"])
    e, s = cfg["eeg"], cfg["assist"]
    arbiter = Arbiter(scene, e["p_on"], e["p_off"], e["hold_s"], s["radius"], s["gain"])
    b = cfg["body"]
    oe = b.get("one_euro", {})
    retarget = Retargeter(b["axes"], b["scale"], b["origin"], min_cutoff=oe.get("min_cutoff", 1.0),
                          beta=oe.get("beta", 0.7))
    if b.get("mode", "cartesian") == "gesture":
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

    def finish(text, addr, intent, latency):
        nonlocal q_cmd, T_des, body_target
        log.info("heard %r -> %s (%.1fs)", text, intent.model_dump(exclude_none=True), latency)
        if intent.action in (Action.stop, Action.reset):  # a slow cloud result must not restart the arm
            for p_text, p_addr, _, fut in pending:
                fut.cancel()
                rx.reply(p_addr, {"kind": "reply", "text": p_text, "say": "Cancelled by stop.", "mode": "hold"})
            pending.clear()
        arbiter.on_intent(intent)
        if intent.action == Action.follow:  # clutch in: the arm stays put, your next wrist pose is neutral
            retarget.engage(arm.fk(q_cmd)[:3, 3])  # Retargeter or GestureMirror: both clutch at the current TCP
            body_target = None
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
                body_addr = addr
            if isinstance(msg, BodyPose) and msg.visible:
                if arbiter.mode == Mode.FOLLOW:
                    tgt = retarget(msg)
                    if tgt is not None:
                        body_target, body_t = tgt, t0
                    if grip_from_hand and msg.hand_open is not None:
                        arbiter.on_hand(msg.hand_open)
            elif isinstance(msg, Utterance):
                stop = urgent_stop(msg.text)
                if stop is not None:  # handled in this tick; never queued behind a cloud call
                    finish(msg.text, addr, stop, 0.0)
                else:  # Nemotron takes seconds: run it off the control loop
                    pending.append((msg.text, addr, time.time(),
                                    pool.submit(parser.parse, msg.text, scene.names(), descriptions)))
            elif isinstance(msg, EEGState):
                arbiter.on_eeg(msg.p_grasp, t0)
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
        if p is not None:
            T_des[:3, 3] = safety.clip_workspace(p)
        else:  # HOLD = stop here: aim at the current commanded pose, the safety layer brakes at max decel
            T_des = arm.fk(q_cmd)
        input_age = 0.0 if arbiter.mode == Mode.AUTO or p is None else t0 - body_t
        q_des = arm.ik_step(q_cmd, T_des)
        q_cmd = safety.step(q_cmd, q_des, dt, input_age)
        q_cmd = np.clip(q_cmd, q - WINDUP, q + WINDUP)  # anti-windup: never run far ahead of the real joints
        ghost_bad = bool(p is not None and (getattr(retarget, "clipped", False) and arbiter.mode == Mode.FOLLOW
                                            or not np.allclose(safety.clip_workspace(p), p)))
        robot.command(q_cmd, grip, ghost=None if p is None else T_des[:3, 3], ghost_bad=ghost_bad)
        if body_addr is not None and t0 >= next_body_status:  # live feedback for the body client's mini-map
            next_body_status = t0 + 0.1
            tgt = None if p is None else np.asarray(p, float)
            rx.reply(body_addr, {"kind": "status", "mode": arbiter.mode.value, "tcp": np.round(tcp, 3).tolist(),
                                 "target": None if tgt is None else np.round(tgt, 3).tolist(),
                                 "clipped": ghost_bad,
                                 "grip": grip, "ws_min": safety.ws_min.tolist(), "ws_max": safety.ws_max.tolist()})
        if t0 >= next_status:
            next_status = t0 + 1.0
            log.info("mode=%s tcp=%s grip=%.0f body=%d pkt/s%s%s", arbiter.mode.value, np.round(tcp, 3), grip, n_body,
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
    ap.add_argument("--backend", default=None, choices=["mock", "isaac", "real"])
    ap.add_argument("--seconds", type=float, default=None)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.backend or cfg["loop"]["backend"], args.seconds)


if __name__ == "__main__":
    main()

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
from pathlib import Path

import numpy as np
import yaml

from .body.retarget import Retargeter
from .control.arbiter import Arbiter, Mode
from .control.kinematics import ArmModel
from .control.safety import SafetyLayer
from .planner.nemotron import IntentParser
from .planner.scene import Scene
from .robot.backends import IsaacArm, MockArm, RealArm
from .transport.udp import Receiver
from .types import BodyPose, EEGState, Utterance

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
    retarget = Retargeter(b["axes"], b["scale"], b["origin"], b["alpha"])
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
    body_target, body_t = None, 0.0
    q_cmd = robot.read_joints()  # commanded trajectory state (the drives track it; see WINDUP)
    T_des = arm.fk(q_cmd)
    t_end = None if seconds is None else time.time() + seconds
    next_status = 0.0
    while t_end is None or time.time() < t_end:
        t0 = time.time()
        for _, msg in rx.poll():
            if isinstance(msg, BodyPose) and msg.visible:
                body_target, body_t = retarget(msg), t0
            elif isinstance(msg, Utterance):
                intent = parser.parse(msg.text, scene.names(), descriptions)
                log.info("heard %r -> %s", msg.text, intent.model_dump(exclude_none=True))
                arbiter.on_intent(intent)
                if arbiter.last_say and arbiter.last_say != intent.say:
                    log.info("arbiter: %s", arbiter.last_say)
            elif isinstance(msg, EEGState):
                arbiter.on_eeg(msg.p_grasp, t0)
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
        input_age = 0.0 if arbiter.mode == Mode.AUTO or p is None else t0 - body_t
        q_des = arm.ik_step(q_cmd, T_des)
        q_cmd = safety.step(q_cmd, q_des, dt, input_age)
        q_cmd = np.clip(q_cmd, q - WINDUP, q + WINDUP)  # anti-windup: never run far ahead of the real joints
        robot.command(q_cmd, grip)
        if t0 >= next_status:
            next_status = t0 + 1.0
            log.info("mode=%s tcp=%s grip=%.0f%s", arbiter.mode.value, np.round(tcp, 3), grip,
                     " ESTOP:" + safety.reason if safety.estopped else "")
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

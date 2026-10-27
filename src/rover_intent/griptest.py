"""Shared helpers for the gripper replay / scenario harnesses (scripts/replay_session.py, scripts/grip_scenarios.py).

Nothing here touches the live system. It can:
  * write a spare-port config and launch a MOCK (or Isaac) app, tracked by PID so the harness only ever kills what
    it started;
  * poll the running app's Query interface at a MODEST rate (>= 0.2 s between polls, a blocking recv, never a tight
    loop) and keep a time series of clutch / mode / gripper (+ source) / TCP / measured gripper;
  * parse the app's own log for the exact gripper transitions (each carries its source), grip-toggle receipts,
    clash count and any errors - the authoritative record, since the app logs every change on the tick it happens;
  * synthesize BodyPose streams that engage the clutch through the normal palm-up path and then drive teleop motion.

NO GPU and NO Isaac here; the Isaac finger check is done by the caller when --backend isaac is used on the live stack.
"""
from __future__ import annotations

import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

# skel2d order used by body/plane.py: L_SH R_SH L_EL R_EL L_WR R_WR L_HIP R_HIP NOSE
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, NOSE = range(9)

# Spare ports for the mock app. NEVER the live ports (47100/47101/47120/47130/47131/47150/47152/47195).
SPARE_PORTS = {"listen_port": 47265, "hud_port": 47266, "hud_video_port": 47267, "bubble_port": 47268,
               "isaac_cmd_port": 47269, "isaac_state_port": 47270}


# --- spare-port mock app ------------------------------------------------------------------------------------------
def write_spare_config(src_config: str, out_path: Path, ports: dict | None = None) -> Path:
    """Copy a config and rewrite the net ports to spare ones, forcing the mock backend off the live ports."""
    ports = ports or SPARE_PORTS
    cfg = yaml.safe_load(Path(src_config).read_text())
    net = cfg.setdefault("net", {})
    net.update(ports)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


class AppProcess:
    """A mock (or Isaac) app we start and stop ourselves. Logs go to a file we can scan for errors + transitions."""

    def __init__(self, config: str, backend: str = "mock", log_path: Path | None = None, arm: str | None = None,
                 venv_python: str | None = None, record: bool = False):
        self.config, self.backend, self.arm, self.record = config, backend, arm, record
        self.log_path = Path(log_path) if log_path else ROOT / "logs" / f"griptest_app_{int(time.time())}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        import sys
        self.python = venv_python or sys.executable   # same interpreter as the harness (the project venv)
        self.proc: subprocess.Popen | None = None
        self._logf = None

    def start(self) -> "AppProcess":
        cmd = [self.python, "-m", "rover_intent.app", "--backend", self.backend, "--config", self.config]
        if not self.record:
            cmd.append("--no-record")
        if self.arm:
            cmd += ["--arm", self.arm]
        env = {"PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "4", "SDL_VIDEODRIVER": "dummy", "PATH":
               "/usr/bin:/bin"}
        import os
        env = {**os.environ, **env}
        self._logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self._logf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        return self

    def wait_ready(self, port: int, host: str = "127.0.0.1", timeout: float = 20.0) -> bool:
        """Ready when a Query gets a state reply. Uses a blocking recv with a short timeout, polled >= 0.2 s apart."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                if self.proc is not None and self.proc.poll() is not None:
                    return False
                try:
                    s.sendto(json.dumps({"kind": "query", "t": time.time()}).encode(), (host, port))
                    data, _ = s.recvfrom(65536)
                    if json.loads(data).get("kind") == "state":
                        return True
                except (socket.timeout, OSError):
                    pass
                time.sleep(0.25)
            return False
        finally:
            s.close()

    def stop(self) -> int | None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5.0)
        if self._logf is not None:
            self._logf.flush()
            self._logf.close()
            self._logf = None
        return None if self.proc is None else self.proc.returncode


# --- Query polling (modest rate, never a tight loop) --------------------------------------------------------------
class AppMonitor(threading.Thread):
    """Polls the app's Query at `hz` (clamped to <= 5 Hz, so >= 0.2 s between polls). Each poll is a blocking recv
    with a short timeout - a bounded wait, not a spin. Keeps a time series for the summary."""

    def __init__(self, host: str, port: int, hz: float = 4.0):
        super().__init__(daemon=True)
        self.addr = (host, port)
        self.interval = max(0.2, 1.0 / max(hz, 0.1))
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.15)
        try:
            while not self._stop.is_set():
                t = time.time()
                try:
                    s.sendto(json.dumps({"kind": "query", "t": t}).encode(), self.addr)
                    data, _ = s.recvfrom(65536)
                    st = json.loads(data)
                    if st.get("kind") == "state":
                        st["_t"] = time.time()
                        self.samples.append(st)
                except (socket.timeout, OSError):
                    pass
                self._stop.wait(self.interval)
        finally:
            s.close()

    def stop(self):
        self._stop.set()

    # summary pieces from the polled series
    def tcp_path_following(self) -> float:
        """Total TCP path length (m) over consecutive samples where the clutch was following."""
        d = 0.0
        prev = None
        for st in self.samples:
            tcp = st.get("tcp")
            foll = st.get("clutch") == "following"
            if prev is not None and foll and prev[1] and tcp is not None and prev[0] is not None:
                d += float(np.linalg.norm(np.asarray(tcp) - np.asarray(prev[0])))
            prev = (tcp, foll)
        return d

    def followed(self) -> bool:
        return any(st.get("clutch") == "following" for st in self.samples)

    def max_clashes(self) -> int:
        return max((int(st.get("grip_clashes") or 0) for st in self.samples), default=0)


# --- app-log parsing (authoritative gripper record) ---------------------------------------------------------------
_RE_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d)")
_RE_GRIP = re.compile(r"gripper (CLOSE|OPEN) \([\d.]+ -> [\d.]+\) in (\w+) via (\w+|\?)")
_RE_TOGGLE = re.compile(r"grip toggle \((\S+)\) in (\w+): gripper=([\d.]+) src=(\w+|None) clashes=(\d+)")
_RE_CLASH = re.compile(r"Grip clash: (\w+) vs (\w+); (\w+) wins \((\w+)\)")
_RE_ERR = re.compile(r"(Traceback \(most recent call last\)|\bERROR\b|\bCRITICAL\b)")


def _ts(line: str) -> float | None:
    m = _RE_TS.match(line)
    if not m:
        return None
    return time.mktime(time.strptime(m.group(1)[:19], "%Y-%m-%d %H:%M:%S")) + float(m.group(1)[20:]) / 1000.0


def parse_app_log(text: str) -> dict:
    """Pull the exact gripper record out of an app log: transitions (t, dir, source), grip-toggle receipts, the
    final clash count and any error lines. The app logs every gripper change on the tick it happens, so this is the
    ground truth the polled series only samples."""
    transitions, toggles, clashes, errors = [], [], [], []
    for line in text.splitlines():
        t = _ts(line)
        mg = _RE_GRIP.search(line)
        if mg:
            transitions.append({"t": t, "dir": mg.group(1).lower(), "mode": mg.group(2), "source": mg.group(3)})
        mt = _RE_TOGGLE.search(line)
        if mt:
            toggles.append({"t": t, "msg_source": mt.group(1), "mode": mt.group(2),
                            "gripper": float(mt.group(3)), "src": mt.group(4), "clashes": int(mt.group(5))})
        mc = _RE_CLASH.search(line)
        if mc:
            clashes.append({"t": t, "earlier": mc.group(1), "winner": mc.group(3), "dir": mc.group(4)})
        if _RE_ERR.search(line):
            errors.append(line.strip())
    return {"transitions": transitions, "toggles": toggles, "clashes": clashes, "errors": errors}


# --- synthetic BodyPose streams ----------------------------------------------------------------------------------
# A fixed, plausible 2D skeleton (normalized image coords, y down). Operator's RIGHT side sits at the SMALLER x, as
# body/plane.py documents. The control (right) wrist is moved to drive lateral/vertical TCP motion.
_SKEL0 = {
    L_SH: (0.74, 0.68), R_SH: (0.41, 0.68), L_EL: (0.80, 0.95), R_EL: (0.35, 0.95),
    L_WR: (0.83, 0.80), R_WR: (0.47, 0.80), L_HIP: (0.70, 1.18), R_HIP: (0.46, 1.18), NOSE: (0.58, 0.45),
}
# fixed 3D points (metres, MediaPipe-ish, y DOWN): wrist held ABOVE the elbow so the open palm counts for engage,
# and wrist - shoulder is constant so calibration always reads "still".
_W3D, _S3D, _E3D = (0.0, -0.30, 0.0), (0.0, 0.0, 0.0), (0.0, -0.10, 0.0)


def make_body(t: float, seq: int, ctrl_dx: float = 0.0, ctrl_dy: float = 0.0, reach_dy: float = 0.0,
              hand_open: float | None = 0.9, visible: bool = True) -> dict:
    """One BodyPose dict. ctrl_dx/ctrl_dy move the right (control) wrist in skel2d (shoulder-widths-ish); reach_dy
    moves the left (reach) wrist. hand_open drives the hand gripper (None = hand not found, so no hand edges)."""
    sk = {k: list(v) for k, v in _SKEL0.items()}
    sk[R_WR][0] += ctrl_dx
    sk[R_WR][1] += ctrl_dy
    sk[L_WR][1] += reach_dy
    skel2d = [tuple(sk[i]) for i in range(9)]
    return {"kind": "body", "t": t, "wrist": _W3D, "shoulder": _S3D, "elbow": _E3D,
            "shoulder_other": (0.33, 0.0, 0.0), "hips": (0.0, 0.5, 0.0), "wrist_other": (0.33, -0.20, 0.0),
            "side": "right", "skel2d": skel2d, "seq": seq, "hand_open": hand_open, "visible": bool(visible)}


def engage_frames(start: float, hz: float = 20.0, palm_s: float = 1.2, calib_s: float = 2.3,
                  seq0: int = 1) -> tuple[list[tuple[float, dict]], float, int]:
    """Palm-up-open held, then a still hold, so the clutch goes idle -> calibrating -> following through the normal
    path. Returns (rows, end_time, next_seq); rows are (send_time, msg)."""
    rows, seq, t, dt = [], seq0, start, 1.0 / hz
    end = start + palm_s + calib_s
    while t <= end:
        rows.append((t, make_body(t, seq, hand_open=0.9)))   # open palm up, body still
        seq += 1
        t += dt
    return rows, t, seq

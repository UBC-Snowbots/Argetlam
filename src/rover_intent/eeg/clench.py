"""Muse 2 double jaw clench -> GripToggle (close if open, open if closed).

This is a MUSCLE (EMG) signal picked up by the headband, not decoded brain activity: clenching the jaw fires the
temporalis/masseter muscles, which swamp TP9/TP10 (behind the ears) and AF7/AF8 (forehead) with broadband
30-100+ Hz noise. The Muse has no motor-cortex electrodes, so imagined-grasp decoding is not realistic on it
(see docs/LOG.md 2026-10-10); this is the reliable hands-free input it does give us.

Pipeline (same code for calibration, live and replay):
  per channel: 60 Hz notch -> 30-100 Hz band-pass -> RMS envelope (30 ms)
  score = median over channels of envelope / that channel's relaxed level   (one loose electrode can't trigger it)
  clench = score above on_ratio for >= 60 ms, ends below 0.75 x on_ratio for >= 40 ms; longer than 1.5 s is ignored
  TOGGLE = exactly two clenches, second starting <= 0.8 s after the first ends, then 0.5 s with no third
           (a third = chewing: discard, and wait for 1 s of quiet)

Standalone: this one file runs on the laptop (pip install numpy scipy brainflow), like laptop/body_client.py.
Usage (laptop with the Muse; Bluetooth must be on that machine; `python clench.py ...` works too):
  python -m rover_intent.eeg.clench calibrate --out data/profiles/clench_pranav.json      # ~1.5 min, guided
  python -m rover_intent.eeg.clench run --profile data/profiles/clench_pranav.json --host <vm>
  python -m rover_intent.eeg.clench replay --profile ... --file data/eeg_recordings/clench_*.npz
  --board muse2 (native BLE, default) | muse2-bled (BLED112 dongle, --serial COM3) | synthetic (pipeline smoke only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
try:
    from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
except ImportError:  # e.g. Windows Smart App Control blocks SciPy's DLLs on the laptop: numpy-only stand-ins
    try:
        from rover_intent.eeg.np_signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
    except ImportError:  # this file copied on its own: keep np_signal.py next to it
        from np_signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


@dataclass
class ClenchConfig:
    band: tuple[float, float] = (30.0, 100.0)
    notch_hz: float = 60.0  # mains (Canada); 50 elsewhere
    env_s: float = 0.03
    on_ratio: float = 8.0  # set by calibration
    off_frac: float = 0.75  # off threshold = max(on_ratio * off_frac, 1.5)
    min_on_s: float = 0.06
    min_off_s: float = 0.04
    max_clench_s: float = 1.5
    max_gap_s: float = 0.8
    confirm_s: float = 0.5
    quiet_s: float = 1.0
    refractory_s: float = 0.5

    @property
    def off_ratio(self) -> float:
        return max(self.on_ratio * self.off_frac, 1.5)


class EmgFrontEnd:
    """Causal, chunk-size-independent filter + envelope. push((n_ch, n) raw) -> (n_ch, n) envelope."""

    def __init__(self, fs: float, n_ch: int, cfg: ClenchConfig):
        hi = min(cfg.band[1], 0.45 * fs)
        bp = butter(4, [cfg.band[0], hi], btype="band", fs=fs, output="sos")
        sos = bp
        if cfg.notch_hz and cfg.notch_hz < fs / 2:
            sos = np.vstack([tf2sos(*iirnotch(cfg.notch_hz, 30.0, fs=fs)), bp])
        self.sos = sos
        self.zi = np.repeat(sosfilt_zi(sos)[:, None, :], n_ch, axis=1)  # (sections, n_ch, 2)
        self.zi *= 0.0
        self.alpha = 1.0 - np.exp(-1.0 / (cfg.env_s * fs))
        self.power = np.zeros(n_ch)

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(x, float))
        y, self.zi = sosfilt(self.sos, x, axis=1, zi=self.zi)
        env = np.empty_like(y)
        p, a = self.power, self.alpha
        for i in range(y.shape[1]):  # one-pole smoother of y^2
            p = p + a * (y[:, i] ** 2 - p)
            env[:, i] = p
        self.power = p
        return np.sqrt(env)


@dataclass
class ClenchDetector:
    fs: float
    rest_level: list[float]  # per-channel relaxed envelope (calibration)
    cfg: ClenchConfig = field(default_factory=ClenchConfig)
    t0: float = 0.0

    def __post_init__(self):
        self.front = EmgFrontEnd(self.fs, len(self.rest_level), self.cfg)
        self.rest = np.maximum(np.asarray(self.rest_level, float), 1e-9)
        self.n = 0
        self.score = 0.0
        self._above_since = self._below_since = None
        self._on_start = None  # start time of the clench in progress
        self._long = False
        self._clenches: list[tuple[float, float]] = []
        self._quiet_until = -np.inf
        self._refractory_until = -np.inf
        self.log: list[tuple[str, float]] = []  # ("clench", t) / ("chew", t) / ("long", t) / ("toggle", t)

    @classmethod
    def from_profile(cls, profile: dict, t0: float = 0.0) -> "ClenchDetector":
        cfg = ClenchConfig(**{k: (tuple(v) if k == "band" else v) for k, v in profile["config"].items()})
        return cls(profile["fs"], profile["rest_level"], cfg, t0)

    def scores(self, x: np.ndarray) -> np.ndarray:
        return np.median(self.front.push(x) / self.rest[:, None], axis=0)

    def push(self, x: np.ndarray) -> list[float]:
        """Feed raw samples (n_ch, n). Returns the times of any toggles that completed inside this chunk."""
        c, out = self.cfg, []
        s_all = self.scores(x)
        for s in s_all:
            t = self.t0 + self.n / self.fs
            self.n += 1
            self.score = float(s)
            if self._on_start is None:  # waiting for an onset
                if s > c.on_ratio:
                    self._above_since = t if self._above_since is None else self._above_since
                    if t - self._above_since >= c.min_on_s:
                        self._on_start, self._below_since = self._above_since, None
                        if t < self._quiet_until:  # still chewing: push the quiet window out
                            self._quiet_until = t + c.quiet_s
                else:
                    self._above_since = None
            else:  # clench in progress: wait for the release
                if s < c.off_ratio:
                    self._below_since = t if self._below_since is None else self._below_since
                    if t - self._below_since >= c.min_off_s:
                        self._end_clench(self._on_start, self._below_since)
                        self._on_start, self._above_since, self._long = None, None, False
                else:
                    self._below_since = None
                    if t - self._on_start > c.max_clench_s and not self._long:
                        self._long = True  # sustained clench (or a pressed jaw): never part of a toggle
                        self.log.append(("long", t))
                        self._clenches = []
            toggled = self._check_sequence(t)
            if toggled is not None:
                out.append(toggled)
        return out

    def _end_clench(self, start: float, end: float) -> None:
        c = self.cfg
        if end - start > c.max_clench_s:
            self._clenches = []
            return
        if start < self._quiet_until:  # still inside a chewing episode: keep waiting for real quiet
            self._quiet_until = max(self._quiet_until, end + c.quiet_s)
            return
        if start < self._refractory_until:
            return
        self.log.append(("clench", start))
        if self._clenches and start - self._clenches[-1][1] > c.max_gap_s:
            self._clenches = []  # previous single clench timed out
        self._clenches.append((start, end))
        if len(self._clenches) >= 3:  # rhythmic: chewing / talking
            self.log.append(("chew", start))
            self._clenches = []
            self._quiet_until = end + c.quiet_s

    def _check_sequence(self, t: float) -> float | None:
        c = self.cfg
        if not self._clenches:
            return None
        if self._on_start is not None:  # a further clench is in progress: let _end_clench decide
            return None
        last_end = self._clenches[-1][1]
        if len(self._clenches) == 1:
            if t - last_end > c.max_gap_s:
                self._clenches = []  # single clench: ignore
            return None
        if t - last_end >= c.confirm_s:  # exactly two, then quiet
            self._clenches = []
            self._refractory_until = t + c.refractory_s
            self.log.append(("toggle", t))
            return t
        return None


# --- calibration ----------------------------------------------------------------------------------------------------
def calibrate(rest: np.ndarray, clench: np.ndarray, other: np.ndarray | None, fs: float,
              cfg: ClenchConfig | None = None) -> dict:
    """rest/clench/other: (n_ch, n) raw recordings. Returns a profile dict (json-able) with a quality verdict."""
    cfg = cfg or ClenchConfig()
    n_ch = rest.shape[0]
    settle = int(1.0 * fs)  # skip the filter start-up transient

    def env(x):
        return EmgFrontEnd(fs, n_ch, cfg).push(x)[:, settle:]

    rest_env = env(rest)
    rest_level = np.median(rest_env, axis=1)
    score = lambda x: np.median(env(x) / rest_level[:, None], axis=0)  # noqa: E731
    rest_hi = float(np.percentile(score(rest), 99))
    clench_s = score(clench)
    clench_hi = float(np.percentile(clench_s, 95))
    other_hi = float(np.percentile(score(other), 99.5)) if other is not None else 0.0
    profile = {"fs": fs, "channels": MUSE_CHANNELS[:n_ch] if n_ch <= 4 else list(range(n_ch)),
               "rest_level": [float(v) for v in rest_level], "config": asdict(cfg)}

    # dry-run the full detector over a segment: (toggles, clenches seen)
    def toggles(x):
        d = ClenchDetector.from_profile(profile)
        n = sum(len(d.push(x[:, i:i + 32])) for i in range(0, x.shape[1], 32))
        return n, sum(1 for k, _ in d.log if k == "clench")

    # Threshold: geometric middle of relaxed and clench levels. Chewing is as strong as clenching, so it is rejected
    # by rhythm (3+ clenches = chewing), not amplitude; only raise the threshold if talking/chewing still produces
    # false toggles, and never above 1/1.15 of the clench level.
    on = max(np.sqrt(max(rest_hi, 1.0) * clench_hi), 3.0)
    profile["config"]["on_ratio"] = round(float(on), 3)
    while other is not None and toggles(other)[0] and on * 1.1 <= clench_hi / 1.15:
        on *= 1.1
        profile["config"]["on_ratio"] = round(float(on), 3)
    cfg.on_ratio = profile["config"]["on_ratio"]

    stats = {"rest_p99": round(rest_hi, 2), "clench_p95": round(clench_hi, 2), "other_p995": round(other_hi, 2),
             "on_ratio": cfg.on_ratio, "margin": round(clench_hi / cfg.on_ratio, 2),
             "rest_toggles": toggles(rest)[0], "clench_detected": toggles(clench)[1],
             "other_toggles": toggles(other)[0] if other is not None else None}
    good = stats["margin"] >= 1.5 and stats["rest_toggles"] == 0 and not stats["other_toggles"]
    ok = stats["margin"] >= 1.15 and stats["rest_toggles"] == 0
    stats["quality"] = "good" if good else "marginal" if ok else "poor"
    profile["stats"] = stats
    return profile


# --- board I/O (BrainFlow) -------------------------------------------------------------------------------------------
def open_board(kind: str, serial: str = "", mac: str = "", name: str = ""):
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

    ids = {"muse2": BoardIds.MUSE_2_BOARD, "muse2-bled": BoardIds.MUSE_2_BLED_BOARD,
           "synthetic": BoardIds.SYNTHETIC_BOARD}
    board_id = ids[kind]
    params = BrainFlowInputParams()
    params.serial_port, params.mac_address, params.serial_number = serial, mac, name
    board = BoardShim(board_id, params)
    fs = BoardShim.get_sampling_rate(board_id)
    chans = BoardShim.get_eeg_channels(board_id)[:4]
    board.prepare_session()
    board.start_stream()
    return board, fs, chans


def _read(board, chans) -> np.ndarray:
    data = board.get_board_data()
    return data[chans] if data.shape[1] else np.zeros((len(chans), 0))


def _record(board, chans, seconds: float, prompt=None) -> np.ndarray:
    chunks, t_end = [], time.time() + seconds
    while time.time() < t_end:
        if prompt:
            prompt(t_end - time.time())
        time.sleep(0.05)
        chunks.append(_read(board, chans))
    chunks.append(_read(board, chans))
    return np.concatenate(chunks, axis=1)


def _say(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def cmd_calibrate(args) -> int:
    board, fs, chans = open_board(args.board, args.serial, args.mac, args.name)
    try:
        _say("Getting a clean signal: sit still, relax your face. Starting in 5 s.")
        time.sleep(5)
        _read(board, chans)  # drop the start-up junk
        _say(f"RELAX for {args.rest_s:.0f} s: jaw loose, eyes open, blink normally.")
        rest = _record(board, chans, args.rest_s)
        _say("Now CLENCH your jaw firmly for about half a second each time you see CLENCH, then relax.")
        time.sleep(2)
        parts = []
        for i in range(args.n_clench):
            print(f"  [{i + 1}/{args.n_clench}] CLENCH", flush=True)
            parts.append(_record(board, chans, 3.0))
        clench = np.concatenate(parts, axis=1)
        _say(f"Now TALK and CHEW normally for {args.other_s:.0f} s (read something aloud, chew gum).")
        other = _record(board, chans, args.other_s)
    finally:
        board.stop_stream()
        board.release_session()
    profile = calibrate(rest, clench, other, fs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1))
    rec = Path(args.rec_dir) / f"clench_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    rec.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rec, rest=rest, clench=clench, other=other, fs=fs)
    s = profile["stats"]
    _say(f"Calibration: {s['quality'].upper()}  (threshold {s['on_ratio']}x relaxed, clench peak "
         f"{s['clench_p95']}x, margin {s['margin']}x; clenches seen {s['clench_detected']}/{args.n_clench}; "
         f"false toggles while relaxed {s['rest_toggles']}, while talking/chewing {s['other_toggles']})")
    print(f"profile -> {out}\nraw recording -> {rec}")
    if s["quality"] == "poor":
        print("Poor separation: wet the sensors / push the band down onto the skin behind the ears, and redo it.")
    return 0 if s["quality"] != "poor" else 1


def toggle_packet(t: float) -> bytes:
    """The GripToggle message (rover_intent.types) as raw JSON, so this file also runs standalone on the laptop."""
    return json.dumps({"kind": "grip_toggle", "t": t, "source": "clench"}).encode()


def cmd_run(args) -> int:
    import socket

    profile = json.loads(Path(args.profile).read_text())
    board, fs, chans = open_board(args.board, args.serial, args.mac, args.name)
    if abs(fs - profile["fs"]) > 1:
        print(f"warning: board runs at {fs} Hz but the profile was made at {profile['fs']} Hz", file=sys.stderr)
    det = ClenchDetector.from_profile(profile, t0=time.time())
    tx = None if args.dry else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"double-clench to toggle the gripper. threshold {det.cfg.on_ratio}x relaxed. Ctrl+C to stop.")
    n_toggles, last_print = 0, 0.0
    try:
        while True:
            x = _read(board, chans)
            if x.shape[1]:
                for t in det.push(x):
                    n_toggles += 1
                    if tx:
                        tx.sendto(toggle_packet(time.time()), (args.host, args.port))
                    print(f"\nTOGGLE #{n_toggles}" + (" (sent)" if tx else " (dry run)"), flush=True)
            if time.time() - last_print > 0.1:
                bar = "#" * min(40, int(40 * det.score / (2 * det.cfg.on_ratio)))
                state = "CLENCH" if det._on_start is not None else ""
                print(f"\r{det.score:6.1f}x |{bar:<40}| {state:<6}", end="", flush=True)
                last_print = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        board.stop_stream()
        board.release_session()
    return 0


def cmd_replay(args) -> int:
    profile = json.loads(Path(args.profile).read_text())
    rec = np.load(args.file)
    for key in [k for k in ("rest", "clench", "other", "data") if k in rec.files]:
        det = ClenchDetector.from_profile(profile)
        x = rec[key]
        toggles = [round(t, 2) for i in range(0, x.shape[1], 32) for t in det.push(x[:, i:i + 32])]
        clenches = sum(1 for k, _ in det.log if k == "clench")
        print(json.dumps({"segment": key, "seconds": round(x.shape[1] / profile["fs"], 1),
                          "clenches": clenches, "toggles_at_s": toggles}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--board", choices=["muse2", "muse2-bled", "synthetic"], default="muse2")
        p.add_argument("--serial", default="", help="BLED112 dongle port for muse2-bled (COM3, /dev/ttyACM0)")
        p.add_argument("--mac", default="", help="Muse MAC address (optional)")
        p.add_argument("--name", default="", help="Muse name, e.g. Muse-1A2B (optional; picks that headband)")
    c = sub.choices["calibrate"]
    c.add_argument("--out", required=True, help="profile json to write")
    c.add_argument("--rec-dir", default="data/eeg_recordings")
    c.add_argument("--rest-s", type=float, default=20.0)
    c.add_argument("--n-clench", type=int, default=8)
    c.add_argument("--other-s", type=float, default=20.0)
    r = sub.choices["run"]
    r.add_argument("--profile", required=True)
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=47100)
    r.add_argument("--dry", action="store_true", help="print toggles only, don't send")
    p = sub.add_parser("replay")
    p.add_argument("--profile", required=True)
    p.add_argument("--file", required=True)
    args = ap.parse_args(argv)
    return {"calibrate": cmd_calibrate, "run": cmd_run, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

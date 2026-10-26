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
import os
import sys
import time
from collections import deque
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
    # Moving baseline: detect a clench as a spike relative to the CURRENT level, not a FIXED relaxed level. The
    # baseline is a robust (running-median) estimate of the recent score, so during body teleop (arms raised, head
    # moving) the score floor is tracked out and a clench both exceeds on_ratio AND releases below off_ratio again.
    # "fixed" reproduces the original behaviour exactly (baseline == 1.0). At true rest the median sits at ~1.0, so
    # "moving" reduces to "fixed" there and loses no detections.
    baseline_mode: str = "moving"  # "fixed" | "moving"
    baseline_s: float = 4.0        # window of the running-median baseline (moving mode)

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
        self.score = 0.0          # raw score: median over channels of envelope / calibrated relaxed level
        self.baseline = 1.0       # moving-baseline estimate of the recent score floor (1.0 in fixed mode)
        self.score_rel = 0.0      # score / baseline: what the thresholds actually compare against
        self._base: deque = deque(maxlen=max(1, int(self.cfg.baseline_s * self.fs)))
        self._above_since = self._below_since = None
        self._on_start = None  # start time of the clench in progress
        self._long = False
        self._clenches: list[tuple[float, float]] = []
        self._quiet_until = -np.inf
        self._refractory_until = -np.inf
        self.log: list[tuple[str, float]] = []  # ("clench", t) / ("chew", t) / ("long", t) / ("toggle", t)
        # read-only state for the live display (so the console shows what the detector sees, not a guess)
        self.n_toggles = 0
        self.last_reject: tuple[str, float] | None = None  # (reason, t) of the most recent discarded sequence
        self._last_toggle_t = -np.inf

    @classmethod
    def from_profile(cls, profile: dict, t0: float = 0.0) -> "ClenchDetector":
        cfg = ClenchConfig(**{k: (tuple(v) if k == "band" else v) for k, v in profile["config"].items()})
        return cls(profile["fs"], profile["rest_level"], cfg, t0)

    def scores(self, x: np.ndarray) -> np.ndarray:
        return np.median(self.front.push(x) / self.rest[:, None], axis=0)

    def _update_baseline(self) -> float:
        """Robust slow estimate of the recent score floor. Running median over the last baseline_s, never below 1.0
        (the calibrated rest level). The floor tracks the CURRENT level (rest, or raised during body teleop), but is
        frozen whenever a clench candidate is live - a bite in progress (_on_start) OR a sequence already underway
        (_clenches). Freezing the whole candidate keeps every bite of a chew train visible, so the 3+ rhythm
        rejection still fires, and stops a sustained clench from pulling the floor up before the next double. The
        median also shrugs off the brief onset samples. Fixed mode -> 1.0 (original behaviour, byte for byte)."""
        if self.cfg.baseline_mode != "moving":
            return 1.0
        base = max(1.0, float(np.median(self._base))) if self._base else 1.0
        if self._on_start is None and not self._clenches:  # quiet (or steadily elevated): let the floor track it
            self._base.append(self.score)
        return base

    def push(self, x: np.ndarray) -> list[float]:
        """Feed raw samples (n_ch, n). Returns the times of any toggles that completed inside this chunk."""
        c, out = self.cfg, []
        s_all = self.scores(x)
        for s in s_all:
            t = self.t0 + self.n / self.fs
            self.n += 1
            self.score = float(s)
            self.baseline = self._update_baseline()
            s = self.score / self.baseline  # detect on the score RELATIVE to the current level
            self.score_rel = float(s)
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
                        self._reject("held too long, ignored", t)
            toggled = self._check_sequence(t)
            if toggled is not None:
                out.append(toggled)
        return out

    def _reject(self, reason: str, t: float) -> None:
        self.last_reject = (reason, t)

    def _end_clench(self, start: float, end: float) -> None:
        c = self.cfg
        if end - start > c.max_clench_s:
            self._clenches = []
            self._reject("held too long, ignored", end)
            return
        if start < self._quiet_until:  # still inside a chewing episode: keep waiting for real quiet
            self._quiet_until = max(self._quiet_until, end + c.quiet_s)
            self._reject("waiting for quiet after chewing", end)
            return
        if start < self._refractory_until:
            return
        self.log.append(("clench", start))
        if self._clenches and start - self._clenches[-1][1] > c.max_gap_s:
            self._clenches = []  # previous single clench timed out
            self._reject("second clench too late", end)
        self._clenches.append((start, end))
        if len(self._clenches) >= 3:  # rhythmic: chewing / talking
            self.log.append(("chew", start))
            self._clenches = []
            self._quiet_until = end + c.quiet_s
            self._reject("looked like chewing (3+), ignored", end)

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
                self._reject("single clench (need a quick second)", t)
            return None
        if t - last_end >= c.confirm_s:  # exactly two, then quiet
            self._clenches = []
            self._refractory_until = t + c.refractory_s
            self.log.append(("toggle", t))
            self.n_toggles += 1
            self._last_toggle_t = t
            self.last_reject = None
            return t
        return None

    # --- read-only state for the live display -------------------------------------------------------------------
    def in_double(self) -> int:
        """How many clenches of the current double have been accepted so far (0, 1 or 2). A clench in progress that
        will extend the count is included, so the display can say 'CLENCH 1 of 2' while the first bite is happening."""
        n = len(self._clenches)
        if self._on_start is not None and not self._long and n < 2:
            n += 1  # the bite happening right now
        return n

    def state_word(self) -> str:
        """Plain-English state the wearer walks through, derived from the detector's own fields (not guessed)."""
        if self._on_start is not None:
            if self._long:
                return "HELD (ignored)"
            return f"CLENCH {self.in_double()} of 2"
        if self._last_toggle_t > -np.inf and (self.t0 + self.n / self.fs) - self._last_toggle_t < 0.8:
            return "TOGGLE SENT"
        if self._clenches:
            n = len(self._clenches)
            return "GOT 2 - confirming" if n >= 2 else "GOT 1 of 2 - clench again"
        return "RELAXED"

    def status(self) -> "ClenchStatus":
        """A snapshot of everything the console display needs; read-only, cheap, safe to call every frame."""
        rej = self.last_reject[0] if self.last_reject is not None else ""
        return ClenchStatus(
            score=self.score, baseline=self.baseline, rel=self.score_rel,
            on_ratio=self.cfg.on_ratio, off_ratio=self.cfg.off_ratio,
            moving=(self.cfg.baseline_mode == "moving"),
            state=self.state_word(), in_double=self.in_double(),
            clenching=self._on_start is not None and not self._long,
            n_toggles=self.n_toggles, last_reject=rej)


@dataclass
class ClenchStatus:
    """Immutable view of the detector for the live console display."""
    score: float       # raw envelope ratio vs the calibrated relaxed level
    baseline: float    # moving-baseline floor (1.0 in fixed mode)
    rel: float         # score / baseline: the value compared against the thresholds
    on_ratio: float
    off_ratio: float
    moving: bool
    state: str         # RELAXED / CLENCH 1 of 2 / CLENCH 2 of 2 / GOT 1 of 2 ... / TOGGLE SENT / HELD (ignored)
    in_double: int     # 0, 1 or 2
    clenching: bool
    n_toggles: int
    last_reject: str   # "" or why the last sequence did not fire

    def gripper_expected(self) -> str:
        """The app toggles the gripper OPEN<->CLOSED on every message and never reports back, so the laptop can only
        show the EXPECTED state: starts OPEN, the first toggle closes it."""
        return "CLOSED" if self.n_toggles % 2 == 1 else "OPEN"


# --- live console display ------------------------------------------------------------------------------------------
class _C:
    """ANSI colours, blanked when colour is off (plain Windows cmd.exe or a redirected log)."""
    def __init__(self, on: bool):
        self.on = on

    def __call__(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s


def enable_ansi(no_color: bool = False) -> bool:
    """Decide whether to emit ANSI, and on Windows 10+ switch the console into VT mode so the codes render. Degrades
    to plain text when stdout is not a tty, --no-color is set, or VT cannot be enabled."""
    if no_color or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING | existing flags
        except Exception:  # noqa: BLE001  (old Windows: no VT -> plain text)
            return False
    return True


def status_line(st: ClenchStatus, *, width: int = 16, color: bool = False) -> str:
    """One glanceable line: a JAW bar with the threshold drawn as a | marker, the live ratio, the state word, and the
    expected gripper state. In moving mode it also shows the current baseline. No trailing newline."""
    c = _C(color)
    on = max(st.on_ratio, 1e-6)
    full = 2.0 * on  # the bar spans 0 .. 2x threshold, so the threshold marker sits in the middle
    filled = int(round(width * min(st.rel, full) / full))
    mark = max(0, min(width - 1, int(round(width * on / full))))  # column of the threshold |
    cells = []
    for i in range(width):
        if i == mark:
            cells.append(c("1;37", "|"))
        elif i < filled:
            cells.append(c("31" if st.clenching else "32", "#"))
        else:
            cells.append(".")
    bar = "".join(cells)
    base = f" base {st.baseline:4.1f}x" if st.moving else ""
    state = c("1;33" if st.clenching else "0", f"{st.state:<20}")
    grip = st.gripper_expected()
    grip_s = c("1;31" if grip == "CLOSED" else "1;32", grip)
    line = (f"JAW [{bar}] {st.rel:5.1f}x (need {st.on_ratio:.1f}x){base}  {state}  gripper: {grip_s}")
    if st.last_reject and not st.clenching:
        line += c("90", f"   [{st.last_reject}]")
    return line


def toggle_banner(st: ClenchStatus, *, dry: bool = False, color: bool = False) -> str:
    """The big, unmissable line printed on its OWN line for every toggle (never overwritten)."""
    c = _C(color)
    grip = st.gripper_expected()  # state AFTER this toggle
    tag = "(dry run)" if dry else "sent"
    stamp = time.strftime("%H:%M:%S")
    msg = f">>> GRIPPER {grip} {tag}  (toggle #{st.n_toggles} -> {grip} expected, {stamp})"
    return c("1;31" if grip == "CLOSED" else "1;32", msg)


class ScoreLogger:
    """Optional ~5 Hz CSV log so a live teleop run can be compared against rest afterwards. Columns:
    t, raw_score, baseline, rel_score, state, n_toggles. Best-effort: never raises into the read loop."""
    def __init__(self, path, period: float = 0.2):
        self.period = period
        self._last = -1e18
        self._fh = None
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(p, "w", buffering=1)
            self._fh.write("t,raw_score,baseline,rel_score,state,n_toggles\n")
            self._t0 = time.time()
        except OSError:
            self._fh = None

    def maybe_log(self, st: ClenchStatus, now: float | None = None) -> None:
        if self._fh is None:
            return
        now = time.time() if now is None else now
        if now - self._last < self.period:
            return
        self._last = now
        try:
            self._fh.write(f"{now - self._t0:.3f},{st.score:.3f},{st.baseline:.3f},"
                           f"{st.rel:.3f},{st.state.replace(',', ' ')},{st.n_toggles}\n")
        except OSError:
            pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


# --- calibration ----------------------------------------------------------------------------------------------------
def calibrate(rest: np.ndarray, clench: np.ndarray, other: np.ndarray | None, fs: float,
              cfg: ClenchConfig | None = None) -> dict:
    """rest/clench/other: (n_ch, n) raw recordings. Returns a profile dict (json-able) with a quality verdict."""
    cfg = cfg or ClenchConfig()
    n_ch = rest.shape[0]

    def env(x):
        # Skip the ~1 s filter start-up transient, but never trim away the whole capture: a short or stuttering
        # capture (e.g. the synthetic board delivering <= 1 s of samples under test timing) must still leave samples
        # to score, or np.percentile/np.median would hit an empty array (IndexError). Keep at least half.
        y = EmgFrontEnd(fs, n_ch, cfg).push(x)
        settle = min(int(1.0 * fs), y.shape[1] // 2)
        return y[:, settle:]

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
    if kind == "hub":  # live EEG relayed from the laptop by eeg.stream_hub; no BrainFlow needed here
        import os

        from rover_intent.eeg.stream_hub import FANOUT_PORT, HubBoard
        host = os.environ.get("HUB_FANOUT_HOST", "127.0.0.1")
        port = int(os.environ.get("HUB_FANOUT_PORT", str(FANOUT_PORT)))
        board = HubBoard(host, port)
        board.prepare_session()
        board.start_stream()
        return board, board.fs, [0, 1, 2, 3]

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


def apply_baseline(profile: dict, baseline: str | None) -> dict:
    """Override a profile's baseline_mode from a CLI flag ("fixed"/"moving"/None = leave as stored). Returns the same
    dict (mutated) so the caller can build a detector from it."""
    if baseline:
        profile.setdefault("config", {})["baseline_mode"] = baseline
    return profile


def cmd_run(args) -> int:
    import socket

    profile = apply_baseline(json.loads(Path(args.profile).read_text()), getattr(args, "baseline", None))
    board, fs, chans = open_board(args.board, args.serial, args.mac, args.name)
    if abs(fs - profile["fs"]) > 1:
        print(f"warning: board runs at {fs} Hz but the profile was made at {profile['fs']} Hz", file=sys.stderr)
    det = ClenchDetector.from_profile(profile, t0=time.time())
    color = enable_ansi(getattr(args, "no_color", False))
    tx = None if args.dry else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    logger = ScoreLogger(args.score_log) if getattr(args, "score_log", None) else None
    mode = det.cfg.baseline_mode
    print(f"double-clench to toggle the gripper. threshold {det.cfg.on_ratio}x relaxed, baseline={mode}. "
          f"Ctrl+C to stop.", flush=True)
    last_print = 0.0
    try:
        while True:
            x = _read(board, chans)
            if x.shape[1]:
                for t in det.push(x):
                    if tx:
                        tx.sendto(toggle_packet(time.time()), (args.host, args.port))
                    print("\n" + toggle_banner(det.status(), dry=tx is None, color=color), flush=True)
            if logger is not None:
                logger.maybe_log(det.status())
            if time.time() - last_print > 0.1:
                print("\r" + status_line(det.status(), color=color) + "   ", end="", flush=True)
                last_print = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if logger is not None:
            logger.close()
        board.stop_stream()
        board.release_session()
    return 0


def cmd_replay(args) -> int:
    profile = apply_baseline(json.loads(Path(args.profile).read_text()), getattr(args, "baseline", None))
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
        p.add_argument("--board", choices=["muse2", "muse2-bled", "synthetic", "hub"], default="muse2")
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
    r.add_argument("--baseline", choices=["fixed", "moving"], default=None,
                   help="override the profile's baseline mode (moving tracks the current level for body teleop)")
    r.add_argument("--score-log", default=None, help="write a ~5 Hz CSV of score/baseline/rel/state for later review")
    r.add_argument("--no-color", action="store_true", help="plain text, no ANSI colours")
    p = sub.add_parser("replay")
    p.add_argument("--profile", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--baseline", choices=["fixed", "moving"], default=None,
                   help="override the profile's baseline mode for the replay")
    args = ap.parse_args(argv)
    return {"calibrate": cmd_calibrate, "run": cmd_run, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

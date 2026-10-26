"""Put the Muse 2 on ANYONE, calibrate, confirm it works, and go live. One command on the Windows laptop.

This is the day-to-day "slip the headband on and go" routine, so a new wearer never has to remember the four
separate steps. It opens the Muse ONCE and keeps it open through all four steps (BLE re-acquire on Windows is slow
and flaky, so we never release the board between calibration and streaming):

  1. FIT CHECK      per-sensor contact quality (TP9/AF7/AF8/TP10), named by where each sits, until all four pass.
  2. QUICK CALIB    ~60-90 s: relax, five double clenches, a short negatives block -> data/profiles/clench_<name>.json.
  3. READY CHECK     live clench bar with the new profile; a double clench prints TOGGLE (test, nothing sent). Enter -> go.
  4. GO              hand the same open board to eeg.stream_tx: raw EEG -> hub, clench -> GripToggle UDP to the app.

It reuses the signal-quality heuristics and block recording from eeg.sunday and the clench fit from eeg.clench, so
there is one set of thresholds, not a second copy. Windows needs only base Python 3.12 + numpy + brainflow; SciPy may
be blocked there, and clench.py already falls back to eeg/np_signal.py.

Usage (laptop with the Muse; Bluetooth on this machine):
  python -m rover_intent.eeg.muse_ready --name alice
  python -m rover_intent.eeg.muse_ready --name alice --board muse2-bled --serial COM3
  python -m rover_intent.eeg.muse_ready --name alice --board synthetic            # no-Muse smoke test
  python -m rover_intent.eeg.muse_ready --name alice --skip-calib --profile data/profiles/clench_alice.json
  python -m rover_intent.eeg.muse_ready --name alice --no-stream                  # stop after the Enter
  --line-hz 50 for 50 Hz mains (default 60). Hub/app hosts/ports override the live defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rover_intent.eeg.clench import (
    MUSE_CHANNELS,
    ClenchDetector,
    ScoreLogger,
    _read,
    _record,
    apply_baseline,
    calibrate,
    enable_ansi,
    open_board,
    status_line,
    toggle_banner,
)
from rover_intent.eeg.sunday import (
    Block,
    SundayConfig,
    _backup_and_write,
    channel_quality,
    record_block,
)
from rover_intent.eeg import stream_tx

# Where each Muse electrode sits, so we can tell the wearer which one to fix and where it is on their head.
SENSOR_SITES = {
    "TP9": "behind the LEFT ear",
    "AF7": "LEFT forehead (above the left eyebrow)",
    "AF8": "RIGHT forehead (above the right eyebrow)",
    "TP10": "behind the RIGHT ear",
}
# What each bad-contact status means and the fix, in plain words (statuses come from sunday.channel_quality).
STATUS_FIX = {
    "flat": "flat / not touching skin: press it to the skin, wet it, or move hair out from under it",
    "drift": "saturated / drifting: sit still, reseat the band so it sits flat, wet the contact",
    "mains": "mains (50/60 Hz) noise: push the band down onto skin, move away from chargers and screens",
}


@dataclass
class ReadyConfig:
    """Durations for the routine. Defaults are the ~2-3 min real run; tests shrink them for the synthetic smoke test."""
    line_hz: float = 60.0
    # fit check
    fit_window_s: float = 1.0      # each quality capture is this long (~1 Hz feedback)
    fit_pass_s: float = 3.0        # all four sensors must stay good for this long in a row
    fit_max_s: float = 0.0         # safety cap (0 = no cap; loop until good or Enter)
    # quick calibration
    rest_s: float = 20.0
    double_reps: int = 5
    double_s: float = 4.0
    neg_chew_s: float = 8.0
    neg_blink_s: float = 6.0


def _hdr(step: str, title: str) -> None:
    """A big, obvious step header the wearer can read from across the room."""
    line = "=" * 64
    print(f"\n{line}\n  {step}  {title}\n{line}", flush=True)


def _say(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def _poll_enter() -> bool:
    """Non-blocking: True if Enter was pressed since the last call (real keyboard only; returns False under a fake
    stdin / pipe). Used so the FIT CHECK can offer 'press Enter to continue anyway' without blocking the live loop."""
    try:
        if os.name == "nt":
            import msvcrt
            hit = False
            while msvcrt.kbhit():
                if msvcrt.getwch() in ("\r", "\n"):
                    hit = True
            return hit
        import select
        if not sys.stdin or not sys.stdin.isatty():
            return False
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            sys.stdin.readline()
            return True
        return False
    except Exception:  # noqa: BLE001  (no usable stdin: just never skip)
        return False


# --- step 1: fit check ---------------------------------------------------------------------------------------------
def fit_check(board, fs, chans, scfg: SundayConfig, cfg: ReadyConfig, *, quiet: bool = False,
              poll_enter=_poll_enter) -> dict:
    """Loop showing per-sensor contact quality (~1 Hz) until all four sensors pass for cfg.fit_pass_s in a row, or the
    wearer presses Enter to continue anyway (a WARN). Returns the last quality dict (with 'skipped' set if Enter)."""
    _hdr("STEP 1/4", "FIT CHECK  -  seat the headband")
    print("Keep still. Each sensor should read 'good'. I will tell you which one to fix and where it sits.", flush=True)
    good_since = None
    t_start = time.time()
    last = {"ok": False}
    while True:
        if poll_enter():
            print("\n(Enter pressed: continuing anyway. WARNING: contact was not confirmed good.)", flush=True)
            last["skipped"] = True
            return last
        x = _record(board, chans, cfg.fit_window_s, None)
        last = channel_quality(x, fs, scfg)
        chmap = last.get("channels", {})
        line = "  ".join(f"{ch}:{chmap.get(ch, {}).get('status', '?')}" for ch in MUSE_CHANNELS[:len(chmap)])
        now = time.time()
        if last.get("ok"):
            good_since = good_since or now
            held = now - good_since
            print(f"\r  {line}   all good {held:3.1f}/{cfg.fit_pass_s:.0f}s   ", end="", flush=True)
            if held >= cfg.fit_pass_s:
                print("\n  All four sensors good. Fit confirmed.", flush=True)
                return last
        else:
            good_since = None
            print(f"\r  {line}", flush=True)
            for ch in last.get("bad", []):
                site = SENSOR_SITES.get(ch, ch)
                status = chmap.get(ch, {}).get("status", "")
                print(f"    FIX {ch} = {site}: {STATUS_FIX.get(status, status)}", flush=True)
            if not quiet:
                print("    (press Enter to go ahead anyway)", flush=True)
        if cfg.fit_max_s and now - t_start > cfg.fit_max_s:
            print("\n  (fit-check time cap reached; continuing)", flush=True)
            last["skipped"] = True
            return last


# --- step 2: quick calibration -------------------------------------------------------------------------------------
def _calib_blocks(cfg: ReadyConfig) -> list[Block]:
    return [
        Block("rest", "rest", "emg", cfg.rest_s,
              "RELAX: jaw loose, eyes open, blink normally, sit still."),
        Block("clench_double", "clench", "emg", cfg.double_s,
              "DOUBLE CLENCH: two quick bites in a row (like a double click), then relax.",
              reps=cfg.double_reps),
        Block("neg_chew", "negative", "emg", cfg.neg_chew_s,
              "CHEW and talk a little (do NOT double-clench)."),
        Block("neg_blink", "negative", "emg", cfg.neg_blink_s,
              "BLINK a few times and move your eyes (no clenching)."),
    ]


def quick_calibrate(board, fs, chans, name: str, cfg: ReadyConfig, *, profiles_dir: str = "data/profiles",
                    quiet: bool = False, ask=input) -> dict:
    """Record ~20 s rest + five double clenches + a short negatives block, fit the clench profile (clench.calibrate),
    write it to <profiles_dir>/clench_<name>.json (old one backed up), and print the fit quality honestly. On a 'poor'
    fit, offer a redo (Enter = redo, 's' = continue anyway). Returns the profile dict (with ['stats'])."""
    while True:
        _hdr("STEP 2/4", "QUICK CALIBRATION  -  learn your double clench")
        blocks = _calib_blocks(cfg)
        rec: dict[str, np.ndarray] = {}
        for b in blocks:
            _say(b.cue)
            rec[b.name] = record_block(board, chans, b, quiet=quiet)
        rest = rec["rest"]
        double = rec["clench_double"]
        neg = np.concatenate([rec["neg_chew"], rec["neg_blink"]], axis=1)
        profile = calibrate(rest, double, neg if neg.shape[1] else None, fs)
        s = profile["stats"]
        _say(f"Fit: {s['quality'].upper()}  (threshold {s['on_ratio']}x relaxed, clench peak {s['clench_p95']}x, "
             f"margin {s['margin']}x; false toggles while relaxed {s['rest_toggles']}, "
             f"while chewing/blinking {s['other_toggles']})")

        clench_path = Path(profiles_dir) / f"clench_{name}.json"
        backup = _backup_and_write(clench_path, profile)
        print(f"  profile -> {clench_path}" + (f"  (old -> {backup})" if backup else "  (no previous profile)"),
              flush=True)

        if s["quality"] != "poor":
            return profile
        print("  Separation is POOR: a double clench may be missed or chewing may toggle it.", flush=True)
        ans = (ask("  Press Enter to redo the calibration, or type 's' then Enter to continue anyway: ") or "").strip()
        if ans.lower() == "s":
            return profile
        # otherwise loop and redo


# --- step 3: ready check (an Enter button) -------------------------------------------------------------------------
def ready_check(board, fs, chans, profile: dict, *, quiet: bool = False, ask=input,
                stop: threading.Event | None = None, color: bool = False) -> int:
    """Live clench status line with the new profile. A detected double clench prints a TOGGLE banner (test, NOT sent).
    Nothing is sent to the app here. Returns when the wearer presses Enter. Returns the number of test toggles seen."""
    _hdr("STEP 3/4", "READY CHECK  -  try it before it is live")
    print("Try a double clench: watch the JAW bar and the state walk CLENCH 1 of 2 -> CLENCH 2 of 2 -> TOGGLE.",
          flush=True)
    print("Nothing is sent yet. Press Enter when ready to go LIVE.", flush=True)
    det = ClenchDetector.from_profile(profile, t0=time.time())
    go = stop or threading.Event()

    def _wait():
        try:
            ask("")
        except (EOFError, OSError):
            pass
        go.set()

    threading.Thread(target=_wait, daemon=True).start()
    n_toggles = 0
    last_print = 0.0
    while not go.is_set():
        x = _read(board, chans)
        if x.shape[1]:
            for _t in det.push(x):
                n_toggles += 1
                print("\n  " + toggle_banner(det.status(), dry=True, color=color) + "  (test, not sent)", flush=True)
        if not quiet and time.time() - last_print > 0.1:
            print("\r  " + status_line(det.status(), color=color) + "  (Enter to go)   ", end="", flush=True)
            last_print = time.time()
        time.sleep(0.02)
    print("", flush=True)
    return n_toggles


# --- orchestration -------------------------------------------------------------------------------------------------
def run(name: str, board_kind: str = "muse2", *, serial: str = "", mac: str = "", muse_name: str = "",
        line_hz: float = 60.0, hub_host: str = "rover-vm", hub_port: int = stream_tx.LISTEN_PORT,
        app_host: str = "rover-vm", app_port: int = 47100, skip_calib: bool = False,
        profile_path: str | None = None, no_stream: bool = False, profiles_dir: str = "data/profiles",
        cfg: ReadyConfig | None = None, quiet: bool | None = None, ask=input, poll_enter=_poll_enter,
        stop: threading.Event | None = None, stream_seconds: float | None = None,
        baseline: str | None = None, score_log: str | None = None, no_color: bool = False) -> dict:
    """The whole routine on ONE already-chosen board. Opens the Muse once and keeps it open through all four steps."""
    cfg = cfg or ReadyConfig(line_hz=line_hz)
    quiet = (board_kind == "synthetic") if quiet is None else quiet
    scfg = SundayConfig(name=name, line_hz=cfg.line_hz)
    color = enable_ansi(no_color)
    result: dict = {"name": name, "board": board_kind}

    _say(f"Muse ready routine for '{name}'. Board: {board_kind}. About 2-3 minutes. The console is your guide.")
    _say("Make sure the Muse is on and Bluetooth is enabled on this laptop.")
    board, fs, chans = open_board(board_kind, serial, mac, muse_name)
    _read(board, chans)  # drop start-up junk
    closed = False
    try:
        # 1. fit check
        result["fit"] = fit_check(board, fs, chans, scfg, cfg, quiet=quiet, poll_enter=poll_enter)

        # 2. calibration (or load an existing profile)
        if skip_calib:
            if not profile_path:
                raise SystemExit("--skip-calib needs --profile <path>")
            profile = json.loads(Path(profile_path).read_text())
            _say(f"Using existing profile {profile_path} (threshold {profile['config']['on_ratio']}x relaxed).")
        else:
            profile = quick_calibrate(board, fs, chans, name, cfg, profiles_dir=profiles_dir, quiet=quiet, ask=ask)
        profile = apply_baseline(profile, baseline)  # CLI --baseline wins over the stored mode for the live run
        result["profile"] = profile.get("stats", {})
        result["baseline_mode"] = profile.get("config", {}).get("baseline_mode", "moving")

        # 3. ready check (Enter button)
        result["ready_toggles"] = ready_check(board, fs, chans, profile, quiet=quiet, ask=ask, color=color)

        if no_stream:
            _say("Stopping before streaming (--no-stream). Profile is saved and the fit is confirmed.")
            return result

        # 4. go: hand the SAME open board to the streamer without releasing the Muse
        _hdr("STEP 4/4", "GO  -  streaming EEG to the hub, clench -> gripper is LIVE")
        print("READY: streaming EEG to the hub and clench -> gripper is LIVE. Ctrl+C to stop.", flush=True)
        go_stop = stop or threading.Event()
        if stream_seconds is not None:
            threading.Timer(stream_seconds, go_stop.set).start()
        stream_tx.stream_session(
            board, fs, chans, board_kind=board_kind, hub_host=hub_host, hub_port=hub_port,
            clench_profile=profile, app_host=app_host, app_port=app_port,
            serial=serial, mac=mac, name=muse_name, quiet=quiet, stop=go_stop,
            announce=not quiet, close_board=True, baseline=baseline, score_log=score_log, no_color=no_color)
        closed = True  # stream_session closed the board
        result["streamed"] = True
        return result
    finally:
        if not closed:
            try:
                board.stop_stream()
                board.release_session()
            except Exception:  # noqa: BLE001
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="wearer's name; profile -> data/profiles/clench_<name>.json")
    ap.add_argument("--board", choices=["muse2", "muse2-bled", "synthetic"], default="muse2")
    ap.add_argument("--serial", default="", help="BLED112 dongle port for muse2-bled (COM3, /dev/ttyACM0)")
    ap.add_argument("--mac", default="", help="Muse MAC address (optional)")
    ap.add_argument("--muse-name", default="", help="specific Muse, e.g. Muse-1A2B (optional)")
    ap.add_argument("--line-hz", type=float, default=60.0, help="mains frequency (60 NA, 50 EU)")
    ap.add_argument("--hub-host", default="rover-vm")
    ap.add_argument("--hub-port", type=int, default=stream_tx.LISTEN_PORT)
    ap.add_argument("--app-host", default="rover-vm")
    ap.add_argument("--app-port", type=int, default=47100)
    ap.add_argument("--skip-calib", action="store_true", help="skip calibration; use --profile (fit check + Enter + go)")
    ap.add_argument("--profile", default=None, help="existing clench profile for --skip-calib")
    ap.add_argument("--no-stream", action="store_true", help="stop after the Enter (do not go live)")
    ap.add_argument("--profiles-dir", default="data/profiles")
    ap.add_argument("--baseline", choices=["fixed", "moving"], default=None,
                    help="clench baseline: moving tracks the current level (body teleop), fixed uses the calibrated "
                         "relaxed level (sitting still). Default = moving (the profile's own setting).")
    ap.add_argument("--score-log", default=None,
                    help="write a ~5 Hz CSV (t, raw score, baseline, rel score, state, toggles) during the GO stream")
    ap.add_argument("--no-color", action="store_true", help="plain text, no ANSI colours (safe everywhere)")
    args = ap.parse_args(argv)

    if args.app_port == 47100 and args.app_host not in ("rover-vm",):
        pass  # the live app default is intentional; no guard needed here
    res = run(args.name, args.board, serial=args.serial, mac=args.mac, muse_name=args.muse_name,
              line_hz=args.line_hz, hub_host=args.hub_host, hub_port=args.hub_port,
              app_host=args.app_host, app_port=args.app_port, skip_calib=args.skip_calib,
              profile_path=args.profile, no_stream=args.no_stream, profiles_dir=args.profiles_dir,
              baseline=args.baseline, score_log=args.score_log, no_color=args.no_color)
    return 0 if res else 1


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())

"""Replay a recorded app session (logs/sessions/<timestamp>.jsonl) back into a RUNNING app over UDP at the original
timing, and record what the app did, then print a pass/fail summary.

Each recorded line is {"t": <app receive time>, "msg": {...}} as written by rover_intent.session.SessionRecorder.
The messages are replayed exactly as recorded (gaps between them reproduced, divided by --speed), except that you may
window the replay (--start/--end seconds from the first line), keep only some kinds (--only body,grip_toggle,...), or
DROP a gripper source to isolate the other: --drop-grip-toggle (no Muse toggles) / --drop-hand (blank hand_open).

While replaying, a monitor polls the app's Query interface at a MODEST rate (2-5 Hz, a blocking recv, never a tight
loop) and logs per time: clutch state, mode, gripper value + source, and the sim joint positions / TCP. At the end it
prints a pass/fail summary:
  (a) the sim arm moved while the clutch was following (TCP path length over a threshold),
  (b) every replayed Muse grip_toggle produced a gripper change within 0.5 s (matched / missed; coalesced-absorbed),
  (c) hand open/fist edges produced gripper changes (count),
  (d) no gripper changes without a cause (spurious count),
  (e) zero app errors.

Validate against a SPARE mock app this script starts and stops itself (never the live ports):
  .venv/bin/python scripts/replay_session.py logs/sessions/20261025_163116.jsonl --spawn-mock \
      --start 2420 --end 2520 --engage
Against a live stack (the coordinator runs this with the Isaac app on 47100; do NOT run it here):
  .venv/bin/python scripts/replay_session.py logs/sessions/20261025_163116.jsonl --host 127.0.0.1 --port 47100

A plain replay with no monitor (back-compatible) still works:
  .venv/bin/python scripts/replay_session.py <file> --host 127.0.0.1 --port 47100 --speed 1.0
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rover_intent import griptest as G   # noqa: E402

COALESCE_S = 0.6
TCP_MOVE_THRESH = 0.02   # m: "the arm moved while following"


def load(path):
    """Yield (t, msg_dict) rows from a session file, skipping blank/garbled lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows.append((float(r["t"]), r["msg"]))
            except (ValueError, KeyError):
                continue
    return rows


def _filter(rows, start=None, end=None, only=None, drop_grip_toggle=False, drop_hand=False):
    if not rows:
        return rows
    t0 = rows[0][0]
    out = []
    for t, msg in rows:
        rel = t - t0
        if start is not None and rel < start:
            continue
        if end is not None and rel > end:
            continue
        kind = msg.get("kind")
        if only and kind not in only:
            continue
        if drop_grip_toggle and kind == "grip_toggle":
            continue
        if drop_hand and kind == "body":
            msg = {**msg, "hand_open": None}   # blank the hand source so only the Muse drives the gripper
        out.append((t, msg))
    return out


def replay(path, host="127.0.0.1", port=47100, rate=1.0, speed=None, sleep=time.sleep, send=None,
           start=None, end=None, only=None, drop_grip_toggle=False, drop_hand=False, on_send=None):
    """Replay rows into the app. `speed` (if given) overrides `rate`. Back-compatible: replay(path, rate=, sleep=,
    send=) still reproduces the gaps divided by rate. on_send(wall_t, msg) is called after each send."""
    rate = speed if speed is not None else rate
    rows = _filter(load(path), start, end, only, drop_grip_toggle, drop_hand)
    if send is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        send = lambda d: sock.sendto(json.dumps(d).encode(), (host, port))
    t_prev = None
    for t, msg in rows:
        if t_prev is not None:
            gap = max(0.0, (t - t_prev) / max(rate, 1e-6))
            if gap > 0:
                sleep(gap)
        send(msg)
        if on_send is not None:
            on_send(time.time(), msg)
        t_prev = t
    return len(rows)


# --- engagement preamble (optional; for a mid-session window) -----------------------------------------------------
def play_engage(send, hz=20.0):
    """Stream a synthetic palm-up-and-hold so the clutch engages through the normal path, so a windowed replay that
    starts mid-session still reaches 'following'. This is an explicit opt-in preamble (--engage); it adds frames,
    it never rewrites the recorded session."""
    rows, _, seq = G.engage_frames(time.time() + 0.05)
    for st, msg in rows:
        d = st - time.time()
        if d > 0:
            time.sleep(d)
        send(msg)
    t0 = time.time()
    while time.time() < t0 + 0.6:
        send(G.make_body(time.time(), seq, hand_open=0.9))
        seq += 1
        time.sleep(1.0 / hz)


# --- summary ------------------------------------------------------------------------------------------------------
def summarize(toggles_sent, parsed, monitor, have_log):
    trans = [t for t in parsed["transitions"] if t["t"] is not None]
    followed = monitor.followed()
    tcp_path = monitor.tcp_path_following()
    # (b) each Muse toggle -> a gripper change within 0.5 s (coalesced-absorbed counted separately)
    matched = missed = absorbed = 0
    lat = []
    for ts in toggles_sent:
        chg = [tr for tr in trans if tr["t"] is not None and ts - 0.05 <= tr["t"] <= ts + 0.5]
        hand_near = [tr for tr in trans if tr["t"] is not None and tr["source"] == "hand"
                     and ts - COALESCE_S <= tr["t"] <= ts + COALESCE_S]
        if chg:
            matched += 1
            lat.append((chg[0]["t"] - ts) * 1000.0)
        elif hand_near:
            absorbed += 1          # coalesced with a near hand edge: correct, not a change
        else:
            missed += 1
    # (c) hand edges that moved the gripper
    hand_changes = sum(1 for tr in trans if tr["source"] == "hand")
    # (d) spurious: a gripper change with no attributable source
    known = {"hand", "muse", "eeg", "voice", "skill"}
    spurious = sum(1 for tr in trans if (tr["source"] or "?") not in known)
    errors = parsed["errors"]
    return {"followed": followed, "tcp_path": tcp_path, "toggles": len(toggles_sent), "matched": matched,
            "missed": missed, "absorbed": absorbed, "toggle_lat_ms": lat, "hand_changes": hand_changes,
            "muse_changes": sum(1 for tr in trans if tr["source"] in ("muse", "eeg")), "spurious": spurious,
            "clashes": monitor.max_clashes(), "errors": errors, "have_log": have_log, "n_transitions": len(trans),
            "n_samples": len(monitor.samples)}


def print_summary(s):
    def pf(ok):
        return "PASS" if ok else "FAIL"
    med = statistics.median(s["toggle_lat_ms"]) if s["toggle_lat_ms"] else None
    p90 = sorted(s["toggle_lat_ms"])[min(len(s["toggle_lat_ms"]) - 1, int(0.9 * (len(s["toggle_lat_ms"]) - 1)))] \
        if s["toggle_lat_ms"] else None
    print()
    print(f"polled samples={s['n_samples']}  app-log transitions={s['n_transitions']}  clashes={s['clashes']}")
    print(f"(a) arm moved while following:   {pf(s['followed'] and s['tcp_path'] > TCP_MOVE_THRESH)}  "
          f"(followed={s['followed']}, tcp_path_following={s['tcp_path']:.3f} m, thresh {TCP_MOVE_THRESH} m)")
    print(f"(b) Muse toggle -> change <0.5s: {pf(s['missed'] == 0)}  "
          f"(toggles={s['toggles']}, matched={s['matched']}, coalesced-absorbed={s['absorbed']}, missed={s['missed']}"
          + (f", lat med/p90={med:.0f}/{p90:.0f} ms" if med is not None else "") + ")")
    print(f"(c) hand edges -> changes:       {s['hand_changes']} hand gripper changes "
          f"({s['muse_changes']} muse/eeg changes)")
    print(f"(d) no change without a cause:   {pf(s['spurious'] == 0)}  (spurious={s['spurious']})")
    if s["have_log"]:
        print(f"(e) zero app errors:             {pf(len(s['errors']) == 0)}  (errors={len(s['errors'])})")
        for e in s["errors"][:5]:
            print(f"      {e}")
    else:
        print("(e) zero app errors:             n/a (no app log; --no-spawn against a live app cannot read it)")


# --- harness driver -----------------------------------------------------------------------------------------------
def run_harness(file, host, port, speed, start, end, only, drop_grip_toggle, drop_hand, poll_hz, engage,
                spawn_mock, config, backend):
    app = None
    if spawn_mock:
        ports = dict(G.SPARE_PORTS, listen_port=port)
        tmp = G.write_spare_config(config, G.ROOT / "logs" / "replay_cfg.yaml", ports)
        app = G.AppProcess(str(tmp), backend, log_path=G.ROOT / "logs" / "replay_app.log").start()
        if not app.wait_ready(port, host):
            app.stop()
            raise SystemExit("mock app did not come up on the spare port")

    snd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(m):
        snd.sendto(json.dumps(m).encode(), (host, port))

    mon = G.AppMonitor(host, port, hz=poll_hz)
    mon.start()
    toggles_sent = []

    def on_send(wall_t, msg):
        if msg.get("kind") == "grip_toggle":
            toggles_sent.append(wall_t)

    try:
        if engage:
            play_engage(send)
        n = replay(file, host=host, port=port, speed=speed, start=start, end=end, only=only,
                   drop_grip_toggle=drop_grip_toggle, drop_hand=drop_hand, send=send, on_send=on_send)
        time.sleep(0.7)
    finally:
        mon.stop()
        mon.join(timeout=2)
        rc = app.stop() if app is not None else None

    have_log = app is not None
    parsed = G.parse_app_log(Path(app.log_path).read_text()) if have_log else \
        {"transitions": [], "toggles": [], "clashes": [], "errors": []}
    print(f"replayed {n} messages from {file} to {host}:{port} at speed {speed}"
          + (f" (window {start}-{end}s)" if (start or end) else "")
          + (" [--drop-grip-toggle]" if drop_grip_toggle else "")
          + (" [--drop-hand]" if drop_hand else ""))
    print_summary(summarize(toggles_sent, parsed, mon, have_log))
    if rc is not None:
        print(f"(mock app exit code {rc}; -15 = SIGTERM from this harness, not a crash)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None, help="app listen port (default 47100, or the spare port with --spawn-mock)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0 = original timing; 2 = twice as fast)")
    ap.add_argument("--rate", type=float, default=None, help="alias for --speed")
    ap.add_argument("--start", type=float, default=None, help="replay from this many seconds into the session")
    ap.add_argument("--end", type=float, default=None, help="replay up to this many seconds into the session")
    ap.add_argument("--only", default=None, help="comma list of kinds to keep (body,grip_toggle,preset,utterance)")
    ap.add_argument("--drop-grip-toggle", action="store_true", help="drop Muse toggles (isolate the hand source)")
    ap.add_argument("--drop-hand", action="store_true", help="blank hand_open (isolate the Muse source)")
    ap.add_argument("--poll-hz", type=float, default=4.0, help="Query poll rate (2-5 Hz)")
    ap.add_argument("--engage", action="store_true", help="prepend a synthetic palm-up engage so a windowed replay reaches 'following'")
    ap.add_argument("--spawn-mock", action="store_true", help="start+stop a spare-port mock app for validation")
    ap.add_argument("--backend", default="mock", choices=["mock", "isaac"])
    ap.add_argument("--config", default=str(G.ROOT / "configs/default.yaml"))
    ap.add_argument("--no-summary", action="store_true", help="plain replay, no monitor/summary (back-compatible)")
    a = ap.parse_args()
    speed = a.rate if a.rate is not None else a.speed
    only = [k.strip() for k in a.only.split(",")] if a.only else None
    port = a.port or (G.SPARE_PORTS["listen_port"] if a.spawn_mock else 47100)
    if a.no_summary:
        n = replay(a.file, a.host, port, speed=speed, start=a.start, end=a.end, only=only,
                   drop_grip_toggle=a.drop_grip_toggle, drop_hand=a.drop_hand)
        print(f"replayed {n} messages from {a.file} to {a.host}:{port} at speed {speed}")
        return
    run_harness(a.file, a.host, port, speed, a.start, a.end, only, a.drop_grip_toggle, a.drop_hand, a.poll_hz,
                a.engage, a.spawn_mock, a.config, a.backend)


if __name__ == "__main__":
    main()

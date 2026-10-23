"""Replay a recorded session (logs/sessions/<timestamp>.jsonl) back into the app over UDP, with the original timing.

    python scripts/replay_session.py logs/sessions/20261023_120000.jsonl            # original speed
    python scripts/replay_session.py logs/sessions/20261023_120000.jsonl --rate 2   # twice as fast
    python scripts/replay_session.py <file> --host 127.0.0.1 --port 47199           # a spare-port app

Each line is {"t": <receive time>, "msg": {...}} as written by rover_intent.session.SessionRecorder. The messages
are sent exactly as recorded; the gaps between them are reproduced (divided by --rate). Query messages were never
recorded, so a replay drives the app with the real inputs only (body poses, utterances, clutch, EEG, presets, toggles).
"""
import argparse
import json
import socket
import time


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


def replay(path, host="127.0.0.1", port=47100, rate=1.0, sleep=time.sleep, send=None):
    rows = load(path)
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
        t_prev = t
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--rate", type=float, default=1.0, help="playback speed multiplier (2 = twice as fast)")
    a = ap.parse_args()
    n = replay(a.file, a.host, a.port, a.rate)
    print(f"replayed {n} messages from {a.file} to {a.host}:{a.port} at {a.rate}x")


if __name__ == "__main__":
    main()

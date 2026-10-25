"""Laptop-side streamer: open the Muse once, push raw EEG to the VM hub, and (optionally) keep doing the clench job.

Only ONE process can hold the Muse. Today that is `clench run` (the live double-clench -> GripToggle detector). This
streamer REPLACES it: it opens the board, reads ~every 50 ms, and

  (a) streams raw EEG over TCP to the hub (--hub-host rover-vm --hub-port 47150) as newline-delimited JSON:
      first a header, then per-chunk frames (see stream_hub.py for the exact wire format). It reconnects forever with
      backoff if the hub is down and DROPS frames while disconnected, so a network blip never disturbs the clench path.
  (b) with --clench-profile, runs the SAME ClenchDetector as `clench run` on the same chunks and sends the same
      GripToggle UDP (--app-host/--app-port, default rover-vm:47100) on each double clench.

Windows: plain Python + numpy + brainflow (no scipy needed beyond clench.py's own numpy fallback; no pylsl).

  python -m rover_intent.eeg.stream_tx --board muse2 --hub-host rover-vm --hub-port 47150
  python -m rover_intent.eeg.stream_tx --board muse2 --hub-host rover-vm \
         --clench-profile data/profiles/clench_pranav.json --app-host rover-vm --app-port 47100
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

from rover_intent.eeg.clench import ClenchDetector, _read, open_board, toggle_packet
from rover_intent.eeg.clench import MUSE_CHANNELS
from rover_intent.eeg.stream_hub import LISTEN_PORT, encode_line, make_frame, make_header


class HubSender:
    """Background TCP sender to the hub. Reconnects forever with backoff; drops frames while disconnected.

    Never blocks or crashes the caller: the board-reading loop just calls submit(); a worker thread owns the socket.
    """

    def __init__(self, host: str, port: int, header: dict, max_queue=400):
        self.host, self.port = host, port
        self._header = header
        self._q: deque = deque()
        self._max = max_queue
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self.connected = False
        self.sent = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, frame: dict):
        if not self.connected:  # hub is down / not yet connected: drop rather than buffer unboundedly
            self.dropped += 1
            return
        with self._cv:
            if len(self._q) >= self._max:
                self._q.popleft()  # connected but the link is backing up: drop the OLDEST, keep the stream fresh
                self.dropped += 1
            self._q.append(frame)
            self._cv.notify()

    def _run(self):
        backoff = 0.2
        while not self._stop.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
            except OSError:
                self.connected = False
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, 3.0)
                continue
            backoff = 0.2
            try:
                sock.sendall(encode_line(self._header))  # fresh connection: header first
                self.connected = True
                while not self._stop.is_set():
                    with self._cv:
                        while not self._q and not self._stop.is_set():
                            self._cv.wait(0.5)
                        batch = list(self._q)
                        self._q.clear()
                    for frame in batch:
                        sock.sendall(encode_line(frame))
                        self.sent += 1
            except OSError:
                self.connected = False
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
            # loop: reconnect (hub came back). Frames submitted meanwhile may have been dropped by the cap.

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify()
        self._thread.join(timeout=2.0)


def stream_loop(board, fs, chans, sender: HubSender, *, detector: ClenchDetector | None = None,
                udp=None, app_addr=None, period=0.05, quiet=False, stop: threading.Event | None = None):
    """Read the board ~every `period` s: frame -> hub, and (if detector) clench -> UDP toggle. Returns on stop/Ctrl+C."""
    stop = stop or threading.Event()
    seq = 0
    n_toggles = 0
    last_print = 0.0
    score = 0.0
    try:
        while not stop.is_set():
            x = _read(board, chans)
            if x.shape[1]:
                now = time.time()  # ~ time of the LAST sample in this chunk
                sender.submit(make_frame(seq, now, x))
                seq += 1
                if detector is not None:
                    for _t in detector.push(x):
                        n_toggles += 1
                        if udp is not None:
                            udp.sendto(toggle_packet(time.time()), app_addr)
                        print(f"\nTOGGLE #{n_toggles} (sent)", flush=True)
                    score = detector.score
            if not quiet and time.time() - last_print > 1.0:
                bar = ""
                if detector is not None:
                    width = min(20, int(20 * score / (2 * detector.cfg.on_ratio)))
                    bar = f" clench {score:5.1f}x |{'#' * width:<20}|"
                print(f"\rfs={fs:g} sent={sender.sent} hub={'yes' if sender.connected else 'no '} "
                      f"dropped={sender.dropped} seq={seq}{bar}", end="", flush=True)
                last_print = time.time()
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    return seq, n_toggles


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=["muse2", "muse2-bled", "synthetic", "hub"], default="muse2")
    ap.add_argument("--serial", default="", help="BLED112 dongle port for muse2-bled (COM3, /dev/ttyACM0)")
    ap.add_argument("--mac", default="", help="Muse MAC address (optional)")
    ap.add_argument("--name", default="", help="Muse name, e.g. Muse-1A2B (optional)")
    ap.add_argument("--hub-host", default="rover-vm")
    ap.add_argument("--hub-port", type=int, default=LISTEN_PORT)
    ap.add_argument("--clench-profile", default=None, help="also run the clench detector and send GripToggle UDP")
    ap.add_argument("--app-host", default="rover-vm")
    ap.add_argument("--app-port", type=int, default=47100)
    ap.add_argument("--period", type=float, default=0.05, help="board read period in seconds (~chunk cadence)")
    args = ap.parse_args(argv)

    board, fs, chans = open_board(args.board, args.serial, args.mac, args.name)
    channels = MUSE_CHANNELS[:len(chans)] if len(chans) <= 4 else list(chans)
    header = make_header(fs, channels, args.board, time.time())
    sender = HubSender(args.hub_host, args.hub_port, header)
    sender.start()

    detector = udp = app_addr = None
    if args.clench_profile:
        profile = json.loads(Path(args.clench_profile).read_text())
        if abs(fs - profile["fs"]) > 1:
            print(f"warning: board runs at {fs} Hz but the profile was made at {profile['fs']} Hz", file=sys.stderr)
        detector = ClenchDetector.from_profile(profile, t0=time.time())
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        app_addr = (args.app_host, args.app_port)
        print(f"clench enabled: double-clench to toggle, threshold {detector.cfg.on_ratio}x relaxed.", flush=True)

    print(f"streaming raw EEG to hub {args.hub_host}:{args.hub_port}. Ctrl+C to stop.", flush=True)
    try:
        stream_loop(board, fs, chans, sender, detector=detector, udp=udp, app_addr=app_addr, period=args.period)
    finally:
        sender.stop()
        board.stop_stream()
        board.release_session()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

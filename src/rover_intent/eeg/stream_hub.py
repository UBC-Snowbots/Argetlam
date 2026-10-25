"""VM-side hub for live raw Muse EEG streamed from the laptop (see src/rover_intent/eeg/stream_tx.py).

One sender (the laptop) connects over TCP to --listen 47150 and pushes newline-delimited JSON: first a header
  {"kind":"eeg_header","fs":256,"channels":["TP9","AF7","AF8","TP10"],"board":...,"t0":...}
then frames
  {"kind":"eeg","seq":n,"t":<sender time.time() of the LAST sample>,"x":[[ch0..],[ch1..],[ch2..],[ch3..]]}

The hub relays the header + every frame to ANY number of LOCAL consumers on 127.0.0.1:47152 (late joiners get the
header first). A new sender on 47150 REPLACES the old one. VM-side tools read the stream with `--board hub`, which uses
HubBoard below: it speaks the subset of BrainFlow's BoardShim the tools call, so `clench run --board hub`,
`muse_session`, `muse_detect` etc. run on live laptop data with no code changes.

Stdlib only (no numpy needed for the server; HubBoard uses numpy, same as the rest of eeg/).

  python -m rover_intent.eeg.stream_hub                         # listen 47150, fan out on 127.0.0.1:47152
  python -m rover_intent.eeg.stream_hub --record data/eeg_sessions   # also append raw frames to <dir>/<ts>/stream.jsonl
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

# --- ports (do NOT reuse the live-system ports; see the repo's hard rules) -----------------------------------------
LISTEN_PORT = 47150  # sender (laptop) -> hub
FANOUT_PORT = 47152  # hub -> local consumers (127.0.0.1 only)


# --- wire framing (shared with stream_tx and HubBoard) --------------------------------------------------------------
def encode_line(obj: dict) -> bytes:
    """One newline-delimited JSON record."""
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


def make_header(fs, channels, board, t0) -> dict:
    return {"kind": "eeg_header", "fs": fs, "channels": list(channels), "board": board, "t0": t0}


def make_frame(seq: int, t: float, x) -> dict:
    # x: (n_ch, n) -> list of per-channel sample lists (floats, microvolts as BrainFlow gives them)
    return {"kind": "eeg", "seq": seq, "t": t, "x": [list(map(float, row)) for row in x]}


class LineReader:
    """Buffered newline-delimited JSON reader over a blocking socket. Iterating yields decoded dicts until EOF."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""

    def __iter__(self):
        while True:
            nl = self.buf.find(b"\n")
            if nl < 0:
                try:
                    chunk = self.sock.recv(65536)
                except OSError:
                    return
                if not chunk:
                    return
                self.buf += chunk
                continue
            line, self.buf = self.buf[:nl], self.buf[nl + 1:]
            if line.strip():
                try:
                    yield json.loads(line.decode("utf-8"))
                except ValueError:
                    continue  # skip a corrupt line rather than drop the stream


# --- hub server -----------------------------------------------------------------------------------------------------
class Hub:
    """TCP hub: one sender on listen_port, fan-out to many consumers on 127.0.0.1:fanout_port."""

    def __init__(self, listen_host="0.0.0.0", listen_port=LISTEN_PORT, fanout_port=FANOUT_PORT,
                 record_dir: str | None = None):
        self.listen_host, self.listen_port, self.fanout_port = listen_host, listen_port, fanout_port
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._consumers: list[socket.socket] = []
        self._header_line: bytes | None = None
        self._sender: socket.socket | None = None
        self._sender_gen = 0  # bumped each time a new sender replaces the old one
        # stats
        self.fs = None
        self.frames = self.samples = self.gaps = 0
        self._last_seq = None
        self._last_t = None  # sender time.time() of the most recent frame's last sample
        self._last_recv = None  # hub time.time() when that frame arrived
        self._rec_fp = None
        self._record_root = Path(record_dir) if record_dir else None
        self._srv = self._fan = None

    # listening sockets ------------------------------------------------------------------------------------------
    def bind_listen(self) -> int:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.listen_host, self.listen_port))
        self._srv.listen(8)
        self.listen_port = self._srv.getsockname()[1]
        return self.listen_port

    def bind_fanout(self) -> int:
        self._fan = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._fan.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._fan.bind(("127.0.0.1", self.fanout_port))
        self._fan.listen(16)
        self.fanout_port = self._fan.getsockname()[1]
        return self.fanout_port

    def start(self):
        if self._srv is None:
            self.bind_listen()
        if self._fan is None:
            self.bind_fanout()
        if self._record_root is not None:
            self._open_record()
        threading.Thread(target=self._accept_senders, daemon=True).start()
        threading.Thread(target=self._accept_consumers, daemon=True).start()

    def _open_record(self):
        root = self._record_root
        # data must stay under the gitignored data/ tree
        parts = root.resolve().parts
        if "data" not in parts:
            raise ValueError(f"--record dir must be under data/ (got {root})")
        d = root / time.strftime("%Y%m%d_%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        self._rec_fp = open(d / "stream.jsonl", "ab")
        print(f"recording raw frames -> {d / 'stream.jsonl'}", flush=True)

    # sender side ------------------------------------------------------------------------------------------------
    def _accept_senders(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except OSError:
                return
            with self._lock:
                self._sender_gen += 1
                gen = self._sender_gen
                old, self._sender = self._sender, conn
                # a new sender replaces the old: reset per-sender stats
                self._header_line = None
                self._last_seq = None
            if old is not None:
                try:
                    old.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                old.close()
            threading.Thread(target=self._read_sender, args=(conn, gen, addr), daemon=True).start()

    def _read_sender(self, conn: socket.socket, gen: int, addr):
        conn.settimeout(None)
        for obj in LineReader(conn):
            with self._lock:
                if gen != self._sender_gen:
                    break  # superseded by a newer sender
            kind = obj.get("kind")
            if kind == "eeg_header":
                self._on_header(obj)
            elif kind == "eeg":
                self._on_frame(obj)
        with self._lock:
            if gen == self._sender_gen:
                self._sender = None
        conn.close()

    def _on_header(self, obj: dict):
        line = encode_line(obj)
        with self._lock:
            self._header_line = line
            self.fs = obj.get("fs")
            consumers = list(self._consumers)
        if self._rec_fp is not None:
            self._rec_fp.write(line)
        self._broadcast(line, consumers)

    def _on_frame(self, obj: dict):
        line = encode_line(obj)
        now = time.time()
        with self._lock:
            self.frames += 1
            x = obj.get("x") or []
            n = len(x[0]) if x else 0
            self.samples += n
            seq = obj.get("seq")
            if self._last_seq is not None and seq is not None and seq != self._last_seq + 1:
                self.gaps += 1
            self._last_seq = seq
            self._last_t = obj.get("t")
            self._last_recv = now
            consumers = list(self._consumers)
        if self._rec_fp is not None:
            self._rec_fp.write(line)
        self._broadcast(line, consumers)

    # consumer side ----------------------------------------------------------------------------------------------
    def _accept_consumers(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._fan.accept()
            except OSError:
                return
            conn.settimeout(5.0)
            with self._lock:
                header = self._header_line
                self._consumers.append(conn)
            if header is not None:
                try:
                    conn.sendall(header)  # late joiner gets the current header first
                except OSError:
                    self._drop_consumer(conn)

    def _broadcast(self, line: bytes, consumers: list[socket.socket]):
        for c in consumers:
            try:
                c.sendall(line)
            except OSError:
                self._drop_consumer(c)

    def _drop_consumer(self, conn: socket.socket):
        with self._lock:
            if conn in self._consumers:
                self._consumers.remove(conn)
        try:
            conn.close()
        except OSError:
            pass

    # stats ------------------------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            lat = None
            if self._last_t is not None and self._last_recv is not None:
                lat = self._last_recv - self._last_t
            return {"sender": self._sender is not None, "fs": self.fs, "frames": self.frames,
                    "samples": self.samples, "gaps": self.gaps, "consumers": len(self._consumers),
                    "latency": lat}

    def stop(self):
        self._stop.set()
        for s in (self._srv, self._fan):
            try:
                s and s.close()
            except OSError:
                pass
        with self._lock:
            consumers, sender = list(self._consumers), self._sender
            self._consumers = []
        for c in consumers + ([sender] if sender else []):
            try:
                c.close()
            except OSError:
                pass
        if self._rec_fp is not None:
            self._rec_fp.close()
            self._rec_fp = None


# --- VM-side board that reads the hub fan-out (BrainFlow BoardShim subset) ------------------------------------------
class HubBoard:
    """A BrainFlow-BoardShim-shaped board that pulls live EEG from the hub fan-out at 127.0.0.1:FANOUT_PORT.

    Exposes the calls the tools use: prepare_session / start_stream / get_board_data / stop_stream / release_session.
    get_board_data() returns everything received since the last call as a (4, n) float array (microvolts), in header
    channel order. A background thread reads frames and reconnects if the hub restarts.
    """

    def __init__(self, host="127.0.0.1", port=FANOUT_PORT, connect_timeout=10.0):
        self.host, self.port, self.connect_timeout = host, port, connect_timeout
        self.fs = None
        self.channels = None
        self._chunks: list = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._header_evt = threading.Event()
        self._thread = None

    def prepare_session(self):
        import numpy as np  # noqa: F401  (imported lazily, like the rest of eeg/)
        self._stop.clear()
        self._header_evt.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        if not self._header_evt.wait(self.connect_timeout):
            self._stop.set()
            raise RuntimeError(f"no EEG header from hub at {self.host}:{self.port} within "
                               f"{self.connect_timeout:.0f}s (is stream_hub running and a sender connected?)")

    def start_stream(self):
        return  # the reader thread is already running; buffering starts at prepare_session

    def _reader(self):
        import numpy as np
        backoff = 0.2
        while not self._stop.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
            except OSError:
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, 2.0)
                continue
            backoff = 0.2
            sock.settimeout(1.0)
            try:
                for obj in LineReader(sock):
                    if self._stop.is_set():
                        break
                    kind = obj.get("kind")
                    if kind == "eeg_header":
                        self.fs = obj.get("fs")
                        self.channels = obj.get("channels")
                        self._header_evt.set()
                    elif kind == "eeg":
                        arr = np.asarray(obj.get("x", []), dtype=float)
                        if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] > 0:
                            with self._lock:
                                self._chunks.append(arr)
            except socket.timeout:
                continue  # no data yet; loop and keep the connection
            except OSError:
                pass
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
            # fell out: hub closed or errored. Reconnect (the hub may have restarted).

    def get_board_data(self):
        import numpy as np
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros((4, 0))
        return np.concatenate(chunks, axis=1)

    def stop_stream(self):
        return

    def release_session(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# --- CLI ------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, default=LISTEN_PORT, help="TCP port for the laptop sender (0.0.0.0)")
    ap.add_argument("--listen-host", default="0.0.0.0")
    ap.add_argument("--fanout-port", type=int, default=FANOUT_PORT, help="local fan-out port (127.0.0.1)")
    ap.add_argument("--record", default=None, help="append raw frames to <dir>/<ts>/stream.jsonl (must be under data/)")
    args = ap.parse_args(argv)

    hub = Hub(args.listen_host, args.listen, args.fanout_port, args.record)
    lp, fp = hub.bind_listen(), hub.bind_fanout()
    hub.start()
    print(f"hub: sender on {args.listen_host}:{lp}, fan-out on 127.0.0.1:{fp}. Ctrl+C to stop.", flush=True)
    prev_f = prev_s = 0
    prev_t = time.time()
    try:
        while True:
            time.sleep(1.0)
            s = hub.snapshot()
            now = time.time()
            dt = max(now - prev_t, 1e-6)
            fps, sps = (s["frames"] - prev_f) / dt, (s["samples"] - prev_s) / dt
            prev_f, prev_s, prev_t = s["frames"], s["samples"], now
            lat = f"{s['latency'] * 1000:.0f}ms" if s["latency"] is not None else "n/a"
            print(f"sender={'yes' if s['sender'] else 'no '} fs={s['fs']} "
                  f"frames/s={fps:.0f} samples/s={sps:.0f} gaps={s['gaps']} "
                  f"consumers={s['consumers']} lat~{lat}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

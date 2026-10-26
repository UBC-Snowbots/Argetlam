"""Live raw-EEG streaming path: laptop stream_tx -> VM stream_hub -> HubBoard (--board hub).

No hardware, no network beyond localhost. Spare ports only (47950-47959); never the live-system ports.
"""
import json
import socket
import threading
import time

import numpy as np
import pytest

from rover_intent.eeg.stream_hub import (
    Hub,
    HubBoard,
    LineReader,
    encode_line,
    make_frame,
    make_header,
)
from rover_intent.eeg.stream_tx import HubSender, stream_loop

# Tests bind OS-chosen high ports (never the live-system ports 47100/.../47150/47152/47195, and never the
# 47950-47959 spare range the brief reserves). See _free_hub below.


def _free_hub():
    """A hub bound to OS-chosen high ports, started. Returns (hub, listen_port, fanout_port)."""
    hub = Hub("127.0.0.1", 0, 0)
    lp, fp = hub.bind_listen(), hub.bind_fanout()
    hub.start()
    return hub, lp, fp


class ConstBoard:
    """Fake BoardShim: get_board_data() returns int(fs*dt) new samples; channel i = i*1000 + running index.

    Distinct per-channel offsets + a globally increasing index let a reader verify channel ORDER and that no sample
    was dropped, duplicated or reordered end to end.
    """

    def __init__(self, fs=256.0, n_ch=4):
        self.fs, self.n_ch = fs, n_ch
        self._last = None
        self._k = 0

    def prepare_session(self):
        self._last = time.time()

    def start_stream(self):
        self._last = time.time()

    def get_board_data(self):
        now = time.time()
        if self._last is None:
            self._last = now
            return np.zeros((self.n_ch, 0))
        k = int(self.fs * (now - self._last))
        if k <= 0:
            return np.zeros((self.n_ch, 0))
        self._last = now
        idx = np.arange(self._k, self._k + k, dtype=float)
        self._k += k
        return np.stack([i * 1000.0 + idx for i in range(self.n_ch)])

    def stop_stream(self):
        pass

    def release_session(self):
        pass


class ReplayBoard:
    """Fake BoardShim replaying a fixed (4, N) signal in chunks, then signalling `done`."""

    def __init__(self, signal, chunk=32):
        self.signal, self.chunk = signal, chunk
        self.i = 0
        self.done = threading.Event()

    def prepare_session(self):
        pass

    def start_stream(self):
        pass

    def get_board_data(self):
        if self.i >= self.signal.shape[1]:
            self.done.set()
            return np.zeros((self.signal.shape[0], 0))
        j = min(self.i + self.chunk, self.signal.shape[1])
        out = self.signal[:, self.i:j]
        self.i = j
        return out

    def stop_stream(self):
        pass

    def release_session(self):
        pass


class DummySender:
    """Minimal sender stub for exercising stream_loop without a hub."""

    def __init__(self):
        self.frames = []
        self.sent = 0
        self.dropped = 0
        self.connected = True

    def submit(self, frame):
        self.frames.append(frame)
        self.sent += 1

    def stop(self):
        pass


# --- framing --------------------------------------------------------------------------------------------------------
def test_framing_round_trip():
    hdr = make_header(256, ["TP9", "AF7", "AF8", "TP10"], "muse2", 123.5)
    x = np.arange(12, dtype=float).reshape(4, 3) + np.array([[0], [1000], [2000], [3000]])
    frame = make_frame(7, 99.25, x)
    a, b = socket.socketpair()
    a.sendall(encode_line(hdr))
    a.sendall(encode_line(frame))
    a.close()
    got = list(LineReader(b))
    b.close()
    assert got[0] == hdr
    assert got[0]["channels"] == ["TP9", "AF7", "AF8", "TP10"]  # order preserved
    assert got[1]["seq"] == 7 and got[1]["t"] == 99.25
    assert np.array_equal(np.asarray(got[1]["x"]), x)


# --- end to end: stream_tx -> hub -> HubBoard -----------------------------------------------------------------------
def test_end_to_end_sample_count_rate_and_channel_order():
    fs = 256.0
    hub, lp, fp = _free_hub()
    board = ConstBoard(fs)
    board.prepare_session()
    sender = HubSender("127.0.0.1", lp, make_header(fs, ["TP9", "AF7", "AF8", "TP10"], "fake", time.time()))
    sender.start()
    hb = HubBoard("127.0.0.1", fp)
    hb.prepare_session()  # waits for the header relayed through the hub
    assert hb.fs == fs and hb.channels == ["TP9", "AF7", "AF8", "TP10"]

    stop = threading.Event()
    th = threading.Thread(target=stream_loop,
                          args=(board, fs, [0, 1, 2, 3], sender),
                          kwargs=dict(period=0.02, quiet=True, stop=stop), daemon=True)
    t0 = time.time()
    th.start()
    time.sleep(1.0)
    stop.set()
    th.join(timeout=2.0)
    dur = time.time() - t0

    chunks = []
    for _ in range(10):  # drain anything still in flight
        c = hb.get_board_data()
        if c.shape[1]:
            chunks.append(c)
        time.sleep(0.05)
    x = np.concatenate(chunks, axis=1)
    hb.release_session()
    sender.stop()
    hub.stop()

    assert x.shape[0] == 4
    # rate: roughly fs * duration samples
    assert fs * dur * 0.5 < x.shape[1] < fs * dur * 1.5, (x.shape[1], fs * dur)
    # channel order + contiguity: row i == row 0 + i*1000, and row 0 increases by exactly 1 each sample
    assert np.allclose(np.diff(x[0]), 1.0)
    for i in range(1, 4):
        assert np.allclose(x[i] - x[0], i * 1000.0)


def test_open_board_hub_kind(monkeypatch):
    """clench.open_board('hub') returns a HubBoard, its header fs, and channels [0,1,2,3]."""
    from rover_intent.eeg.clench import open_board
    fs = 256.0
    hub, lp, fp = _free_hub()
    sender = HubSender("127.0.0.1", lp, make_header(fs, ["TP9", "AF7", "AF8", "TP10"], "fake", time.time()))
    sender.start()
    # feed one frame so the header propagates
    board = ConstBoard(fs)
    board.prepare_session()
    stop = threading.Event()
    th = threading.Thread(target=stream_loop, args=(board, fs, [0, 1, 2, 3], sender),
                          kwargs=dict(period=0.02, quiet=True, stop=stop), daemon=True)
    th.start()
    monkeypatch.setenv("HUB_FANOUT_HOST", "127.0.0.1")
    monkeypatch.setenv("HUB_FANOUT_PORT", str(fp))
    try:
        hubboard, got_fs, chans = open_board("hub")
        assert got_fs == fs and chans == [0, 1, 2, 3]
        time.sleep(0.2)
        data = hubboard.get_board_data()
        assert data.shape[0] == 4
        hubboard.release_session()
    finally:
        stop.set()
        th.join(timeout=2.0)
        sender.stop()
        hub.stop()


# --- hub replaces a reconnecting sender -----------------------------------------------------------------------------
def test_hub_replaces_reconnecting_sender():
    hub, lp, fp = _free_hub()
    # a consumer that reads raw lines
    cons = socket.create_connection(("127.0.0.1", fp), timeout=2.0)
    cons.settimeout(2.0)

    def send_header_and_frame(board_name, seq):
        s = socket.create_connection(("127.0.0.1", lp), timeout=2.0)
        s.sendall(encode_line(make_header(256, ["TP9", "AF7", "AF8", "TP10"], board_name, time.time())))
        s.sendall(encode_line(make_frame(seq, time.time(), np.zeros((4, 2)))))
        return s

    a = send_header_and_frame("senderA", 0)
    time.sleep(0.2)
    b = send_header_and_frame("senderB", 100)  # replaces A
    time.sleep(0.2)
    # B's frame must reach the consumer; A's late frame must not
    b.sendall(encode_line(make_frame(101, time.time(), np.ones((4, 2)))))
    time.sleep(0.1)
    try:
        a.sendall(encode_line(make_frame(1, time.time(), np.full((4, 2), 7.0))))
    except OSError:
        pass  # hub already closed A: fine
    time.sleep(0.2)

    reader = LineReader(cons)
    seqs, boards = [], []
    cons.settimeout(0.5)
    try:
        for obj in reader:
            if obj["kind"] == "eeg":
                seqs.append(obj["seq"])
            else:
                boards.append(obj["board"])
            if len(seqs) >= 3:
                break
    except (socket.timeout, OSError):
        pass
    snap = hub.snapshot()  # while B is still connected: closing it first would race the hub noticing the EOF
    cons.close()
    a.close()
    b.close()
    hub.stop()

    assert 101 in seqs  # B's post-replacement frame arrived
    assert 1 not in seqs  # A's frame after being replaced did NOT
    assert snap["sender"] is True


# --- late consumer gets the header first ----------------------------------------------------------------------------
def test_late_consumer_gets_header_first():
    hub, lp, fp = _free_hub()
    s = socket.create_connection(("127.0.0.1", lp), timeout=2.0)
    s.sendall(encode_line(make_header(256, ["TP9", "AF7", "AF8", "TP10"], "muse2", time.time())))
    for i in range(5):
        s.sendall(encode_line(make_frame(i, time.time(), np.zeros((4, 4)))))
    time.sleep(0.3)  # all sent before the consumer joins
    cons = socket.create_connection(("127.0.0.1", fp), timeout=2.0)
    cons.settimeout(2.0)
    first = next(iter(LineReader(cons)))
    cons.close()
    s.close()
    hub.stop()
    assert first["kind"] == "eeg_header" and first["channels"] == ["TP9", "AF7", "AF8", "TP10"]


# --- tx survives hub down then up -----------------------------------------------------------------------------------
def test_tx_survives_hub_down_then_up():
    fs = 256.0
    # pick a port, keep it closed for now
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    lp = probe.getsockname()[1]
    probe.close()

    sender = HubSender("127.0.0.1", lp, make_header(fs, ["TP9", "AF7", "AF8", "TP10"], "fake", time.time()))
    sender.start()
    board = ConstBoard(fs)
    board.prepare_session()
    stop = threading.Event()
    th = threading.Thread(target=stream_loop, args=(board, fs, [0, 1, 2, 3], sender),
                          kwargs=dict(period=0.02, quiet=True, stop=stop), daemon=True)
    th.start()
    time.sleep(0.4)  # hub is down: tx must not crash, frames dropped
    assert th.is_alive() and sender.connected is False
    assert sender.dropped > 0

    # bring the hub up on the same port
    hub = Hub("127.0.0.1", lp, 0)
    hub.bind_listen()
    fp = hub.bind_fanout()
    hub.start()
    hb = HubBoard("127.0.0.1", fp)
    hb.prepare_session()  # will get the header once tx reconnects
    time.sleep(0.6)
    got = hb.get_board_data()

    stop.set()
    th.join(timeout=2.0)
    hb.release_session()
    sender.stop()
    hub.stop()
    assert sender.connected is True and sender.sent > 0
    assert got.shape[0] == 4 and got.shape[1] > 0


# --- clench path still sends a toggle -------------------------------------------------------------------------------
def test_clench_path_sends_toggle_on_double_clench():
    from rover_intent.eeg.clench import ClenchDetector, calibrate
    from rover_intent.transport.udp import _MSG
    from rover_intent.types import GripToggle
    from test_clench import double, simulate  # reuse the existing synthetic-signal helpers

    fs = 256.0
    rest = simulate(20, blinks=np.arange(1.5, 20, 3.1), seed=1)
    clench = simulate(24, clenches=[(i * 3 + 0.8, 0.5) for i in range(8)], blinks=[2.2, 11.4], seed=2)
    chew = [(10 + 0.7 * i, 0.3, 50) for i in range(12)]
    other = simulate(20, talk=[(0.5, 9)], clenches=chew, seed=3)
    profile = calibrate(rest, clench, other, fs)

    sig = simulate(5, clenches=double(1.0), seed=10)  # a clean double clench
    det = ClenchDetector.from_profile(profile, t0=0.0)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(1.0)
    app_port = rx.getsockname()[1]
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    board = ReplayBoard(sig, chunk=32)
    sender = DummySender()
    stream_loop(board, fs, [0, 1, 2, 3], sender, detector=det, udp=udp, app_addr=("127.0.0.1", app_port),
                period=0.0, quiet=True, stop=board.done)

    data, _ = rx.recvfrom(4096)
    rx.close()
    udp.close()
    msg = _MSG.validate_python(json.loads(data))
    assert isinstance(msg, GripToggle) and msg.source == "clench"
    assert sender.sent > 0  # frames were also streamed while the clench job ran


def test_stream_loop_no_data_watchdog_reopens_a_frozen_board():
    """BrainFlow can freeze silently (Muse sleeps / BLE drop, 2026-10-25): no samples, no error. The watchdog must
    reopen the board and frames must resume."""
    import threading as _th
    import time as _t

    import numpy as _np

    from rover_intent.eeg.stream_tx import stream_loop

    class Board:
        def __init__(self, n_good):
            self.n_good = n_good

        def get_board_data(self):
            if self.n_good > 0:
                self.n_good -= 1
                return _np.ones((4, 12))
            return _np.zeros((4, 0))     # frozen: no data, no error

        def stop_stream(self):
            pass

        def release_session(self):
            pass

    class Sender:
        sent, dropped, connected = 0, 0, True

        def __init__(self):
            self.frames = []

        def submit(self, f):
            self.frames.append(f)

    sender, stop, opened = Sender(), _th.Event(), []

    def reopen():
        b = Board(10 ** 6)               # the reopened board streams again
        opened.append(b)
        return b, [0, 1, 2, 3]

    th = _th.Thread(target=stream_loop, args=(Board(3), 256, [0, 1, 2, 3], sender),
                    kwargs=dict(period=0.01, quiet=True, stop=stop, reopen=reopen, stall_s=0.3), daemon=True)
    th.start()
    t_end = _t.time() + 3.0
    while _t.time() < t_end and not (opened and len(sender.frames) > 10):
        _t.sleep(0.05)
    stop.set()
    th.join(timeout=2)
    assert len(opened) == 1               # reopened exactly once
    assert len(sender.frames) > 10        # frames resumed after the reconnect (3 before the freeze)

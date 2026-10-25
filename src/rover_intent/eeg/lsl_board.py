"""Read the Muse through an LSL stream (e.g. Petal Metrics) instead of BrainFlow's own Bluetooth connection.

Petal Metrics (petal.tech) connects to the Muse itself and publishes the EEG as a Lab Streaming Layer stream (type
"EEG", 256 Hz, TP9 AF7 AF8 TP10 plus AUX). `LslBoard` wraps that stream with the few BoardShim calls the Muse tools
use (prepare_session / start_stream / get_board_data / stop_stream / release_session), so `--board petal` plugs into
clench.py, sunday.py and muse_detect.py without touching the recording, fitting or live-test code.

get_board_data() returns everything received since the last call as a (channels, samples) array, like BrainFlow.
The channel order is TP9, AF7, AF8, TP10 (taken from the stream's channel labels when it has them, else the first 4),
which is the same order BrainFlow's MUSE_2_BOARD gives.
"""
from __future__ import annotations

import numpy as np

MUSE_LABELS = ("TP9", "AF7", "AF8", "TP10")


def _labels(info) -> list[str]:
    """Channel labels from the stream description (empty list if the stream does not describe them)."""
    out = []
    try:
        ch = info.desc().child("channels").child("channel")
        while not ch.empty():
            out.append(ch.child_value("label") or ch.child_value("name"))
            ch = ch.next_sibling()
    except Exception:
        return []
    return out


def pick_channels(labels: list[str], n_channels: int) -> list[int]:
    """Indices of TP9 AF7 AF8 TP10 in the stream; falls back to the first 4 channels if the labels don't match."""
    up = [str(x).upper() for x in labels]
    if all(m in up for m in MUSE_LABELS):
        return [up.index(m) for m in MUSE_LABELS]
    if n_channels < 4:
        raise RuntimeError(f"LSL EEG stream has {n_channels} channels; the Muse tools need 4 (TP9 AF7 AF8 TP10)")
    return [0, 1, 2, 3]


def choose_stream(streams):
    """Prefer a stream whose name mentions Petal (or Muse), else the first EEG stream found."""
    for key in ("petal", "muse"):
        for s in streams:
            if key in s.name().lower():
                return s
    return streams[0]


class LslBoard:
    """BoardShim look-alike over an LSL inlet. Data rows are already reordered to TP9 AF7 AF8 TP10."""

    def __init__(self, info, inlet, picks: list[int]):
        self.info, self.inlet, self.picks = info, inlet, picks

    def prepare_session(self) -> None:   # BrainFlow API parity; the inlet is already open
        pass

    def start_stream(self) -> None:
        pass

    def get_board_data(self) -> np.ndarray:
        rows = []
        while True:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=4096)
            if not chunk:
                break
            rows.append(np.asarray(chunk, dtype=float))
        if not rows:
            return np.zeros((len(self.picks), 0))
        return np.concatenate(rows, axis=0)[:, self.picks].T

    def stop_stream(self) -> None:
        pass

    def release_session(self) -> None:
        try:
            self.inlet.close_stream()
        except Exception:
            pass


def open_lsl(stream_type: str = "EEG", timeout: float = 10.0):
    """Find the EEG stream on the local network and open it. Returns (board, fs, chans) like clench.open_board;
    chans is [0, 1, 2, 3] because LslBoard already returns only the 4 Muse channels in order."""
    try:
        import pylsl
    except ImportError as e:
        raise RuntimeError("--board petal needs pylsl: pip install pylsl (it is in the [muse] extra)") from e
    streams = pylsl.resolve_byprop("type", stream_type, timeout=timeout)
    if not streams:
        raise RuntimeError(f"no LSL stream of type '{stream_type}' found in {timeout:.0f} s. In Petal Metrics: "
                           "connect the Muse, pick LSL as the streaming protocol and press Stream, then retry.")
    info = choose_stream(streams)
    inlet = pylsl.StreamInlet(info, max_buflen=60, recover=True)
    full = inlet.info(timeout=5.0)
    picks = pick_channels(_labels(full), full.channel_count())
    fs = int(round(full.nominal_srate())) or 256
    board = LslBoard(full, inlet, picks)
    board.get_board_data()   # drop anything buffered before the session starts
    return board, fs, [0, 1, 2, 3]


def main(argv=None) -> int:
    """Local check before a session: find the Petal/LSL EEG stream and print per-channel signal level for a while.
    Clench your jaw during it: the numbers should jump (EMG). `python -m rover_intent.eeg.lsl_board [--seconds 15]`"""
    import argparse
    import time

    ap = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--type", default="EEG", help="LSL stream type to look for (Petal Metrics uses EEG)")
    a = ap.parse_args(argv)
    try:
        board, fs, _ = open_lsl(a.type)
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1
    print(f"OK: stream '{board.info.name()}' type {board.info.type()}, {board.info.channel_count()} channels "
          f"at {fs} Hz; using {', '.join(MUSE_LABELS)} = stream channels {board.picks}")
    print("signal level (RMS, stream units, usually microvolts) every 0.5 s; clench your jaw to see it jump:")
    print("   " + "  ".join(f"{m:>6}" for m in MUSE_LABELS) + "   samples")
    t_end, total = time.time() + a.seconds, 0
    try:
        while time.time() < t_end:
            time.sleep(0.5)
            x = board.get_board_data()
            total += x.shape[1]
            if x.shape[1]:
                rms = np.sqrt(np.mean((x - x.mean(axis=1, keepdims=True)) ** 2, axis=1))
                print("   " + "  ".join(f"{v:6.1f}" for v in rms) + f"   {x.shape[1]:5d}", flush=True)
            else:
                print("   (no samples in the last 0.5 s: is Petal still streaming?)", flush=True)
    finally:
        board.release_session()
    rate = total / a.seconds
    ok = rate > 0.8 * fs
    print(f"{'OK' if ok else 'WARN'}: received {rate:.0f} samples/s (expected ~{fs})")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

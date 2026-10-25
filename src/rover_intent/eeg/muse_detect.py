"""Live Muse detectors: SSVEP (CCA / FBCCA, optional per-user TRCA) and alpha (eyes-closed), with run / replay.

Emits a one-shot confirm/grab event over UDP in the existing message style (rover_intent.types):
  - GripToggle (default): the event the arbiter already consumes (eeg/clench.py uses the same one). Zero app changes.
  - EEGSelect (--emit eeg_select): a richer event that also says WHICH SSVEP target was chosen and the confidence,
    for a multi-target menu. The arbiter does not consume it yet; docs/muse_readiness.md shows how it would wire in.

Recommended use (see docs/muse_readiness.md): SSVEP as "look at the confirm box to grab", GATED by a grab candidate
(pass --gated and feed the gate), with alpha "close your eyes to confirm" as the fallback when SSVEP is too weak on
the dry temporal electrodes. On a Muse there is no occipital electrode, so expect this to be weak unless a posterior
AUX electrode is added; the offline study quantifies it.

Core is pure/streaming so tests drive it with synthetic signals. Standalone on the laptop, like eeg/clench.py:
  python -m rover_intent.eeg.muse_detect run --detector ssvep --freqs 12,15 --board muse2 --host <vm>
  python -m rover_intent.eeg.muse_detect run --detector alpha --profile data/profiles/muse_pranav.json --host <vm>
  python -m rover_intent.eeg.muse_detect replay --detector ssvep --freqs 12,15 --file data/eeg_recordings/session.npz
  --board muse2|muse2-bled|synthetic   --emit grip_toggle|eeg_select   --thr 0.25 --margin 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rover_intent.eeg.alpha import AlphaConfig, AlphaDetector
from rover_intent.eeg.ssvep import TRCA, cca_scores, fbcca_scores, notch_mains, reference_signals


@dataclass
class SsvepStream:
    """Sliding-window SSVEP decision. push(chunk (n_ch, n)) -> list of (t, freq, confidence) for fired decisions.
    Fires when the top candidate's score exceeds `thr` and leads the runner-up by `margin`, held past a refractory."""
    freqs: tuple
    fs: float
    win_s: float = 3.0
    step_s: float = 0.25
    method: str = "fbcca"           # cca | fbcca | trca
    harmonics: int = 5
    thr: float = 0.25               # on the top score (CCA/FBCCA rho; TRCA correlation)
    margin: float = 0.02            # top minus runner-up
    refractory_s: float = 2.0
    confirm_freq: float | None = None   # if set, only this target fires a confirm/grab
    trca: TRCA | None = None
    notch: bool = True                  # remove 50/60 Hz mains before scoring (live EEG)
    t0: float = 0.0

    def __post_init__(self):
        self.win = int(self.win_s * self.fs)
        self.step = int(self.step_s * self.fs)
        self.buf = None
        self.n = 0
        self._next = self.win
        self._refractory_until = -np.inf
        self._refs = reference_signals(self.freqs, self.fs, self.win, self.harmonics)
        self.last_scores = np.zeros(len(self.freqs))

    def _score(self, X):
        if self.notch:
            X = notch_mains(X, self.fs)
        if self.method == "trca" and self.trca is not None:
            return self.trca.scores(X)
        if self.method == "cca":
            return cca_scores(X, self.freqs, self.fs, self.harmonics, refs=self._refs)
        return fbcca_scores(X, self.freqs, self.fs, self.harmonics, refs=self._refs)

    def push(self, chunk: np.ndarray) -> list:
        chunk = np.asarray(chunk, float)
        if chunk.shape[1] == 0:
            return []
        self.buf = chunk if self.buf is None else np.concatenate([self.buf, chunk], axis=1)
        if self.buf.shape[1] > self.win:  # keep only the last window
            self.buf = self.buf[:, -self.win:]
        self.n += chunk.shape[1]
        out = []
        while self.n >= self._next and self.buf.shape[1] >= self.win:
            t = self.t0 + self._next / self.fs
            self._next += self.step
            sc = self._score(self.buf[:, -self.win:])
            self.last_scores = sc
            order = np.argsort(sc)[::-1]
            top, run = sc[order[0]], (sc[order[1]] if len(sc) > 1 else -np.inf)
            if top >= self.thr and (top - run) >= self.margin and t >= self._refractory_until:
                f = float(self.freqs[order[0]])
                if self.confirm_freq is None or abs(f - self.confirm_freq) < 1e-6:
                    out.append((t, f, float(top)))
                    self._refractory_until = t + self.refractory_s
        return out


@dataclass
class AlphaStream:
    """Sliding-window wrapper around AlphaDetector. push(chunk) -> list of (t,) when a confirm fires."""
    fs: float
    threshold: float
    win_s: float = 2.0
    step_s: float = 0.25
    cfg: AlphaConfig = field(default_factory=AlphaConfig)
    t0: float = 0.0

    def __post_init__(self):
        self.win = int(self.win_s * self.fs)
        self.step = int(self.step_s * self.fs)
        self.buf = None
        self.n = 0
        self._next = self.win
        self.det = AlphaDetector(self.fs, self.threshold, self.cfg, self.t0)
        self.metric = 0.0

    def push(self, chunk: np.ndarray) -> list:
        chunk = np.asarray(chunk, float)
        if chunk.shape[1] == 0:
            return []
        self.buf = chunk if self.buf is None else np.concatenate([self.buf, chunk], axis=1)
        if self.buf.shape[1] > self.win:
            self.buf = self.buf[:, -self.win:]
        self.n += chunk.shape[1]
        out = []
        while self.n >= self._next and self.buf.shape[1] >= self.win:
            t = self.t0 + self._next / self.fs
            self._next += self.step
            if self.det.push_window(self.buf[:, -self.win:], t):
                out.append((t,))
            self.metric = self.det.metric
        return out


# --- event emission --------------------------------------------------------------------------------------------------
def grip_toggle_packet(t: float, source: str) -> bytes:
    return json.dumps({"kind": "grip_toggle", "t": t, "source": source}).encode()


def eeg_select_packet(t: float, label: str, freq: float, confidence: float, source: str) -> bytes:
    return json.dumps({"kind": "eeg_select", "t": t, "label": label, "freq": freq,
                       "confidence": round(confidence, 4), "source": source}).encode()


def _emit(tx, host, port, emit_kind, source, t, freq=0.0, conf=0.0, label="grab"):
    if tx is None:
        return
    pkt = (eeg_select_packet(t, label, freq, conf, source) if emit_kind == "eeg_select"
           else grip_toggle_packet(t, source))
    tx.sendto(pkt, (host, port))


# --- CLI -------------------------------------------------------------------------------------------------------------
def _make_streams(args, fs, profile):
    freqs = tuple(float(x) for x in args.freqs.split(",")) if args.freqs else ()
    streams = {}
    if args.detector in ("ssvep", "both"):
        trca = None
        if args.method == "trca" and profile and profile.get("trca"):
            trca = _load_trca(profile["trca"], fs)
        streams["ssvep"] = SsvepStream(freqs, fs, args.win, args.step, args.method, args.harmonics,
                                       args.thr, args.margin, confirm_freq=args.confirm_freq, trca=trca)
    if args.detector in ("alpha", "both"):
        thr = profile.get("alpha_threshold") if profile else None
        thr = thr if thr is not None else args.alpha_thr
        streams["alpha"] = AlphaStream(fs, thr, args.alpha_win, args.step)
    return streams


def _load_trca(d, fs):
    t = TRCA(fs, ensemble=d.get("ensemble", True))
    t.classes_ = np.array(d["classes"])
    t.filters_ = np.array(d["filters"])
    t.templates_ = [np.array(x) for x in d["templates"]]
    return t


def cmd_run(args):
    import socket
    from rover_intent.eeg.clench import _read, open_board
    profile = json.loads(Path(args.profile).read_text()) if args.profile else {}
    board, fs, chans = open_board(args.board, args.serial, args.mac, args.name)
    streams = _make_streams(args, fs, profile)
    tx = None if args.dry else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"detectors: {list(streams)} fs={fs} emit={args.emit}. Ctrl+C to stop.", flush=True)
    n = 0
    try:
        while True:
            x = _read(board, chans)
            if x.shape[1]:
                for name, st in streams.items():
                    for ev in st.push(x):
                        n += 1
                        t = time.time()
                        if name == "ssvep":
                            _, f, conf = ev
                            label = "grab" if (args.confirm_freq is None or abs(f - args.confirm_freq) < 1e-6) else "cancel"
                            _emit(tx, args.host, args.port, args.emit, "ssvep", t, f, conf, label)
                            print(f"\nSSVEP {f:g}Hz conf={conf:.2f} -> {label}" + ("" if tx else " (dry)"), flush=True)
                        else:
                            _emit(tx, args.host, args.port, args.emit, "alpha", t, 0.0, 1.0, "grab")
                            print("\nALPHA eyes-closed -> grab" + ("" if tx else " (dry)"), flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        board.stop_stream()
        board.release_session()
    return 0


def cmd_replay(args):
    rec = np.load(args.file, allow_pickle=True)
    fs = float(rec["fs"]) if "fs" in rec.files else 256.0
    x = rec["eeg"] if "eeg" in rec.files else rec[[k for k in ("data", "clench", "rest") if k in rec.files][0]]
    profile = json.loads(Path(args.profile).read_text()) if args.profile else {}
    streams = _make_streams(args, fs, profile)
    events = {name: [] for name in streams}
    for i in range(0, x.shape[1], 32):
        for name, st in streams.items():
            events[name] += st.push(x[:, i:i + 32])
    for name, evs in events.items():
        if name == "ssvep":
            print(json.dumps({"detector": "ssvep", "n_events": len(evs),
                              "events": [{"t_s": round(e[0], 2), "freq": e[1], "conf": round(e[2], 3)} for e in evs]}))
        else:
            print(json.dumps({"detector": "alpha", "n_events": len(evs),
                              "events_at_s": [round(e[0], 2) for e in evs]}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--detector", choices=["ssvep", "alpha", "both"], default="ssvep")
        p.add_argument("--freqs", default="12,15")
        p.add_argument("--win", type=float, default=3.0, help="SSVEP window (s)")
        p.add_argument("--alpha-win", type=float, default=2.0)
        p.add_argument("--step", type=float, default=0.25)
        p.add_argument("--method", choices=["cca", "fbcca", "trca"], default="fbcca")
        p.add_argument("--harmonics", type=int, default=5)
        p.add_argument("--thr", type=float, default=0.25)
        p.add_argument("--margin", type=float, default=0.02)
        p.add_argument("--alpha-thr", type=float, default=0.5)
        p.add_argument("--confirm-freq", type=float, default=None, help="only this SSVEP freq fires a grab")
        p.add_argument("--profile", default="", help="per-user profile json (alpha threshold, TRCA)")
        p.add_argument("--emit", choices=["grip_toggle", "eeg_select"], default="grip_toggle")
    r = sub.choices["run"]
    r.add_argument("--board", choices=["muse2", "muse2-bled", "synthetic"], default="muse2")
    r.add_argument("--serial", default="")
    r.add_argument("--mac", default="")
    r.add_argument("--name", default="")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=47100)
    r.add_argument("--dry", action="store_true")
    rp = sub.choices["replay"]
    rp.add_argument("--file", required=True)
    args = ap.parse_args(argv)
    args.profile = args.profile or ""
    return {"run": cmd_run, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Turn a Muse session recording (from eeg/muse_session.py) into a day-one readiness report:
  - SSVEP SNR per flicker frequency per channel (dB), from the cued trials
  - alpha ratio: eyes-closed vs eyes-open relative-alpha power, per channel and overall (the Berger effect)
  - SSVEP accuracy vs window length (CCA and FBCCA over the session's own frequencies)
  - alpha eyes-closed-vs-open balanced accuracy vs window length
  - frame-drop summary (from the recorder's frame log)

build_report(eeg, fs, markers, meta, frame_report) is pure so tests feed it a synthetic session. The CLI loads an npz.
  python -m rover_intent.eeg.muse_report --file data/eeg_recordings/session.npz [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rover_intent.eeg.alpha import AlphaConfig, AlphaDetector, rel_alpha
from rover_intent.eeg.ssvep import cca_scores, fbcca_scores, itr_bits_per_min, notch_mains, reference_signals

VIS_LATENCY = 0.14
SSVEP_WINDOWS = [0.5, 1.0, 2.0, 3.0]
ALPHA_WINDOWS = [0.5, 1.0, 2.0]


def ssvep_snr(x: np.ndarray, fs: float, freq: float, n_neighbors: int = 6) -> float:
    """SNR in dB at `freq`: power in the nearest FFT bin vs the mean of the surrounding bins (Tsinghua convention)."""
    n = x.shape[-1]
    if n < int(fs):
        return float("nan")
    X = np.abs(np.fft.rfft(x * np.hanning(n), axis=-1)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    k = int(np.argmin(np.abs(f - freq)))
    lo, hi = max(1, k - n_neighbors), min(len(f) - 1, k + n_neighbors + 1)
    neigh = np.concatenate([X[..., lo:k], X[..., k + 1:hi]], axis=-1)
    denom = neigh.mean(axis=-1)
    snr = X[..., k] / np.maximum(denom, 1e-20)
    return float(10 * np.log10(np.maximum(snr, 1e-20)))


def _phases_from_markers(markers, n_total, meta):
    """markers: array (m,2) of (sample_idx, code). Returns list of (kind, start, end, freq)."""
    freqs = [float(f) for f in meta["config"]["freqs"]]
    rows = sorted([(int(s), float(c)) for s, c in markers], key=lambda r: r[0])
    out = []
    for i, (s, c) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else n_total
        if c >= 100:
            idx = int(round(c - 100))
            kind, freq = "ssvep", (freqs[idx] if idx < len(freqs) else None)
        elif c == 10:
            kind, freq = "eyes_open", None
        elif c == 11:
            kind, freq = "eyes_closed", None
        else:
            kind, freq = "other", None
        out.append((kind, s, end, freq))
    return out


def _trial_segment(eeg, start, end, fs, win_s=None):
    s = start + int(VIS_LATENCY * fs)
    e = end if win_s is None else min(end, s + int(win_s * fs))
    return eeg[:, s:e]


def build_report(eeg, fs, markers, meta, frame_report=None) -> dict:
    eeg = np.asarray(eeg, float)
    if eeg.shape[1] > 64:
        eeg = notch_mains(eeg, fs)  # live EEG has mains; a 5th harmonic ref of 12/15 Hz lands on 60 Hz otherwise
    channels = meta.get("channels", [f"ch{i}" for i in range(eeg.shape[0])])
    freqs = [float(f) for f in meta["config"]["freqs"]]
    phases = _phases_from_markers(markers, eeg.shape[1], meta)
    rep = {"channels": channels, "freqs": freqs, "fs": fs,
           "n_ssvep_trials": sum(p[0] == "ssvep" for p in phases),
           "n_alpha_blocks": {"open": sum(p[0] == "eyes_open" for p in phases),
                              "closed": sum(p[0] == "eyes_closed" for p in phases)}}

    # --- SSVEP SNR per freq per channel ---
    snr = {f"{f:g}": {} for f in freqs}
    ssvep_trials = {f: [] for f in freqs}
    for kind, s, e, f in phases:
        if kind == "ssvep" and f is not None:
            seg = _trial_segment(eeg, s, e, fs)
            if seg.shape[1] >= int(fs):
                ssvep_trials[f].append(seg)
    for f in freqs:
        for ci, ch in enumerate(channels):
            vals = [ssvep_snr(seg[ci], fs, f) for seg in ssvep_trials[f] if seg.shape[1] >= int(fs)]
            vals = [v for v in vals if np.isfinite(v)]
            snr[f"{f:g}"][ch] = round(float(np.mean(vals)), 2) if vals else None
    rep["ssvep_snr_db"] = snr
    rep["ssvep_snr_best_channel"] = {
        f"{f:g}": (max(((snr[f"{f:g}"][c], c) for c in channels if snr[f"{f:g}"][c] is not None),
                       default=(None, None))[1]) for f in freqs}

    # --- SSVEP accuracy vs window ---
    rep["ssvep_accuracy"] = {}
    for method, fn in [("cca", cca_scores), ("fbcca", fbcca_scores)]:
        rep["ssvep_accuracy"][method] = {}
        for win in SSVEP_WINDOWS:
            n = int(win * fs)
            refs = reference_signals(freqs, fs, n, 5)
            correct, total = 0, 0
            for ti, f in enumerate(freqs):
                for kind, s, e, ff in phases:
                    if kind == "ssvep" and ff == f:
                        seg = _trial_segment(eeg, s, e, fs, win)
                        if seg.shape[1] < n:
                            continue
                        sc = fn(seg[:, :n], freqs, fs, 5, refs=refs)
                        correct += int(np.argmax(sc) == ti)
                        total += 1
            acc = correct / total if total else None
            rep["ssvep_accuracy"][method][str(win)] = {
                "acc": round(acc, 3) if acc is not None else None, "n": total,
                "itr_bits_per_min": round(itr_bits_per_min(acc, len(freqs), win), 2) if acc else None}

    # --- alpha ratio eyes-closed vs open ---
    open_wins = [eeg[:, s:e] for k, s, e, _ in phases if k == "eyes_open"]
    closed_wins = [eeg[:, s:e] for k, s, e, _ in phases if k == "eyes_closed"]
    rep["alpha"] = {}
    if open_wins and closed_wins:
        ro = np.array([rel_alpha(w, fs) for w in open_wins if w.shape[1] >= int(fs)])  # (blocks, n_ch)
        rc = np.array([rel_alpha(w, fs) for w in closed_wins if w.shape[1] >= int(fs)])
        rep["alpha"]["rel_alpha_open"] = {ch: round(float(ro[:, i].mean()), 4) for i, ch in enumerate(channels)}
        rep["alpha"]["rel_alpha_closed"] = {ch: round(float(rc[:, i].mean()), 4) for i, ch in enumerate(channels)}
        rep["alpha"]["closed_over_open_ratio"] = {
            ch: round(float(rc[:, i].mean() / max(ro[:, i].mean(), 1e-9)), 2) for i, ch in enumerate(channels)}
        rep["alpha"]["overall_ratio"] = round(float(rc.mean() / max(ro.mean(), 1e-9)), 2)
        # accuracy vs window: slice blocks into windows, calibrate threshold, balanced acc
        rep["alpha"]["accuracy"] = {}
        for win in ALPHA_WINDOWS:
            ow = _slice_blocks(open_wins, fs, win)
            cw = _slice_blocks(closed_wins, fs, win)
            if len(ow) >= 2 and len(cw) >= 2:
                det, st = AlphaDetector.calibrate(ow, cw, fs, AlphaConfig())
                mo = np.array([rel_alpha(w, fs).mean() for w in ow])
                mc = np.array([rel_alpha(w, fs).mean() for w in cw])
                bal = 0.5 * ((mo < det.threshold).mean() + (mc >= det.threshold).mean())
                rep["alpha"]["accuracy"][str(win)] = {"bal_acc": round(float(bal), 3),
                                                      "separation": round(st["separation"], 2)}
    rep["frame_report"] = frame_report or {}
    rep["verdict"] = _verdict(rep)
    return rep


def _slice_blocks(blocks, fs, win_s):
    n = int(win_s * fs)
    out = []
    for b in blocks:
        for i in range(0, b.shape[1] - n + 1, n):
            out.append(b[:, i:i + n])
    return out


def _verdict(rep: dict) -> dict:
    """Simple go / marginal / no-go per control, for the day-one check."""
    v = {}
    # SSVEP: best FBCCA accuracy at the longest window
    acc = rep.get("ssvep_accuracy", {}).get("fbcca", {})
    best = max((d["acc"] for d in acc.values() if d.get("acc") is not None), default=None)
    n_classes = len(rep["freqs"]) or 2
    chance = 1.0 / n_classes
    v["ssvep"] = ("good" if best and best >= 0.85 else "marginal" if best and best >= chance + 0.2 else "weak")
    v["ssvep_best_acc"] = best
    al = rep.get("alpha", {})
    ratio = al.get("overall_ratio")
    bacc = max((d["bal_acc"] for d in al.get("accuracy", {}).values()), default=None)
    v["alpha"] = ("good" if ratio and ratio >= 1.5 and bacc and bacc >= 0.8
                  else "marginal" if ratio and ratio >= 1.2 else "weak")
    v["alpha_ratio"] = ratio
    fr = rep.get("frame_report", {})
    v["display_ok"] = bool(fr) and fr.get("frac_dropped", 1) < 0.02
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    rec = np.load(args.file, allow_pickle=True)
    eeg = rec["eeg"]
    fs = float(rec["fs"])
    markers = rec["markers"] if "markers" in rec.files else np.zeros((0, 2))
    meta = json.loads(str(rec["meta"])) if "meta" in rec.files else {"config": {"freqs": []}}
    frame_report = json.loads(str(rec["frame_report"])) if "frame_report" in rec.files else {}
    rep = build_report(eeg, fs, markers, meta, frame_report)
    text = json.dumps(rep, indent=1)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

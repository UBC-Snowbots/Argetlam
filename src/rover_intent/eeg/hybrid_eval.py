"""Pseudo-online evaluation of the hybrid grasp decoder on BNCI 001-2020 (gel, 8 motor channels).

The decoder slides over continuous held-out EEG every 1/8 s (causal filters, as live), p_grasp goes through the real
Arbiter (hysteresis 0.7/0.3 + 0.5 s dwell). A grasp offset in the data counts as the user releasing (gripper forced open).
Per decoder (hybrid / stage1-only / stage2-only) we report:
  hit rate      trials with a gripper close in [grasp onset - 0.5 s, grasp onset + 1.5 s]
  latency       first close relative to grasp onset (median, s)
  premature     trials where it closed between movement onset - 1 s and grasp onset - 0.5 s
  false/min     closes outside every trial span [mo - 1 s, grasp offset + 2 s] and outside the eye task,
                split into rest blocks vs other idle time
Modes: calibrated = train on the first half of the subject's trials, test on the rest (chronological, no leakage);
       new = train on the other 14 subjects, test on this subject's whole session.

    python -m rover_intent.eeg.hybrid_eval --subjects 1-15 --out models/hybrid_eval.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
from pathlib import Path

import numpy as np

from ..control.arbiter import Arbiter
from ..planner.scene import Scene
from ..types import Action, Intent
from .grasp_experiments import GRASP_ONSET, load_full
from .hybrid import MOTOR_8, MRCP, MUBETA, HybridDecoder, car, sos_for, windows_at
from .reach_grasp import MOVE_ONSET

GRASP_OFFSET = (534562, 534563)
STEP = 32  # samples (1/8 s at 256 Hz)


def prepare(subject):
    from scipy.signal import sosfilt
    eeg, _eog, labels, fs, codes, pos = load_full(subject)
    x = eeg[[labels.index(c) for c in MOTOR_8]]
    mb = sosfilt(sos_for(MUBETA, fs), x, axis=1).astype(np.float32)
    mr = sosfilt(sos_for(MRCP, fs), car(x), axis=1).astype(np.float32)
    mo = np.sort(pos[np.isin(codes, MOVE_ONSET)])
    go_all, off_all = np.sort(pos[np.isin(codes, GRASP_ONSET)]), np.sort(pos[np.isin(codes, GRASP_OFFSET)])
    trials = []
    for m in mo:
        g = go_all[(go_all > m) & (go_all < m + 3 * fs)]
        if not len(g):
            continue
        o = off_all[(off_all > g[0]) & (off_all < g[0] + 15 * fs)]
        trials.append((m, g[0], o[0] if len(o) else g[0] + 2 * fs))
    others = np.sort(pos[~np.isin(codes, (768, 769))])
    eye = [(a, b) for on, off in ((10, 11), (12, 13), (14, 15))
           for a, b in zip(np.sort(pos[codes == on]), np.sort(pos[codes == off]))]
    rest = list(zip(np.sort(pos[codes == 768]), np.sort(pos[codes == 769])))
    del eeg, _eog
    return {"fs": fs, "mb": mb, "mr": mr, "trials": np.array(trials, int), "others": others, "eye": eye,
            "rest": rest, "n": mb.shape[1]}


def train_windows(S, trial_mask, t_lo=0, t_hi=None):
    """Window END indices for the two stages, from the selected trials (and idle/return periods between them)."""
    fs, n = S["fs"], S["n"]
    t_hi = n if t_hi is None else t_hi
    tr = S["trials"][trial_mask]
    ok = lambda e: np.array([v for v in e if max(t_lo, fs) <= v <= t_hi], int)
    idle = []
    for m, _, _ in tr:
        prev = S["others"][S["others"] < m - 1]
        if len(prev) == 0 or m - prev[-1] >= 4 * fs:
            idle.append(m - 2 * fs)
    return {"s1_pos": ok(tr[:, 1]),                                   # [go-1, go]
            "s1_neg": ok(np.r_[idle, tr[:, 2] + int(1.5 * fs)]),      # idle + arm returning after release
            "s2_pos": ok(tr[:, 1] + fs // 2),                         # centred on grasp onset
            "s2_neg": ok(tr[:, 0] + fs // 2)}                         # centred on reach start


def collect(S, W):
    fs = S["fs"]
    return (windows_at(S["mb"], W["s1_pos"], fs), windows_at(S["mb"], W["s1_neg"], fs),
            windows_at(S["mr"], W["s2_pos"], fs), windows_at(S["mr"], W["s2_neg"], fs))


def pseudo_online(dec, S, start, batch=3000):
    fs = S["fs"]
    ends = np.arange(max(start, fs), S["n"] + 1, STEP)
    pc, pl = [], []
    for i in range(0, len(ends), batch):
        e = ends[i:i + batch]
        pc.append(dec.p_coming(windows_at(S["mb"], e, fs)))
        pl.append(dec.p_close(windows_at(S["mr"], e, fs)))
    pc, pl = np.concatenate(pc), np.concatenate(pl)
    return ends, {"hybrid": dec.fuse(pc, pl, STEP / fs), "stage1_only": pc, "stage2_only": pl}


def score(S, ends, p, test_trials, on=0.7, off=0.3):
    """Run the real Arbiter over p and count events."""
    fs = S["fs"]
    arb = Arbiter(Scene({}), eeg_on=on, eeg_off=off)
    arb.on_intent(Intent(action=Action.follow))
    releases = set(int(t[2] // STEP) for t in test_trials)
    closes, g_prev = [], 0.0
    for e, v in zip(ends, p):
        if int(e // STEP) in releases:
            arb.gripper = 0.0  # the user lets go at grasp offset
        arb.on_eeg(float(v), now=e / fs)
        if arb.gripper == 1.0 and g_prev == 0.0:
            closes.append(e)
        g_prev = arb.gripper
    closes = np.array(closes, int)
    hits, lat, prem = 0, [], 0
    for m, g, o in test_trials:
        c = closes[(closes >= g - fs // 2) & (closes <= g + int(1.5 * fs))]
        if len(c):
            hits += 1
            lat.append((c[0] - g) / fs)
        if np.any((closes >= m - fs) & (closes < g - fs // 2)):
            prem += 1
    t0, t1 = ends[0], ends[-1]
    busy = [(m - fs, o + 2 * fs) for m, _, o in S["trials"]] + list(S["eye"])
    in_span = lambda t, spans: any(a <= t <= b for a, b in spans)
    free = [c for c in closes if not in_span(c, busy)]
    f_rest = [c for c in free if in_span(c, S["rest"])]
    f_other = [c for c in free if not in_span(c, S["rest"])]
    grid = np.arange(t0, t1, fs)  # 1 s grid to measure idle durations
    rest_min = sum(1 for t in grid if in_span(t, S["rest"]) and not in_span(t, busy)) / 60
    other_min = sum(1 for t in grid if not in_span(t, S["rest"]) and not in_span(t, busy)) / 60
    return {"trials": len(test_trials), "hit_rate": hits / max(1, len(test_trials)),
            "latency_median_s": float(np.median(lat)) if lat else None,
            "premature_rate": prem / max(1, len(test_trials)),
            "false_per_min_rest": len(f_rest) / rest_min if rest_min else None,
            "false_per_min_other_idle": len(f_other) / other_min if other_min else None,
            "idle_min": round(rest_min + other_min, 1), "on": on}


THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def pick_threshold(runs, max_false=1.0):
    """runs: list of (S, ends, p, trials). Highest mean hit rate with median false/min (other idle) <= max_false;
    if none qualifies, the threshold with the fewest false closes. Returns (on, off)."""
    best, fallback = None, None
    for th in THRESHOLDS:
        sc = [score(S, e, p, t, th, max(0.2, th - 0.3)) for S, e, p, t in runs]
        hit = float(np.mean([x["hit_rate"] for x in sc]))
        fp = float(np.median([x["false_per_min_other_idle"] or 0.0 for x in sc]))
        if fp <= max_false and (best is None or hit > best[0]):
            best = (hit, th)
        if fallback is None or fp < fallback[0]:
            fallback = (fp, th)
    th = best[1] if best else fallback[1]
    return th, max(0.2, th - 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-15")
    ap.add_argument("--out", default="models/hybrid_eval.json")
    ap.add_argument("--save-model", default="models/hybrid_all15.joblib")
    args = ap.parse_args()
    a, _, b = args.subjects.partition("-")
    subs = list(range(int(a), int(b or a) + 1))
    data, res = {}, {"calibrated": {}, "new": {}}
    rng = np.random.default_rng(0)
    for s in subs:
        S = prepare(s)
        data[s] = S
        n_tr = len(S["trials"])
        half = np.arange(n_tr) < n_tr // 2
        split = int(S["trials"][n_tr // 2][0] - 2 * S["fs"])
        # thresholds: fit on quarter 1, tune on quarter 2 (never the test half), then refit on the whole first half
        q = np.arange(n_tr) < n_tr // 4
        qsplit = int(S["trials"][n_tr // 4][0] - 2 * S["fs"])
        dec_q = HybridDecoder(fs=S["fs"]).fit(*collect(S, train_windows(S, q, t_hi=qsplit)))
        e_q, P_q = pseudo_online(dec_q, S, qsplit)
        mask_e = e_q <= split
        tune_trials = S["trials"][(~q) & half]
        th = {k: pick_threshold([(S, e_q[mask_e], v[mask_e], tune_trials)]) for k, v in P_q.items()}
        dec = HybridDecoder(fs=S["fs"]).fit(*collect(S, train_windows(S, half, t_hi=split)))
        ends, P = pseudo_online(dec, S, split)
        res["calibrated"][s] = {k: score(S, ends, v, S["trials"][~half], *th[k]) for k, v in P.items()}
        res.setdefault("calibrated_fixed_0.7", {})[s] = {k: score(S, ends, v, S["trials"][~half]) for k, v in P.items()}
        h = res["calibrated"][s]["hybrid"]
        print(f"subject {s} calibrated: hybrid hit {h['hit_rate']:.2f} lat {h['latency_median_s']} "
              f"prem {h['premature_rate']:.2f} false/min rest {h['false_per_min_rest']} other {h['false_per_min_other_idle']}",
              flush=True)
    wins = {s: collect(data[s], train_windows(data[s], np.ones(len(data[s]["trials"]), bool))) for s in subs}
    if len(subs) > 1:
        for s in subs:
            tr = [wins[t] for t in subs if t != s]
            dec = HybridDecoder(fs=data[s]["fs"]).fit(*[np.concatenate([w[i] for w in tr]) for i in range(4)])
            # thresholds tuned on 3 of the training subjects (not the test subject)
            tune_subs = rng.choice([t for t in subs if t != s], size=min(3, len(subs) - 1), replace=False)
            tune = {t: pseudo_online(dec, data[t], 0) for t in tune_subs}
            th = {k: pick_threshold([(data[t], tune[t][0], tune[t][1][k], data[t]["trials"]) for t in tune_subs])
                  for k in ("hybrid", "stage1_only", "stage2_only")}
            ends, P = pseudo_online(dec, data[s], 0)
            res["new"][s] = {k: score(data[s], ends, v, data[s]["trials"], *th[k]) for k, v in P.items()}
            h = res["new"][s]["hybrid"]
            print(f"subject {s} new-person: hybrid hit {h['hit_rate']:.2f} lat {h['latency_median_s']} "
                  f"prem {h['premature_rate']:.2f} false/min rest {h['false_per_min_rest']} "
                  f"other {h['false_per_min_other_idle']}", flush=True)
        import joblib
        dec = HybridDecoder(fs=data[subs[0]]["fs"]).fit(*[np.concatenate([wins[t][i] for t in subs]) for i in range(4)])
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"decoder": dec, "channels": MOTOR_8, "trained_on": f"BNCI 001-2020 gel subjects {subs}"},
                    args.save_model)
        print("saved", args.save_model)
    summary = {}
    for mode, r in res.items():
        for dname in ("hybrid", "stage1_only", "stage2_only"):
            rows = [r[s][dname] for s in r]
            if not rows:
                continue
            med = lambda k: float(np.median([x[k] for x in rows if x[k] is not None])) if any(
                x[k] is not None for x in rows) else None
            summary[f"{mode}/{dname}"] = {"hit_rate_mean": round(float(np.mean([x["hit_rate"] for x in rows])), 3),
                                          "latency_median_s": med("latency_median_s"),
                                          "premature_mean": round(float(np.mean([x["premature_rate"] for x in rows])), 3),
                                          "false_per_min_rest_median": med("false_per_min_rest"),
                                          "false_per_min_other_median": med("false_per_min_other_idle"),
                                          "on_threshold_median": med("on")}
    for k, v in summary.items():
        print(f"{k:24} {v}")
    Path(args.out).write_text(json.dumps({"summary": summary, "per_subject": res}, indent=1, default=float))
    print("saved", args.out)


if __name__ == "__main__":
    main()

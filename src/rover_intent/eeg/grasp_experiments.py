"""Follow-up experiments on BNCI 001-2020 (gel, 15 subjects). See docs/LOG.md for the questions.

E1 baseline   : intent [mo-1, mo] vs in-run idle [mo-3, mo-2]           (repeat of reach_grasp.py)
E2 eog_only   : same windows, 6 EOG channels only                      (eye-movement confound check)
E3 eog_clean  : same windows, EEG with EOG regressed out               (regression fitted on the eye-movement task)
E4 pre_grasp  : [go-1, go] (about to close the hand) vs idle
E5 close_vs_reach : [go-0.5, go+0.5] vs [mo-0.5, mo+0.5]               (both move the arm; only one closes the hand)
E6 mrcp_retry : [mo-0.5, mo+0.5] and [mo, mo+1] vs idle, causal and zero-phase filters
mo = movement onset, go = grasp onset (hand reaches the object). All windows 1 s. Balanced classes (chance 50 %).
Within-subject 5-fold CV and leave-one-subject-out. Channel sets: 8 OpenBCI motor channels and all 58.

    python -m rover_intent.eeg.grasp_experiments --subjects 1-15 --out models/grasp_experiments.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
from pathlib import Path

import numpy as np

from .reach_grasp import MOTOR_8, MOVE_ONSET, balanced, fetch, models

GRASP_ONSET = (501794, 501795)
EYE_ON, EYE_OFF = (10, 12, 14), (11, 13, 15)
BANDS = {"mrcp": (0.3, 3.0), "mu_beta": (8.0, 30.0), "eog_lf": (0.3, 5.0)}


def load_full(subject):
    import scipy.io as sio
    m = sio.loadmat(fetch(subject), squeeze_me=True, struct_as_record=False)
    h, e = m["header"], m["events"]
    eeg_i, eog_i = np.asarray(h.channels_eeg) - 1, np.asarray(h.channels_eog) - 1
    labels = [str(x).strip() for x in np.asarray(h.channels_labels)[eeg_i]]
    sig = m["signal"]
    return (sig[eeg_i].astype(np.float32), sig[eog_i].astype(np.float32), labels, int(h.sample_rate),
            np.asarray(e.codes).astype(np.int64), np.asarray(e.positions).astype(np.int64) - 1)


def eog_regression(eeg, eog, codes, pos):
    """EEG_clean = EEG - EOG @ B, B fitted by least squares on the eye-movement task segments (Schlögl-style)."""
    segs = []
    for on, off in zip(EYE_ON, EYE_OFF):
        for a, b in zip(np.sort(pos[codes == on]), np.sort(pos[codes == off])):
            if b > a:
                segs.append(slice(a, b))
    if not segs:
        return None
    E = np.concatenate([eog[:, s] for s in segs], axis=1).T.astype(np.float64)
    Y = np.concatenate([eeg[:, s] for s in segs], axis=1).T.astype(np.float64)
    E -= E.mean(0); Y -= Y.mean(0)
    B = np.linalg.lstsq(E, Y, rcond=None)[0]  # (n_eog, n_eeg)
    return (eeg - (B.T @ (eog - eog.mean(1, keepdims=True)))).astype(np.float32)


def filt(x, band, fs, zero_phase):
    from scipy.signal import butter, sosfilt, sosfiltfilt
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    return (sosfiltfilt(sos, x, axis=1) if zero_phase else sosfilt(sos, x, axis=1)).astype(np.float32)


def feats(x, starts, fs, kind):
    W = np.stack([x[:, s:s + fs] for s in starts])
    if kind == "mu_beta":
        return W
    return W[:, :, :: fs // 16].reshape(len(starts), -1)  # amplitudes at 16 Hz (mrcp, eog_lf)


def trial_times(codes, pos, fs):
    mo = np.sort(pos[np.isin(codes, MOVE_ONSET)])
    go_all = np.sort(pos[np.isin(codes, GRASP_ONSET)])
    pairs = []  # (mo, go) with go the first grasp onset after mo, within 3 s
    for m in mo:
        g = go_all[(go_all > m) & (go_all < m + 3 * fs)]
        if len(g):
            pairs.append((m, g[0]))
    others = np.sort(pos[~np.isin(codes, (768, 769))])
    idle = []
    for m in mo:
        prev = others[(others < m - 1)]
        if len(prev) == 0 or m - prev[-1] >= 4 * fs:
            idle.append(m - 3 * fs)
    return mo, np.array(pairs, int).reshape(-1, 2), np.array(idle, int)


def main():
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-15")
    ap.add_argument("--out", default="models/grasp_experiments.json")
    ap.add_argument("--only", default="", help="run only experiments whose name starts with this")
    args = ap.parse_args()
    a, _, b = args.subjects.partition("-")
    subs = list(range(int(a), int(b or a) + 1))
    rng = np.random.default_rng(0)
    mk = models()
    mk["eog_lf"] = mk["mrcp"]  # same shrinkage-LDA on low-frequency amplitudes
    within, pooled, notes = {}, {}, {}

    for s in subs:
        eeg, eog, labels, fs, codes, pos = load_full(s)
        n = eeg.shape[1]
        mo, pairs, idle = trial_times(codes, pos, fs)
        ok = lambda st: np.array([x for x in st if 0 <= x and x + fs <= n], int)
        h = fs // 2
        W = {"intent": ok(mo - fs), "idle": ok(idle), "pre_grasp": ok(pairs[:, 1] - fs),
             "grasp_c": ok(pairs[:, 1] - h), "reach_c": ok(pairs[:, 0] - h),
             "mo_centered": ok(mo - h), "mo_after": ok(mo)}
        clean = eog_regression(eeg, eog, codes, pos)
        notes[s] = {k: len(v) for k, v in W.items()} | {"eog_regression": clean is not None,
                                                        "reach_s_median": float(np.median(pairs[:, 1] - pairs[:, 0]) / fs)}
        car = eeg - eeg.mean(0, keepdims=True)
        sig_cache = {}

        def sig(src, kind, zp, chans):
            key = (src, kind, zp, tuple(chans) if chans else None)
            if key not in sig_cache:
                if src == "eog":
                    x = eog
                else:
                    base = {"raw": eeg, "clean": clean}[src]
                    if kind == "mrcp":
                        base = base - base.mean(0, keepdims=True) if src == "clean" else car
                    x = base[[labels.index(c) for c in chans]]
                sig_cache[key] = filt(x, BANDS[kind], fs, zp)
            return sig_cache[key]

        exps = []  # (name, pos_key, neg_key, src, kind, zero_phase, chanset)
        for cs in ("motor8", "all58"):
            for kind in ("mu_beta", "mrcp"):
                exps += [("E1_baseline", "intent", "idle", "raw", kind, False, cs),
                         ("E4_pre_grasp", "pre_grasp", "idle", "raw", kind, False, cs),
                         ("E5_close_vs_reach", "grasp_c", "reach_c", "raw", kind, False, cs)]
                if clean is not None:
                    exps += [("E3_eog_clean", "intent", "idle", "clean", kind, False, cs),
                             ("E3_eog_clean_pre_grasp", "pre_grasp", "idle", "clean", kind, False, cs),
                             ("E3_eog_clean_close_vs_reach", "grasp_c", "reach_c", "clean", kind, False, cs)]
                for zp in (False, True):
                    tag = "zerophase" if zp else "causal"
                    exps += [(f"E6_mrcp_retry_centered_{tag}", "mo_centered", "idle", "raw", kind, zp, cs),
                             (f"E6_mrcp_retry_after_{tag}", "mo_after", "idle", "raw", kind, zp, cs)]
        for name, pk, nk in [("E2_eog_only", "intent", "idle"), ("E2_eog_only_pre_grasp", "pre_grasp", "idle"),
                             ("E2_eog_only_close_vs_reach", "grasp_c", "reach_c")]:
            exps.append((name, pk, nk, "eog", "eog_lf", False, "eog6"))

        exps = [e for e in exps if e[0].startswith(args.only)]
        for name, pk, nk, src, kind, zp, cs in exps:
            chans = MOTOR_8 if cs == "motor8" else (labels if cs == "all58" else None)
            x = sig(src, kind, zp, chans)
            X, y = balanced(feats(x, W[pk], fs, kind), feats(x, W[nk], fs, kind), rng)
            key = f"{name}/{cs}/{kind}"
            acc = cross_val_score(mk[kind](), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
            within.setdefault(key, []).append(float(acc))
            pooled.setdefault(key, []).append((X, y))
        del eeg, eog, clean, car, sig_cache
        print(f"subject {s}: {notes[s]}", flush=True)

    results = {}
    for key, data in pooled.items():
        lo = []
        if len(data) > 1:
            for i in range(len(data)):
                Xtr = np.concatenate([d[0] for j, d in enumerate(data) if j != i])
                ytr = np.concatenate([d[1] for j, d in enumerate(data) if j != i])
                lo.append(float(mk[key.split("/")[-1]]().fit(Xtr, ytr).score(*data[i])))
        w = within[key]
        results[key] = {"within": round(float(np.mean(w)), 3), "within_std": round(float(np.std(w)), 3),
                        "ge70": f"{sum(v >= 0.7 for v in w)}/{len(w)}",
                        "loso": round(float(np.mean(lo)), 3) if lo else None}
    for k in sorted(results):
        r = results[k]
        print(f"{k:55} within {r['within']:.3f}±{r['within_std']:.3f} ≥70%:{r['ge70']:>6}  loso {r['loso']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"subjects": subs, "notes": notes, "results": results}, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()

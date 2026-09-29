"""Grasp-intent decoding on BNCI Horizon 001-2020 (Schwarz et al. 2020, Front. Neurosci., CC BY 4.0).

Self-initiated reach-and-grasp (palmar: jar, lateral: spoon), 15 subjects per EEG system; we use the gel system
(58 EEG channels, 256 Hz). Question: can we detect *intent to grasp* before the arm moves?

  positive  = [-1.0, 0.0] s before movement onset (both grasp types): intent, no movement yet
  control A = 1 s windows from the rest blocks            (standard, but recorded at the END of the session:
                                                            drift over time can inflate accuracy)
  control B = [-3.0, -2.0] s before movement onset         (idle gaze inside the same runs: no time confound,
                                                            the realistic "hand hovering, not grasping yet" case)
Features:
  mrcp      : causal 0.3-3 Hz band-pass, common average reference, 16 Hz samples of the window -> shrinkage LDA
              (the movement-related cortical potential approach used by the Graz group)
  mu_beta   : causal 8-30 Hz band-pass -> covariance -> Riemann tangent space -> logistic regression (our current model)
Filters are causal (sosfilt) so the numbers are achievable online. Channel sets: all 58, and the 8 OpenBCI motor channels.

    python -m rover_intent.eeg.reach_grasp --subjects 1-15 --out models/reach_grasp.json
"""
from __future__ import annotations

import os

# Tiny matrices + 30 BLAS threads = the run stalls (seen: 2900% CPU for 10 min on one subject). Cap threads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://bnci-horizon-2020.eu/database/data-sets/001-2020/G{:02d}.mat"
DATA = Path(__file__).resolve().parents[3] / "../data/bnci_001_2020"
MOVE_ONSET = (503587, 503588)  # palmar, lateral
REST_ON, REST_OFF = 768, 769
MOTOR_8 = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Pz"]
MRCP_BAND, MUBETA_BAND = (0.3, 3.0), (8.0, 30.0)
MRCP_FS = 16


def fetch(subject: int) -> Path:
    p = (DATA / f"G{subject:02d}.mat").resolve()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(subject), tmp)
        tmp.rename(p)
    return p


def load(subject: int):
    import scipy.io as sio
    m = sio.loadmat(fetch(subject), squeeze_me=True, struct_as_record=False)
    h, e = m["header"], m["events"]
    idx = np.asarray(h.channels_eeg) - 1  # MATLAB 1-based
    labels = [str(x).strip() for x in np.asarray(h.channels_labels)[idx]]
    # positions are uint32 in the files: "onset - 3 s" wrapped to ~4.29e9 for an early trial (subject 8). Use int64.
    pos = np.asarray(e.positions).astype(np.int64) - 1
    return m["signal"][idx].astype(np.float32), labels, int(h.sample_rate), np.asarray(e.codes).astype(np.int64), pos


def causal_band(x, band, fs):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, band, btype="band", fs=fs, output="sos"), x, axis=1).astype(np.float32)


def windows(codes, pos, fs, n_total):
    """Sample indices (start) of 1 s windows: intent, rest-block, in-run idle."""
    onsets = np.sort(pos[np.isin(codes, MOVE_ONSET)])
    intent = onsets - fs
    # idle B: [-3,-2] s before onset, only if nothing else happened in the 4 s before the onset
    others = np.sort(pos[~np.isin(codes, [REST_ON, REST_OFF])])
    idle = []
    for o in onsets:
        prev = others[others < o]
        prev = prev[prev < o - 1]  # ignore the onset marker itself
        if len(prev) == 0 or o - prev[-1] >= 4 * fs:
            idle.append(o - 3 * fs)
    rest = []
    for a, b in zip(np.sort(pos[codes == REST_ON]), np.sort(pos[codes == REST_OFF])):
        rest += list(range(a + 5 * fs, b - 6 * fs, fs))  # skip 5 s at the edges
    ok = lambda arr: np.array([s for s in arr if 0 <= s and s + fs <= n_total], int)
    return ok(intent), ok(rest), ok(idle)


def cut(x, starts, fs):
    return np.stack([x[:, s:s + fs] for s in starts])


def features(eeg, labels, fs, starts, chans):
    ci = [labels.index(c) for c in chans]
    car = eeg - eeg.mean(axis=0, keepdims=True)
    mrcp = causal_band(car[ci], MRCP_BAND, fs)
    mubeta = causal_band(eeg[ci], MUBETA_BAND, fs)
    step = fs // MRCP_FS
    return {"mrcp": cut(mrcp, starts, fs)[:, :, ::step].reshape(len(starts), -1),
            "mu_beta": cut(mubeta, starts, fs)}


def models():
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"mrcp": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            "mu_beta": lambda: make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=1000))}


def balanced(Xp, Xn, rng):
    n = min(len(Xp), len(Xn))
    ip, ineg = rng.choice(len(Xp), n, replace=False), rng.choice(len(Xn), n, replace=False)
    return np.concatenate([Xp[ip], Xn[ineg]]), np.r_[np.ones(n), np.zeros(n)].astype(int)


def main():
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-15")
    ap.add_argument("--out", default="models/reach_grasp.json")
    args = ap.parse_args()
    a, _, b = args.subjects.partition("-")
    subs = list(range(int(a), int(b or a) + 1))
    rng = np.random.default_rng(0)
    make = models()
    chan_sets = {"all58": None, "motor8": MOTOR_8}
    within = {}   # (chanset, control, feat) -> list of per-subject acc
    pooled = {}   # (chanset, control, feat) -> list of (X, y) per subject, for leave-subject-out
    counts = {}
    for s in subs:
        eeg, labels, fs, codes, pos = load(s)
        i_int, i_rest, i_idle = windows(codes, pos, fs, eeg.shape[1])
        counts[s] = {"intent": len(i_int), "rest": len(i_rest), "idle": len(i_idle)}
        for cs, chans in chan_sets.items():
            chans = chans or labels
            F = {k: features(eeg, labels, fs, st, chans) for k, st in
                 [("intent", i_int), ("rest", i_rest), ("idle", i_idle)]}
            for control in ("rest", "idle"):
                for feat in ("mrcp", "mu_beta"):
                    X, y = balanced(F["intent"][feat], F[control][feat], rng)
                    acc = cross_val_score(make[feat](), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
                    within.setdefault((cs, control, feat), []).append(float(acc))
                    pooled.setdefault((cs, control, feat), []).append((X, y))
        del eeg
        print(f"subject {s}: {counts[s]} " + " ".join(
            f"{cs}/{c}/{f}={within[(cs, c, f)][-1]:.2f}" for cs in chan_sets for c in ("rest", "idle")
            for f in ("mrcp", "mu_beta")), flush=True)

    loso = {}
    for key, data in pooled.items():
        if len(data) < 2:
            loso[key] = [float("nan")]
            continue
        accs = []
        for i in range(len(data)):
            Xtr = np.concatenate([d[0] for j, d in enumerate(data) if j != i])
            ytr = np.concatenate([d[1] for j, d in enumerate(data) if j != i])
            accs.append(float(make[key[2]]().fit(Xtr, ytr).score(*data[i])))
        loso[key] = accs
    report = {"dataset": "BNCI 001-2020 gel (Schwarz et al. 2020), CC BY 4.0", "subjects": subs, "counts": counts,
              "chance": 0.5, "results": {}}
    for key in within:
        cs, c, f = key
        report["results"][f"{cs}/{c}/{f}"] = {
            "within_subject_cv": round(float(np.mean(within[key])), 3), "within_std": round(float(np.std(within[key])), 3),
            "subjects_ge_0.70": f"{sum(v >= 0.7 for v in within[key])}/{len(subs)}",
            "leave_subject_out": round(float(np.mean(loso[key])), 3), "loso_std": round(float(np.std(loso[key])), 3)}
    print(json.dumps(report["results"], indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()

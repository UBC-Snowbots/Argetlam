"""Train + evaluate the binary motor-imagery decoder (clench vs rest) on EEGMMIDB.

    python -m rover_intent.eeg.train_mi --subjects 1-20 --out models/mi_riemann.joblib

Reports two numbers, honestly separated:
  within-subject 5-fold CV  (what a calibrated user could get)
  leave-subjects-out        (what a brand-new user gets with zero calibration)
Models: CSP+LDA (classic baseline) and Riemannian tangent space + logistic regression.
Chance is 50% (classes are balanced).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import balance, load_subject


def pipelines():
    from mne.decoding import CSP
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return {
        "csp_lda": lambda: make_pipeline(CSP(n_components=6, log=True), LinearDiscriminantAnalysis()),
        "riemann_ts_lr": lambda: make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=1000)),
    }


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    from sklearn.model_selection import cross_val_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-20")
    ap.add_argument("--model", default="riemann_ts_lr")
    ap.add_argument("--out", default="models/mi_riemann.joblib")
    args = ap.parse_args()
    subs = parse_range(args.subjects)
    pipes = pipelines()

    data, within, skipped = {}, {k: [] for k in pipes}, []
    for s in list(subs):
        try:
            X, y = balance(*load_subject(s))
        except Exception as e:  # a few EEGMMIDB subjects have odd runs; skip, don't crash the sweep
            print(f"subject {s}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            skipped.append(s)
            subs.remove(s)
            continue
        data[s] = (X, y)
        for k, make in pipes.items():
            within[k].append(cross_val_score(make(), X, y, cv=5).mean())
        print(f"subject {s}: n={len(y)} " + " ".join(f"{k}={within[k][-1]:.2f}" for k in pipes), flush=True)

    # leave-subjects-out: 5 folds over subjects
    folds = np.array_split(np.array(subs), min(5, len(subs)))
    loso = {k: [] for k in pipes}
    for f in folds:
        tr = [s for s in subs if s not in f]
        Xtr = np.concatenate([data[s][0] for s in tr]); ytr = np.concatenate([data[s][1] for s in tr])
        for k, make in pipes.items():
            m = make().fit(Xtr, ytr)
            loso[k] += [m.score(*data[s]) for s in f]

    report = {k: {"within_subject_cv": float(np.mean(within[k])), "within_std": float(np.std(within[k])),
                  "subjects_within_ge_0.70": f"{int(np.sum(np.array(within[k]) >= 0.70))}/{len(subs)}",
                  "leave_subjects_out": float(np.mean(loso[k])), "loso_std": float(np.std(loso[k]))}
              for k in pipes}
    report["n_subjects"], report["skipped"] = len(subs), skipped
    print(json.dumps(report, indent=2))

    Path(args.out).with_suffix(".json").parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    import joblib
    Xall = np.concatenate([d[0] for d in data.values()]); yall = np.concatenate([d[1] for d in data.values()])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipes[args.model]().fit(Xall, yall), "report": report, "subjects": subs}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

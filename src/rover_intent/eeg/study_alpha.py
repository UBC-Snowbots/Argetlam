"""Task A: eyes-closed alpha proxy study on EEGMMIDB baseline runs R01 (eyes open) vs R02 (eyes closed).

Question for the Muse: how well does 8-12 Hz power separate eyes-closed from eyes-open as a function of window length,
using only electrodes near the Muse's (AF7, AF8, and the nearest temporal sites TP7/TP8/T9/T10 - EEGMMIDB has no
TP9/TP10), compared with a full posterior montage (O1, Oz, O2, P7, P8) as a ceiling? Two usage scenarios:
  within-subject calibrated: train and test on the same person (temporal 2-fold within the runs)
  new-person: leave-one-subject-out (train on everyone else, test on the held-out person, cold)

LIMITATION (stated in the report): R01 and R02 are separate 1-minute runs, so this measures steady-state separability,
NOT the onset latency of a real eyes-open -> eyes-closed switch.

Run through task-spooler (downloads 218 EDFs on first run, then analyses). One subject in memory at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from rover_intent.eeg.alpha import alpha_log_features, rel_alpha

# default data location: the shared data/ folder next to the repo (override with --data / --out)
_DATA = Path(__file__).resolve().parents[3].parent / "data"

# EEGMMIDB 10-10 names after eegbci.standardize()
MUSE_PROXY = ["AF7", "AF8", "TP7", "TP8", "T9", "T10"]  # nearest to Muse TP9/AF7/AF8/TP10 (no TP9/TP10 in this montage)
POSTERIOR = ["O1", "Oz", "O2", "P7", "P8"]              # ceiling: a real occipital/parietal montage
WINDOWS_S = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
OVERLAP = 0.5
FS = 160.0


def _load_run(subject: int, run: int, channels, data_path: str):
    """Load one baseline run, filter once, return (dict ch_name -> row, n_times). Channels is the union needed."""
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")
    files = eegbci.load_data(subject, [run], path=data_path, update_path=False)
    raw = mne.io.read_raw_edf(files[0], preload=True)
    eegbci.standardize(raw)
    raw.notch_filter(60.0, fir_design="firwin", verbose="ERROR")
    raw.filter(1.0, 40.0, fir_design="firwin", verbose="ERROR")
    raw.pick(channels)
    data = raw.get_data()  # (n_ch, n_times), volts
    return {ch: data[i] for i, ch in enumerate(raw.ch_names)}


def _windows(x: np.ndarray, win_s: float, fs: float):
    n = int(win_s * fs)
    step = max(1, int(n * (1 - OVERLAP)))
    return [x[:, i:i + n] for i in range(0, x.shape[1] - n + 1, step)]


def _features_from_runs(runs, channels, win_s):
    """runs: dict label -> (dict ch -> row). Return (F, y, order) for the given channel set and window length."""
    F, y, order = [], [], []
    for label, rundata in runs.items():
        x = np.array([rundata[ch] for ch in channels])  # (n_ch, n_times)
        wins = _windows(x, win_s, FS)
        for k, w in enumerate(wins):
            F.append(alpha_log_features(w, FS))
            y.append(label)
            order.append(k / max(1, len(wins) - 1))  # 0..1 position within its run
    return np.array(F), np.array(y), np.array(order)


def _bal_acc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = [(y_pred[y_true == c] == c).mean() for c in (0, 1) if (y_true == c).any()]
    return float(np.mean(accs))


def _within_subject(F, y, order):
    """Temporal 2-fold: train first half of each run / test second half, and the reverse. Balanced accuracy."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    accs = []
    for lo, hi in [(0.5, 1.0), (0.0, 0.5)]:
        test = (order >= lo) & (order < hi) | ((hi == 1.0) & (order == 1.0))
        train = ~test
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        clf = LDA().fit(F[train], y[train])
        accs.append(_bal_acc(y[test], clf.predict(F[test])))
    return float(np.mean(accs)) if accs else np.nan


def _threshold_within(F_relmean, y, order):
    """Simple calibrated threshold on mean relative-alpha (the AlphaDetector path), temporal 2-fold."""
    accs = []
    for lo, hi in [(0.5, 1.0), (0.0, 0.5)]:
        test = ((order >= lo) & (order < hi)) | ((hi == 1.0) & (order == 1.0))
        train = ~test
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        mo, mc = F_relmean[train][y[train] == 0], F_relmean[train][y[train] == 1]
        cand = np.sort(np.concatenate([mo, mc]))
        mids = (cand[:-1] + cand[1:]) / 2
        best_t, best = np.median(cand), -1
        for t in mids:
            a = 0.5 * ((mo < t).mean() + (mc >= t).mean())
            if a > best:
                best, best_t = a, t
        pred = (F_relmean[test] >= best_t).astype(int)
        accs.append(_bal_acc(y[test], pred))
    return float(np.mean(accs)) if accs else np.nan


def run_study(subjects, data_path, out_path):
    os.environ.setdefault("MNE_DATA", data_path)
    results = {"channel_sets": {"muse_proxy": MUSE_PROXY, "posterior": POSTERIOR},
               "windows_s": WINDOWS_S, "scenarios": {}, "n_subjects": 0, "limitation":
               "R01/R02 are separate runs; onset latency of a live open->closed switch is not measured."}
    cache = {}  # (set, win) -> list of (F, y, order) per subject
    done = []
    union = sorted(set(MUSE_PROXY + POSTERIOR))
    for name, chans in [("muse_proxy", MUSE_PROXY), ("posterior", POSTERIOR)]:
        for win in WINDOWS_S:
            cache[(name, win)] = []
    for si, subj in enumerate(subjects):
        t0 = time.time()
        try:  # load both baseline runs once, filter once, keep all needed channels
            runs = {0: _load_run(subj, 1, union, data_path), 1: _load_run(subj, 2, union, data_path)}
        except Exception as e:  # noqa: BLE001
            print(f"[{si+1}/{len(subjects)}] S{subj:03d} SKIP ({e})", flush=True)
            continue
        for name, chans in [("muse_proxy", MUSE_PROXY), ("posterior", POSTERIOR)]:
            for win in WINDOWS_S:
                cache[(name, win)].append(_features_from_runs(runs, chans, win))
        done.append(subj)
        print(f"[{si+1}/{len(subjects)}] S{subj:03d} ok {time.time()-t0:.1f}s", flush=True)
    results["n_subjects"] = len(done)
    results["subjects"] = done

    for name in ("muse_proxy", "posterior"):
        results["scenarios"].setdefault("within_lda", {})[name] = {}
        results["scenarios"].setdefault("within_threshold", {})[name] = {}
        results["scenarios"].setdefault("newperson_lda", {})[name] = {}
        for win in WINDOWS_S:
            per_subj = cache[(name, win)]
            # within-subject
            wacc, tacc = [], []
            for F, y, order in per_subj:
                wacc.append(_within_subject(F, y, order))
                # the log relative-alpha features are the 2nd half of the vector; their channel mean is the
                # AlphaDetector metric, so the simple-threshold scenario reuses them (no need to re-read EDFs)
                nrel = F.shape[1] // 2
                relmean = F[:, nrel:].mean(axis=1)
                tacc.append(_threshold_within(relmean, y, order))
            wacc = np.array([a for a in wacc if not np.isnan(a)])
            tacc = np.array([a for a in tacc if not np.isnan(a)])
            results["scenarios"]["within_lda"][name][str(win)] = {
                "mean": float(np.mean(wacc)), "std": float(np.std(wacc)), "n": int(len(wacc))}
            results["scenarios"]["within_threshold"][name][str(win)] = {
                "mean": float(np.mean(tacc)), "std": float(np.std(tacc)), "n": int(len(tacc))}
            # new-person LOSO
            results["scenarios"]["newperson_lda"][name][str(win)] = _loso(per_subj)
            print(f"{name} win={win}: within_lda={results['scenarios']['within_lda'][name][str(win)]['mean']:.3f} "
                  f"newperson={results['scenarios']['newperson_lda'][name][str(win)]['mean']:.3f}", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_path}")
    return results


def _loso(per_subj):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    accs = []
    n = len(per_subj)
    Fs = [p[0] for p in per_subj]
    ys = [p[1] for p in per_subj]
    for i in range(n):
        Ftr = np.concatenate([Fs[j] for j in range(n) if j != i])
        ytr = np.concatenate([ys[j] for j in range(n) if j != i])
        if len(np.unique(ys[i])) < 2:
            continue
        sc = StandardScaler().fit(Ftr)
        clf = LDA().fit(sc.transform(Ftr), ytr)
        accs.append(_bal_acc(ys[i], clf.predict(sc.transform(Fs[i]))))
    accs = np.array(accs)
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "n": int(len(accs))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(_DATA / "mne_data"))
    ap.add_argument("--out", default=str(_DATA / "eeg_studies/alpha_results.json"))
    ap.add_argument("--subjects", type=int, default=109)
    args = ap.parse_args(argv)
    subjects = list(range(1, args.subjects + 1))
    run_study(subjects, args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task B: SSVEP proxy study on the Tsinghua benchmark (Wang et al. 2016, MOABB 'Wang2016': 35 subjects, 40 targets
8.0-15.8 Hz, 64 ch + stim, 250 Hz, 6 blocks). We run a subset of subjects.

Question for the Muse: how well can we read which flicker frequency a person is looking at, using only electrodes near
the Muse's (TP7/TP8/T7/T8 - this montage has no TP9/TP10 and no AF7/AF8, it has AF3/AF4), an Oz-only "AUX electrode"
scenario, and the standard 9 occipital channels as a ceiling? Training-free CCA and FBCCA, and calibrated ensemble
TRCA (leave-one-block-out). 2-class (12 vs 15 Hz) and 4-class (12/13/14/15 Hz), avoiding ~10 Hz alpha. Accuracy and
ITR vs window length.

Chance: 2-class 50%, 4-class 25%. Run through task-spooler (downloads ~100 MB/subject on first run). One subject in
memory at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from rover_intent.eeg.ssvep import TRCA, cca_scores, fbcca_scores, itr_bits_per_min, reference_signals

# default data location: the shared data/ folder next to the repo (override with --data / --out)
_DATA = Path(__file__).resolve().parents[3].parent / "data"

MUSE_PROXY = ["TP7", "TP8", "T7", "T8"]            # nearest to Muse TP9/TP10 (+ temporal); no AF7/AF8 in this montage
AUX_OZ = ["Oz"]                                    # single "AUX electrode over the occiput" scenario
OCCIPITAL9 = ["Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]  # standard Tsinghua ceiling
CLASS_SETS = {"2class": [12.0, 15.0], "4class": [12.0, 13.0, 14.0, 15.0]}
ALL_FREQS = sorted(set(sum(CLASS_SETS.values(), [])))
WINDOWS_S = [0.5, 1.0, 2.0, 3.0, 4.0]
VIS_LATENCY = 0.14  # s, standard Tsinghua benchmark onset delay
HARMONICS = 5
FS = 250.0


def _freq_label(f: float) -> str:
    return f"{f:g}"


def load_subject_epochs(subject: int, data_path: str):
    """Return dict freq -> (n_trials, n_ch, n_times) at FS, tmin=VIS_LATENCY, max window, and the channel names."""
    import mne
    from moabb.datasets import Wang2016
    mne.set_log_level("ERROR")
    os.environ.setdefault("MNE_DATA", data_path)
    ds = Wang2016()
    data = ds.get_data(subjects=[subject])
    raw = None
    for _s, sessions in data.items():
        for _sess, runs in sessions.items():
            for _run, r in runs.items():
                raw = r
                break
            break
        break
    events, evid = mne.events_from_annotations(raw)
    want = {_freq_label(f): evid[_freq_label(f)] for f in ALL_FREQS if _freq_label(f) in evid}
    tmax = VIS_LATENCY + max(WINDOWS_S)
    picks = [c for c in raw.ch_names if c != "stim"]
    ep = mne.Epochs(raw, events, want, tmin=VIS_LATENCY, tmax=tmax, baseline=None, picks=picks,
                    preload=True, verbose="ERROR")
    ch_names = ep.ch_names
    out = {}
    for f in ALL_FREQS:
        lab = _freq_label(f)
        if lab in want:
            X = ep[lab].get_data()  # (n_trials, n_ch, n_times)
            out[f] = X.astype(np.float32)
    return out, ch_names


def _pick(X, ch_names, wanted):
    idx = [ch_names.index(c) for c in wanted if c in ch_names]
    return X[:, idx, :], [c for c in wanted if c in ch_names]


def _slice(X, win_s):
    n = int(win_s * FS)
    return X[:, :, :n]


def eval_trainfree(epochs_by_freq, ch_names, chans, classes, win_s, method):
    """CCA or FBCCA. Returns accuracy over all trials of the class set."""
    refs = reference_signals(classes, FS, int(win_s * FS), HARMONICS)
    correct, total = 0, 0
    for true_i, f in enumerate(classes):
        if f not in epochs_by_freq:
            continue
        X, used = _pick(epochs_by_freq[f], ch_names, chans)
        if not used:
            return np.nan
        X = _slice(X, win_s)
        for trial in X:
            if method == "cca":
                sc = cca_scores(trial, classes, FS, HARMONICS, refs=refs)
            else:
                sc = fbcca_scores(trial, classes, FS, HARMONICS, refs=refs)
            correct += int(np.argmax(sc) == true_i)
            total += 1
    return correct / total if total else np.nan


def eval_trca(epochs_by_freq, ch_names, chans, classes, win_s, n_blocks=6):
    """Ensemble TRCA, leave-one-block-out CV. Assumes trials are block-ordered (6 blocks, one trial/freq/block)."""
    present = [f for f in classes if f in epochs_by_freq]
    if len(present) < len(classes):
        return np.nan
    # build per-class arrays, sliced
    per_class = {}
    for f in present:
        X, used = _pick(epochs_by_freq[f], ch_names, chans)
        if not used:
            return np.nan
        per_class[f] = _slice(X, win_s)
    nb = min(n_blocks, min(v.shape[0] for v in per_class.values()))
    accs = []
    for b in range(nb):
        train_ep, train_lab, test = [], [], []
        for ci, f in enumerate(present):
            Xc = per_class[f]
            test_idx = b
            tr = [i for i in range(Xc.shape[0]) if i != test_idx]
            for i in tr:
                train_ep.append(Xc[i]); train_lab.append(ci)
            test.append((Xc[test_idx], ci))
        model = TRCA(FS, ensemble=True).fit(np.array(train_ep), np.array(train_lab))
        for Xt, ci in test:
            accs.append(int(model.predict(Xt) == ci))
    return float(np.mean(accs)) if accs else np.nan


def run_study(subjects, data_path, out_path):
    results = {"dataset": "Wang2016 (Tsinghua benchmark, Wang et al. 2016)", "fs": FS,
               "channel_sets": {"muse_proxy": MUSE_PROXY, "aux_oz": AUX_OZ, "occipital9": OCCIPITAL9},
               "class_sets": CLASS_SETS, "windows_s": WINDOWS_S, "harmonics": HARMONICS,
               "note_channels": "This montage lacks AF7/AF8 (has AF3/AF4) and TP9/TP10; muse_proxy uses TP7/TP8/T7/T8.",
               "chance": {"2class": 0.5, "4class": 0.25}, "subjects": [], "per_subject": {}, "summary": {}}
    # accumulate: acc[setting][method][chanset][win] -> list over subjects
    acc = {}
    for si, subj in enumerate(subjects):
        t0 = time.time()
        try:
            epochs, ch_names = load_subject_epochs(subj, data_path)
        except Exception as e:  # noqa: BLE001
            print(f"[{si+1}/{len(subjects)}] S{subj} FAILED: {e}", flush=True)
            continue
        n_tr = {f: epochs[f].shape[0] for f in epochs}
        results["subjects"].append(subj)
        results["per_subject"][str(subj)] = {}
        for setting, classes in CLASS_SETS.items():
            for method in ("cca", "fbcca", "trca"):
                for cs_name, chans in [("muse_proxy", MUSE_PROXY), ("aux_oz", AUX_OZ), ("occipital9", OCCIPITAL9)]:
                    for win in WINDOWS_S:
                        key = (setting, method, cs_name, str(win))
                        if method == "trca":
                            a = eval_trca(epochs, ch_names, chans, classes, win)
                        else:
                            a = eval_trainfree(epochs, ch_names, chans, classes, win, method)
                        acc.setdefault(key, []).append(a)
        print(f"[{si+1}/{len(subjects)}] S{subj} ok trials={n_tr} {time.time()-t0:.1f}s", flush=True)

    # summarise
    for (setting, method, cs_name, win), vals in acc.items():
        vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
        if not len(vals):
            continue
        d = results["summary"].setdefault(setting, {}).setdefault(method, {}).setdefault(cs_name, {})
        n_classes = len(CLASS_SETS[setting])
        mean_acc = float(np.mean(vals))
        d[win] = {"acc_mean": round(mean_acc, 4), "acc_std": round(float(np.std(vals)), 4),
                  "n_subjects": int(len(vals)),
                  "itr_bits_per_min": round(itr_bits_per_min(mean_acc, n_classes, float(win)), 2)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_path}", flush=True)
    # print a compact headline
    for setting in CLASS_SETS:
        for method in ("cca", "fbcca", "trca"):
            for cs in ("muse_proxy", "aux_oz", "occipital9"):
                row = results["summary"].get(setting, {}).get(method, {}).get(cs, {})
                if row:
                    cells = " ".join(f"{w}s={row[w]['acc_mean']:.2f}" for w in map(str, WINDOWS_S) if w in row)
                    print(f"{setting:7s} {method:6s} {cs:11s} {cells}", flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(_DATA / "mne_data"))
    ap.add_argument("--out", default=str(_DATA / "eeg_studies/ssvep_results.json"))
    ap.add_argument("--subjects", default="1-9", help="e.g. 1-9 or 1,2,3")
    args = ap.parse_args(argv)
    if "-" in args.subjects:
        a, b = args.subjects.split("-")
        subjects = list(range(int(a), int(b) + 1))
    else:
        subjects = [int(x) for x in args.subjects.split(",")]
    run_study(subjects, args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

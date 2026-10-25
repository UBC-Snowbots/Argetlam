"""Imagined-fist motor imagery on the Muse 2 (4 temporal/frontal channels).

The Muse 2 has only TP9, AF7, AF8, TP10 (behind the ears + forehead). There is NO electrode over the motor
strip, where imagined-hand-movement signals (mu/beta ERD over C3/C4) actually show up. So the existing EEGMMIDB
models (eeg/train_mi.py, 8 motor-strip channels) cannot run on a Muse at all.

This module does the honest thing. It retrains the SAME task (imagined fist T1/T2 vs rest T0, EEGMMIDB imagery
runs 4/8/12) using only the nearest EEGMMIDB electrodes to the Muse's placement, so we can measure up front what a
new user would actually get on a Muse before anyone straps one on. It then provides a live cue protocol to measure
that on the real headband and tools to collect labelled data and train a Muse-native personal model.

Artifact honesty (the Muse's main danger): the band sits over muscle (jaw/temporalis at TP9/TP10) and eyes
(AF7/AF8). Both produce strong signals that a decoder can latch onto and fake "imagery" accuracy. We fight this
three ways: (1) an EMG guard that rejects trials whose 30-100 Hz power is far above the rest level; (2) control
decoders trained on EMG-only (>30 Hz) and EOG-only (<4 Hz frontal) features on the SAME trials - if a control
matches the main model, the main model is decoding artifacts, and we print that verdict; (3) a binomial chance
threshold and a label-permutation p-value so "above chance" is a real claim, not noise.

Muse-proxy channels (Muse order): T9 (for TP9), AF7, AF8, T10 (for TP10). EEGMMIDB's montage has no TP9/TP10;
after eegbci.standardize it has T9/T10 (used) and TP7/TP8 (fallback). Channels are configurable (--channels) so a
5th AUX electrode (e.g. a C3 over the motor strip on the Muse AUX port) can be added later without code changes;
the board read in clench.open_board currently exposes 4 channels, so adding AUX also needs that slice widened.

Commands (python -m rover_intent.eeg.muse_mi <cmd>):
  train-base      base (new-user) model on EEGMMIDB: Muse-proxy vs motor-8, LOSO + within-subject, + EOG control
  live-eval       cue protocol on the real board + run a model: trial/window accuracy, binomial CI + permutation,
                  EMG guard
  collect         cue protocol, save labelled raw trials (optionally show live p) for personal training
  train-personal  Muse-native (board fs, 4 ch) model on collected sessions: leave-one-run-out CV, EA, controls

Boards come from rover_intent.eeg.clench.open_board, so --board synthetic works now and --board muse2 later.
Heavy training runs via tsp with <= 4 threads (the host CPU cooler is faulty).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .clench import MUSE_CHANNELS, _record, _say, open_board

# imagery band + sliding-window defaults
BAND = (8.0, 30.0)
SUBBANDS = [(8.0, 13.0), (13.0, 30.0)]   # mu, beta (band-power + shrinkage LDA baseline)
STEP_S = 0.25            # slide the window this often inside a cue
SKIP_S = 0.5             # skip the first half-second of each cue (score 0.5-4 s post-cue)
BASE_WIN_S = 2.0         # EEGMMIDB epoch is 0.5-2.5 s (2 s); the base model covariance is a 2 s window

# artifact controls
EMG_GUARD_BAND = (30.0, 100.0)   # muscle: reject a trial above rest by >EMG_GUARD_SD standard deviations
EMG_GUARD_SD = 3.0
EMG_GUARD_RATIO = 2.0            # fallback when the rest spread is ~0
EMG_CONTROL_BAND = (30.0, 100.0)  # EMG-only control decoder features
EOG_CONTROL_BAND = (0.5, 4.0)     # EOG-only control decoder features (frontal channels)
FRONTAL = ("AF7", "AF8")


# --- channel resolution ---------------------------------------------------------------------------------------------
def resolve_proxy_channels(subject: int = 1, want: list[str] | None = None) -> tuple[list[str], str]:
    """Muse-proxy channels in Muse order, checked against the real EEGMMIDB montage after standardize().
    `want` lets a caller request explicit names (e.g. add 'C3' for a future AUX electrode)."""
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("ERROR")
    f = eegbci.load_data(subject, [4], update_path=True)
    raw = mne.io.read_raw_edf(f[0], preload=False)
    eegbci.standardize(raw)
    names = set(raw.ch_names)
    if want:
        missing = [c for c in want if c not in names]
        if missing:
            raise RuntimeError(f"requested channels not in EEGMMIDB montage: {missing}")
        return list(want), f"explicit channels {want}"
    if {"T9", "T10"} <= names:
        return ["T9", "AF7", "AF8", "T10"], "T9/T10 stand in for Muse TP9/TP10 (nearest temporal sites)"
    if {"TP7", "TP8"} <= names:
        return ["TP7", "AF7", "AF8", "TP8"], "T9/T10 absent; fell back to TP7/TP8 for TP9/TP10"
    raise RuntimeError(f"no Muse-proxy channels in montage; have AF7/AF8={'AF7' in names}/{'AF8' in names}")


def frontal_indices(channels: list[str]) -> list[int]:
    """Indices of the frontal (EOG-prone) channels within a channel list."""
    return [i for i, c in enumerate(channels) if c in FRONTAL]


# --- cue scheduler (pure) -------------------------------------------------------------------------------------------
def build_cue_order(trials: int, seed: int = 0, runs: int = 1, executed_per_run: int = 0) -> tuple[list[str], list[int]]:
    """Balanced, per-run randomized order of 'rest'/'imagine' (+ optional 'executed' real-clench trials).
    Returns (labels, run index per trial). Each run holds trials//runs trials, balanced, shuffled independently."""
    runs = max(1, runs)
    per_run = max(2, trials // runs)
    labels, run_ids = [], []
    rng = np.random.default_rng(seed)
    for r in range(runs):
        n_im = per_run // 2
        block = ["imagine"] * n_im + ["rest"] * (per_run - n_im) + ["executed"] * executed_per_run
        rng.shuffle(block)
        labels += block
        run_ids += [r] * len(block)
    return labels, run_ids


# --- signal processing (pure) ---------------------------------------------------------------------------------------
def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resample (ch, n) from fs_in to fs_out. Identity when the rates match."""
    from scipy.signal import resample_poly

    fin, fout = int(round(fs_in)), int(round(fs_out))
    if fin == fout:
        return np.asarray(x, float)
    g = math.gcd(fin, fout)
    return resample_poly(np.asarray(x, float), fout // g, fin // g, axis=1)


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase Butterworth band-pass of (ch, n). hi is capped below Nyquist."""
    from scipy.signal import butter, sosfiltfilt

    lo, hi = band
    hi = min(hi, 0.45 * fs)
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    x = np.asarray(x, float)
    if x.shape[1] <= 3 * sos.shape[0]:  # too short for filtfilt padding
        return x
    return sosfiltfilt(sos, x, axis=1)


def sliding_windows(n: int, fs: float, win_s: float, step_s: float, skip_s: float) -> list[tuple[int, int]]:
    """(start, end) index pairs for equal-length windows, starting after skip_s."""
    win, step, skip = int(round(win_s * fs)), max(1, int(round(step_s * fs))), int(round(skip_s * fs))
    if win <= 0:
        return []
    out, a = [], skip
    while a + win <= n:
        out.append((a, a + win))
        a += step
    return out


def process_trial(raw: np.ndarray, board_fs: float, model_fs: float, band: tuple[float, float],
                  win_s: float, step_s: float = STEP_S, skip_s: float = SKIP_S):
    """Resample a raw trial (ch, n) to the model fs, band-pass, then slice sliding windows. The band-pass runs on
    the whole resampled trial (consistent with how the EEGMMIDB epochs were filtered); for true real-time this
    would run causally per chunk. Returns (windows (m, ch, nt), window-centre times in s from trial start)."""
    x = bandpass(resample_to(raw, board_fs, model_fs), model_fs, band)
    wins = sliding_windows(x.shape[1], model_fs, win_s, step_s, skip_s)
    if not wins:
        return np.zeros((0, raw.shape[0], 0)), []
    W = np.stack([x[:, a:b] for a, b in wins]).astype(np.float64)
    centres = [((a + b) / 2) / model_fs for a, b in wins]
    return W, centres


def euclidean_align(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-session Euclidean alignment (He & Wu 2020): whiten each window by R^-1/2, R = mean covariance of the
    session. Unsupervised (no labels), so it is legitimate to compute on the eval session itself. Band-pass first."""
    if len(X) == 0:
        return X
    covs = np.array([x @ x.T / x.shape[1] for x in X])
    R = covs.mean(axis=0)
    R = R + eps * (np.trace(R) / R.shape[0]) * np.eye(R.shape[0])
    from scipy.linalg import fractional_matrix_power

    P = np.real(fractional_matrix_power(R, -0.5))
    return np.stack([P @ x for x in X])


def predict_windows(model, W: np.ndarray) -> np.ndarray:
    """p(imagine) per window. Falls back to the hard label if the model has no predict_proba."""
    if len(W) == 0:
        return np.zeros(0)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(W))[:, 1]
    return model.predict(W).astype(float)


def emg_rms(raw: np.ndarray, fs: float, band: tuple[float, float] = EMG_GUARD_BAND) -> np.ndarray:
    """Per-channel RMS of high-frequency (muscle) power. (ch,)"""
    return np.sqrt(np.mean(bandpass(raw, fs, band) ** 2, axis=1))


def emg_metric(raw: np.ndarray, fs: float) -> float:
    """One EMG number per trial: the strongest channel (jaw hits TP9/TP10, frown hits AF7/AF8)."""
    r = emg_rms(np.asarray(raw, float), fs)
    return float(np.max(r)) if r.size else 0.0


# --- accuracy / CI / significance (pure) ----------------------------------------------------------------------------
def balanced_accuracy(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    accs = [float((yp[yt == c] == c).mean()) for c in (0, 1) if np.any(yt == c)]
    return float(np.mean(accs)) if accs else float("nan")


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - conf) / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def binom_chance_threshold(n: int, alpha: float = 0.05) -> float:
    """Smallest accuracy that is above chance at one-sided p<alpha for n 2-class trials (binomial, p0=0.5)."""
    if n <= 0:
        return 1.0
    from scipy.stats import binom

    k = 0
    while k <= n and binom.sf(k - 1, n, 0.5) > alpha:
        k += 1
    return k / n


def label_permutation_p(y_true, y_pred, n_perm: int = 2000, seed: int = 0) -> float:
    """Permutation p-value: how often shuffled labels reach the observed balanced accuracy for these predictions."""
    yt = np.asarray(y_true)
    obs = balanced_accuracy(yt, y_pred)
    if math.isnan(obs):
        return 1.0
    rng = np.random.default_rng(seed)
    ge = sum(balanced_accuracy(rng.permutation(yt), y_pred) >= obs for _ in range(n_perm))
    return (ge + 1) / (n_perm + 1)


# --- pipelines ------------------------------------------------------------------------------------------------------
class BandPower:
    """Transformer: (n, ch, t) -> log band-power per channel over SUBBANDS (features (n, ch*len(bands)))."""

    def __init__(self, bands, fs):
        self.bands, self.fs = bands, fs

    def get_params(self, deep=True):
        return {"bands": self.bands, "fs": self.fs}

    def set_params(self, **p):
        self.__dict__.update(p)
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        n, ch, t = X.shape
        freqs = np.fft.rfftfreq(t, 1 / self.fs)
        P = np.abs(np.fft.rfft(X * np.hanning(t), axis=2)) ** 2
        feats = [np.log(P[:, :, (freqs >= lo) & (freqs < hi)].mean(axis=2) + 1e-12) for lo, hi in self.bands]
        return np.concatenate(feats, axis=1)


def mi_pipelines(fs: float, subbands=SUBBANDS) -> dict:
    """Riemann TS + LR and CSP + LDA (reused from train_mi) plus a log band-power + shrinkage-LDA baseline."""
    from mne.decoding import CSP
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return {
        "csp_lda": lambda: make_pipeline(CSP(n_components=4, log=True), LinearDiscriminantAnalysis()),
        "riemann_ts_lr": lambda: make_pipeline(Covariances("oas"), TangentSpace(),
                                               LogisticRegression(max_iter=1000)),
        "bandpower_lda": lambda: make_pipeline(BandPower(subbands, fs),
                                               LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    }


def bandpower_lda(fs: float, bands) -> object:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline

    return make_pipeline(BandPower(bands, fs), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


# --- window assembly (pure) -----------------------------------------------------------------------------------------
def session_windows(trials: list[dict], model_fs: float, band: tuple[float, float], win_s: float,
                    align: bool = True, channels_idx: list[int] | None = None,
                    step_s: float = STEP_S, skip_s: float = SKIP_S, labels=("rest", "imagine")):
    """Windows + labels + per-trial group (original trial index) + per-run group for ONE session. Euclidean
    alignment, when on, is applied over the whole session's windows. channels_idx selects a channel subset."""
    Xs, ys, tg, rg = [], [], [], []
    for li, tr in enumerate(trials):
        if tr["label"] not in labels:
            continue
        raw = np.asarray(tr["raw"], float)
        if channels_idx is not None:
            raw = raw[channels_idx]
        W, _ = process_trial(raw, tr.get("fs", model_fs), model_fs, band, win_s, step_s, skip_s)
        if len(W) == 0:
            continue
        Xs.append(W)
        ys.append(np.full(len(W), 1 if tr["label"] == "imagine" else 0))
        tg.append(np.full(len(W), li))
        rg.append(np.full(len(W), int(tr.get("run", 0))))
    if not Xs:
        n_ch = len(channels_idx) if channels_idx else (np.asarray(trials[0]["raw"]).shape[0] if trials else 4)
        return np.zeros((0, n_ch, 0)), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    X = np.concatenate(Xs)
    if align:
        X = euclidean_align(X)
    return X, np.concatenate(ys), np.concatenate(tg), np.concatenate(rg)


def assemble_sessions(sessions: list[list[dict]], model_fs: float, band: tuple[float, float], win_s: float,
                      align: bool, channels_idx: list[int] | None = None):
    """Per-session windows (EA per session), concatenated with globally-unique trial/run/session group ids."""
    Xa, ya, tga, rga, sga = [], [], [], [], []
    t_off = 0
    for si, trials in enumerate(sessions):
        X, y, tg, rg = session_windows(trials, model_fs, band, win_s, align=align, channels_idx=channels_idx)
        if len(X) == 0:
            continue
        Xa.append(X)
        ya.append(y)
        tga.append(tg + t_off)
        rga.append(si * 100000 + rg)
        sga.append(np.full(len(X), si))
        t_off = int(tga[-1].max()) + 1
    if not Xa:
        return (np.zeros((0, 4, 0)), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int))
    return (np.concatenate(Xa), np.concatenate(ya), np.concatenate(tga),
            np.concatenate(rga), np.concatenate(sga))


# --- session evaluation (pure) --------------------------------------------------------------------------------------
def _trial_stats(records: list[dict], n_perm: int = 2000, seed: int = 0) -> dict:
    if not records:
        return {"n": 0}
    pred = [1 if r["mean_p"] >= 0.5 else 0 for r in records]
    true = [r["y"] for r in records]
    correct = int(sum(int(a == b) for a, b in zip(pred, true)))
    lo, hi = wilson_ci(correct, len(records))
    return {"n": len(records), "n_imagine": int(sum(true)), "correct": correct,
            "accuracy": round(correct / len(records), 4),
            "balanced_accuracy": round(balanced_accuracy(true, pred), 4),
            "ci95": [round(lo, 4), round(hi, 4)], "chance": 0.5,
            "chance_threshold_p05": round(binom_chance_threshold(len(records)), 4),
            "permutation_p": round(label_permutation_p(true, pred, n_perm, seed), 4)}


def evaluate_session(trials: list[dict], model_meta: dict, step_s: float = STEP_S, skip_s: float = SKIP_S,
                     n_perm: int = 2000) -> tuple[dict, list[dict]]:
    """trials: [{'label','raw' (ch, n),'fs'}]. Honest report: window-level balanced accuracy; trial-level accuracy
    with a binomial 95% CI, the p<0.05 chance threshold and a permutation p-value vs 50% chance; and the EMG-guard
    split (accuracy with/without trials whose muscle power is >3 SD above the session's rest)."""
    model, model_fs = model_meta["model"], model_meta["fs"]
    band, win_s = tuple(model_meta["band"]), model_meta["win_s"]
    align = bool(model_meta.get("align", False))
    X, y, tg, _rg = session_windows(trials, model_fs, band, win_s, align=align, step_s=step_s, skip_s=skip_s)
    p = predict_windows(model, X)

    per = []
    for li, tr in enumerate(trials):
        if tr["label"] not in ("rest", "imagine"):
            continue
        mask = tg == li
        pv = p[mask]
        per.append({"trial": li, "label": tr["label"], "y": 1 if tr["label"] == "imagine" else 0,
                    "n_windows": int(mask.sum()), "mean_p": float(np.mean(pv)) if len(pv) else float("nan"),
                    "emg": emg_metric(tr["raw"], tr.get("fs", model_fs)),
                    "p": [round(float(v), 4) for v in pv]})
    valid = [r for r in per if r["n_windows"] > 0]

    rest_emg = np.array([r["emg"] for r in valid if r["label"] == "rest"])
    if rest_emg.size:
        thr = float(rest_emg.mean() + EMG_GUARD_SD * rest_emg.std()) if rest_emg.std() > 0 \
            else EMG_GUARD_RATIO * float(np.median(rest_emg))
    else:
        thr = float("inf")
    for r in per:
        r["emg_flag"] = bool(r["emg"] > thr)

    win_pred = [int(v >= 0.5) for v in p]
    flagged = [r for r in valid if r["emg_flag"]]
    summary = {
        "n_trials": len([t for t in trials if t["label"] in ("rest", "imagine")]),
        "n_valid_trials": len(valid), "n_windows": int(len(y)), "chance": 0.5,
        "model_fs": model_fs, "band": list(band), "win_s": win_s, "euclidean_align": align,
        "window_level": {"balanced_accuracy": round(balanced_accuracy(y, win_pred), 4), "n_windows": int(len(y)),
                         "note": "windows within a trial are correlated; trial-level stats are the honest ones"},
        "trial_level": _trial_stats(valid, n_perm),
        "emg_guard": {"band_hz": list(EMG_GUARD_BAND), "sd_above_rest": EMG_GUARD_SD,
                      "threshold_rms": round(thr, 4) if math.isfinite(thr) else None, "n_flagged": len(flagged),
                      "note": "high 30-100 Hz power vs rest = real muscle (jaw/forehead), not imagery",
                      "trial_level_unflagged": _trial_stats([r for r in valid if not r["emg_flag"]], n_perm)},
    }
    return summary, per


# --- UI: headless or tkinter (tkinter only off :0; tests use headless) ----------------------------------------------
class HeadlessCue:
    """Prints cues and a countdown. Used in tests and with --headless."""

    def intro(self, msg):
        _say(msg)

    def set_cue(self, big, sub, trial_i, n):
        _say(f"[{trial_i + 1}/{n}] {big}  ({sub})")

    def prompt(self, remaining, total=1.0):
        print(f"\r    {remaining:4.1f}s ", end="", flush=True)

    def done_line(self):
        print()

    def close(self):
        pass


class TkCue:
    """Big centred cue text + countdown on the current DISPLAY. NEVER used in tests (headless only)."""

    def __init__(self, title="Muse MI"):
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="black")
        self.root.geometry("900x600")
        self.big = tk.Label(self.root, text="", font=("Helvetica", 72, "bold"), fg="white", bg="black")
        self.big.pack(expand=True)
        self.sub = tk.Label(self.root, text="", font=("Helvetica", 22), fg="#aaaaaa", bg="black")
        self.sub.pack()
        self.count = tk.Label(self.root, text="", font=("Helvetica", 36), fg="#66ccff", bg="black")
        self.count.pack(pady=20)
        self.root.update()

    def intro(self, msg):
        self.big.config(text="Get ready", fg="white")
        self.sub.config(text=msg)
        self.count.config(text="")
        self.root.update()

    def set_cue(self, big, sub, trial_i, n):
        colour = {"IMAGINE": "#44dd88", "CLENCH (REAL)": "#ffaa44"}.get(big, "#dddddd")
        self.big.config(text=big, fg=colour)
        self.sub.config(text=f"[{trial_i + 1}/{n}]  {sub}")
        self.root.update()

    def prompt(self, remaining, total=1.0):
        self.count.config(text=f"{remaining:4.1f}s")
        self.root.update()

    def done_line(self):
        self.count.config(text="")
        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def make_ui(headless, title="Muse MI"):
    if headless or not os.environ.get("DISPLAY"):
        return HeadlessCue()
    try:
        return TkCue(title)
    except Exception as e:  # no tk, or display refused: never crash a live run
        print(f"(tk unavailable, using headless cues: {e})", flush=True)
        return HeadlessCue()


# --- cue protocol runtime -------------------------------------------------------------------------------------------
CUES = {
    "imagine": ("IMAGINE", "imagine squeezing your RIGHT hand into a fist. Do NOT move. Do NOT clench your jaw."),
    "rest": ("REST", "relax. Do nothing. Let your hand and face stay still."),
    "executed": ("CLENCH (REAL)", "actually squeeze your RIGHT hand into a fist now."),
}


def run_protocol(board_kind, trials, cue_s, rest_s, seed, headless, runs=1, executed_per_run=0,
                 jitter_s=1.0, model_meta=None, settle_s=3.0):
    """Run the cue protocol on a real/synthetic board. Returns (trial records, board_fs, labels, run ids).
    Each trial record is {'label','raw' (ch, n),'fs','run'}. The inter-trial pause is jittered (rest_s..rest_s+jitter).
    If model_meta is given, prints live p per trial."""
    board, fs, chans = open_board(board_kind)
    order, run_ids = build_cue_order(trials, seed, runs, executed_per_run)
    ui = make_ui(headless)
    rng = np.random.default_rng(seed)
    collected = []
    try:
        ui.intro(f"{len(order)} cues over {runs} run(s), {cue_s:.0f} s each. Settling {settle_s:.0f} s.")
        time.sleep(min(settle_s, 5.0))
        _record(board, chans, 0.2)  # drop start-up junk
        for i, (label, run) in enumerate(zip(order, run_ids)):
            big, sub = CUES[label]
            ui.set_cue("+", "relax", i, len(order))
            _record(board, chans, rest_s + float(rng.uniform(0, jitter_s)))  # jittered pause, not recorded
            ui.set_cue(big, sub, i, len(order))
            raw = _record(board, chans, cue_s, lambda rem, tot=cue_s: ui.prompt(rem, tot))
            ui.done_line()
            collected.append({"label": label, "raw": raw, "fs": float(fs), "run": int(run)})
            if model_meta is not None and raw.shape[1] and label in ("rest", "imagine"):
                W, _ = process_trial(raw, float(fs), model_meta["fs"], tuple(model_meta["band"]), model_meta["win_s"])
                if model_meta.get("align"):
                    W = euclidean_align(W)
                p = predict_windows(model_meta["model"], W)
                mp = float(np.mean(p)) if len(p) else float("nan")
                print(f"    cue={label:<9} windows={len(p):2d} mean p(imagine)={mp:.2f}", flush=True)
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass
        ui.close()
    return collected, float(fs), order, run_ids


def save_session(out_dir, trials, board_fs, order, run_ids, summary, per_trial, model_meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"trial{idx:03d}": np.asarray(t["raw"], float) for idx, t in enumerate(trials)}
    np.savez_compressed(out_dir / "trials.npz", labels=np.array(order),
                        runs=np.array(run_ids, int), fs=board_fs, **arrays)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "board_fs": board_fs, "channels": MUSE_CHANNELS,
            "n_trials": len(trials), "labels": order, "runs": list(run_ids)}
    if model_meta is not None:
        meta["model"] = {k: model_meta.get(k) for k in ("fs", "band", "win_s", "channels", "align")}
    if summary is not None:
        meta["summary"] = summary
    if per_trial is not None:
        meta["per_trial"] = [{k: v for k, v in r.items() if k != "p" or len(r.get("p", [])) <= 200}
                             for r in per_trial]
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def load_session(session_dir) -> tuple[list[dict], float, list[str], list[int]]:
    d = np.load(Path(session_dir) / "trials.npz")
    order = [str(x) for x in d["labels"]]
    fs = float(d["fs"])
    runs = [int(x) for x in d["runs"]] if "runs" in d.files else [0] * len(order)
    trials = [{"label": order[i], "raw": d[f"trial{i:03d}"], "fs": fs, "run": runs[i]} for i in range(len(order))]
    return trials, fs, order, runs


# --- train-base: EEGMMIDB, Muse-proxy vs motor-8 + EOG control ------------------------------------------------------
def load_subject_multiband(subject, specs, runs=(4, 8, 12), sfreq=160.0, epoch=(0.5, 2.5)):
    """One EDF read per subject, several (channels, band) epoch sets out of it. specs: {name: (channels, band)}.
    Returns (dict name -> X (ep, ch, t) float32, y (ep,))."""
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("WARNING")
    files = eegbci.load_data(subject, list(runs), update_path=True)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq)
    events, ev_id = mne.events_from_annotations(raw, event_id={"T0": 0, "T1": 1, "T2": 2})
    out, y = {}, None
    for name, (channels, band) in specs.items():
        r = raw.copy().pick(list(channels)).filter(*band, fir_design="firwin")
        ep = mne.Epochs(r, events, ev_id, tmin=epoch[0], tmax=epoch[1], baseline=None, preload=True)
        out[name] = ep.get_data().astype(np.float32)
        y = (ep.events[:, 2] > 0).astype(int)
    return out, y


def _balanced_idx(y, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    neg = rng.choice(neg, size=min(len(neg), len(pos)), replace=False)
    return np.sort(np.concatenate([pos, neg]))


def _within_loso(data, subs, key, make_pipes):
    """within-subject 5-fold CV + leave-subjects-out (5 folds), for each pipeline in make_pipes (name->factory)."""
    from sklearn.model_selection import cross_val_score

    within = {k: [] for k in make_pipes}
    for s in subs:
        X, y = data[s][key], data[s]["y"]
        for k, make in make_pipes.items():
            within[k].append(float(cross_val_score(make(), X, y, cv=5).mean()))
    folds = np.array_split(np.array(subs), min(5, len(subs)))
    loso = {k: [] for k in make_pipes}
    for f in folds:
        tr = [s for s in subs if s not in f]
        Xtr = np.concatenate([data[s][key] for s in tr])
        ytr = np.concatenate([data[s]["y"] for s in tr])
        for k, make in make_pipes.items():
            m = make().fit(Xtr, ytr)
            loso[k] += [float(m.score(data[s][key], data[s]["y"])) for s in f]
    return {k: {"within_subject_cv": round(float(np.mean(within[k])), 4),
                "within_std": round(float(np.std(within[k])), 4),
                "within_ge_0.60": f"{int(np.sum(np.array(within[k]) >= 0.60))}/{len(subs)}",
                "leave_subjects_out": round(float(np.mean(loso[k])), 4),
                "loso_std": round(float(np.std(loso[k])), 4)} for k in make_pipes}


def cmd_train_base(args) -> int:
    import joblib

    from .data import MOTOR_8
    from .train_mi import parse_range

    subs = parse_range(args.subjects)
    want = [c.strip() for c in args.channels.split(",")] if args.channels else None
    proxy, note = resolve_proxy_channels(subs[0], want)
    frontal = [c for c in proxy if c in FRONTAL] or proxy[:1]
    specs = {"muse_proxy": (proxy, BAND), "motor_8": (MOTOR_8, BAND), "eog_control": (frontal, EOG_CONTROL_BAND)}

    mi = {k: v for k, v in mi_pipelines(160.0).items()}
    eog = {"eog_bandpower_lda": lambda: bandpower_lda(160.0, [EOG_CONTROL_BAND])}

    data, kept, skipped = {}, [], []
    for s in subs:
        try:
            out, y = load_subject_multiband(s, specs)
        except Exception as e:
            print(f"subject {s}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            skipped.append(s)
            continue
        idx = _balanced_idx(y)
        data[s] = {"y": y[idx], **{name: out[name][idx] for name in specs}}
        kept.append(s)
        print(f"subject {s}: n={len(idx)}", flush=True)

    if not kept:
        print("no subjects loaded; aborting")
        return 1

    results = {"muse_proxy": _within_loso(data, kept, "muse_proxy", mi),
               "motor_8": _within_loso(data, kept, "motor_8", mi),
               "eog_control": _within_loso(data, kept, "eog_control", eog)}
    main_loso = results["muse_proxy"][args.model]["leave_subjects_out"]
    eog_loso = results["eog_control"]["eog_bandpower_lda"]["leave_subjects_out"]
    chance_hi = binom_chance_threshold(len(kept))
    verdict = ("EOG control matches/beats the main model: the Muse-proxy 'imagery' accuracy may be eye artifact, "
               "not motor imagery." if eog_loso >= main_loso - 0.03 and eog_loso > 0.5 + 1e-6 else
               "EOG control is below the main model: the main model is not merely decoding eye artifact.")

    report = {"task": "imagined fist (T1/T2) vs rest (T0), EEGMMIDB imagery runs 4/8/12",
              "fs": 160, "band": list(BAND), "epoch_s": [0.5, 2.5], "win_s": BASE_WIN_S, "chance": 0.5,
              "n_subjects": len(kept), "subjects": kept, "skipped": skipped,
              "channel_sets": {"muse_proxy": proxy, "motor_8": MOTOR_8, "eog_control": frontal},
              "proxy_note": note, "results": results,
              "eog_control_verdict": verdict,
              "eog_control_loso": eog_loso, "main_loso": main_loso, "main_model": args.model}
    print(json.dumps(report, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    Xall = np.concatenate([data[s]["muse_proxy"] for s in kept])
    yall = np.concatenate([data[s]["y"] for s in kept])
    model = mi[args.model]().fit(Xall, yall)
    joblib.dump({"model": model, "fs": 160, "channels": proxy, "band": list(BAND),
                 "win_s": BASE_WIN_S, "align": False, "report": report}, out)
    print("saved", out)
    return 0


# --- train-personal -------------------------------------------------------------------------------------------------
def _grouped_cv_trial(X, y, groups, trial_groups, make, trials_by_tg, board_fs, n_splits):
    """Leave-group-out CV, then a trial-level vote (mean window pred per trial). Returns (trial_true, trial_pred,
    window_preds). groups = run id (leave-one-run-out); trial_groups keep each trial's windows together."""
    from sklearn.model_selection import GroupKFold

    preds = np.zeros(len(y), int)
    for tr_i, te_i in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make().fit(X[tr_i], y[tr_i])
        preds[te_i] = m.predict(X[te_i])
    t_true, t_pred = [], []
    for tg in np.unique(trial_groups):
        mask = trial_groups == tg
        t_true.append(int(y[mask][0]))
        t_pred.append(int(preds[mask].mean() >= 0.5))
    return t_true, t_pred, preds


def cmd_train_personal(args) -> int:
    import joblib

    raw_sessions = [load_session(Path(d)) for d in args.sessions]
    sessions = [trials for trials, _fs, _o, _r in raw_sessions]
    fs_set = {round(fs) for _t, fs, _o, _r in raw_sessions}
    if len(fs_set) != 1:
        print(f"sessions have mixed sample rates {fs_set}; train one rate at a time")
        return 1
    board_fs = float(fs_set.pop())
    model_fs, win_s, align = board_fs, args.win_s, not args.no_align

    X, y, tg, rg, sg = assemble_sessions(sessions, model_fs, BAND, win_s, align)
    n_mi_trials = sum(1 for s in sessions for t in s if t["label"] in ("rest", "imagine"))
    if len(X) == 0 or len(np.unique(y)) < 2:
        print(f"not enough separable windows (mi trials={n_mi_trials}, windows={len(X)}, classes={np.unique(y)})")
        return 1

    # trial-group map for the vote; group for CV is the run (leave-one-run-out) if >=2 runs else the trial
    cv_groups = rg if len(np.unique(rg)) >= 2 else tg
    cv_kind = "leave-one-run-out" if len(np.unique(rg)) >= 2 else "leave-one-trial-out"
    n_splits = max(2, min(5, len(np.unique(cv_groups))))
    # trials in the same global-trial-id order as np.unique(tg) (sessions concatenated in order)
    flat_trials = [t for s in sessions for t in s if t["label"] in ("rest", "imagine")]

    chance_hi = binom_chance_threshold(len(np.unique(tg)))
    report = {"fs": model_fs, "band": list(BAND), "win_s": win_s, "chance": 0.5,
              "euclidean_align": align, "cv_kind": cv_kind, "n_sessions": len(sessions),
              "n_mi_trials": n_mi_trials, "n_windows": int(len(X)),
              "n_imagine_trials": int(sum(1 for t in flat_trials if t["label"] == "imagine")),
              "chance_threshold_p05": round(chance_hi, 4), "sessions": [str(d) for d in args.sessions], "cv": {}}

    pipes = mi_pipelines(model_fs)
    for name, make in pipes.items():
        t_true, t_pred, win_preds = _grouped_cv_trial(X, y, cv_groups, tg, make, None, board_fs, n_splits)
        correct = int(sum(int(a == b) for a, b in zip(t_true, t_pred)))
        lo, hi = wilson_ci(correct, len(t_true))
        # EMG guard on the trials (order matches np.unique(tg))
        t_emg = [emg_metric(np.asarray(flat_trials[i]["raw"], float), board_fs) for i in range(len(t_true))]
        rest_emg = np.array([e for e, lab in zip(t_emg, t_true) if lab == 0])
        thr = (rest_emg.mean() + EMG_GUARD_SD * rest_emg.std()) if rest_emg.size and rest_emg.std() > 0 else (
            EMG_GUARD_RATIO * float(np.median(rest_emg)) if rest_emg.size else float("inf"))
        keep = [i for i, e in enumerate(t_emg) if e <= thr]
        uc = int(sum(int(t_true[i] == t_pred[i]) for i in keep))
        ulo, uhi = wilson_ci(uc, len(keep)) if keep else (0.0, 1.0)
        report["cv"][name] = {
            "window_balanced_accuracy": round(balanced_accuracy(y, win_preds), 4),
            "trial_accuracy": round(correct / len(t_true), 4),
            "trial_balanced_accuracy": round(balanced_accuracy(t_true, t_pred), 4),
            "trial_ci95": [round(lo, 4), round(hi, 4)], "n_trials": len(t_true),
            "permutation_p": round(label_permutation_p(t_true, t_pred), 4),
            "emg_guard": {"n_flagged": len(t_true) - len(keep), "threshold_rms": round(float(thr), 4)
                          if math.isfinite(thr) else None,
                          "trial_accuracy_unflagged": round(uc / len(keep), 4) if keep else None,
                          "trial_ci95_unflagged": [round(ulo, 4), round(uhi, 4)]}}

    # artifact control decoders on the SAME trials: EMG-only (>30 Hz) and EOG-only (<4 Hz frontal)
    controls = {}
    Xe, ye, tge, rge, _sge = assemble_sessions(sessions, model_fs, EMG_CONTROL_BAND, win_s, False)
    frontal_idx = frontal_indices(MUSE_CHANNELS)
    Xo, yo, tgo, rgo, _sgo = assemble_sessions(sessions, model_fs, EOG_CONTROL_BAND, win_s, False,
                                               channels_idx=frontal_idx)
    for cname, (Xc, yc, tgc, rgc, bands) in {
        "emg_only": (Xe, ye, tge, rge, [EMG_CONTROL_BAND]),
        "eog_only": (Xo, yo, tgo, rgo, [EOG_CONTROL_BAND]),
    }.items():
        if len(Xc) == 0 or len(np.unique(yc)) < 2:
            controls[cname] = {"note": "no windows"}
            continue
        cg = rgc if len(np.unique(rgc)) >= 2 else tgc
        ns = max(2, min(5, len(np.unique(cg))))
        tt, tp, _ = _grouped_cv_trial(Xc, yc, cg, tgc, lambda b=bands: bandpower_lda(model_fs, b),
                                      None, board_fs, ns)
        c = int(sum(int(a == b) for a, b in zip(tt, tp)))
        controls[cname] = {"trial_accuracy": round(c / len(tt), 4),
                           "trial_balanced_accuracy": round(balanced_accuracy(tt, tp), 4), "n_trials": len(tt)}

    main_acc = report["cv"][args.model]["trial_balanced_accuracy"]
    susp = [c for c in ("emg_only", "eog_only")
            if controls.get(c, {}).get("trial_balanced_accuracy", 0) >= main_acc - 0.03
            and controls[c].get("trial_balanced_accuracy", 0) > 0.5 + 1e-6]
    report["artifact_controls"] = controls
    report["artifact_verdict"] = ("WARNING: control(s) " + ", ".join(susp) + " match the main model; it is likely "
                                  "decoding artifact, not imagery." if susp else
                                  "Controls are below the main model: not explained by EMG/EOG artifact alone.")

    if len(sessions) >= 2:  # cross-session (leave-one-session-out) for the main model
        from sklearn.model_selection import LeaveOneGroupOut

        preds = np.zeros(len(y), int)
        for tr_i, te_i in LeaveOneGroupOut().split(X, y, sg):
            preds[te_i] = pipes[args.model]().fit(X[tr_i], y[tr_i]).predict(X[te_i])
        tt, tp = [], []
        for tgi in np.unique(tg):
            m = tg == tgi
            tt.append(int(y[m][0]))
            tp.append(int(preds[m].mean() >= 0.5))
        report["cross_session"] = {"model": args.model, "n_sessions": len(sessions),
                                   "trial_balanced_accuracy": round(balanced_accuracy(tt, tp), 4),
                                   "trial_accuracy": round(sum(int(a == b) for a, b in zip(tt, tp)) / len(tt), 4)}

    print(json.dumps(report, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    model = pipes[args.model]().fit(X, y)
    joblib.dump({"model": model, "fs": model_fs, "channels": MUSE_CHANNELS, "band": list(BAND),
                 "win_s": win_s, "align": align, "report": report}, out)
    print("saved", out, "(personal model: gitignored under models/, it is your biometric data)")
    return 0


# --- live-eval / collect commands -----------------------------------------------------------------------------------
def _load_model(path) -> dict:
    import joblib

    meta = joblib.load(path)
    meta.setdefault("band", list(BAND))
    meta.setdefault("win_s", BASE_WIN_S)
    meta.setdefault("align", False)
    return meta


def cmd_live_eval(args) -> int:
    meta = _load_model(args.model)
    print(f"model: fs={meta['fs']} channels={meta.get('channels')} band={meta['band']} "
          f"win_s={meta['win_s']} align={meta['align']}", flush=True)
    trials, board_fs, order, run_ids = run_protocol(args.board, args.trials, args.cue_s, args.rest_s, args.seed,
                                                     args.headless, runs=args.runs, executed_per_run=args.executed,
                                                     model_meta=meta)
    summary, per = evaluate_session(trials, meta)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = save_session(Path(args.out_root) / f"{ts}_eval", trials, board_fs, order, run_ids, summary, per, meta)
    tl, eg = summary["trial_level"], summary["emg_guard"]["trial_level_unflagged"]
    print("\n=== live-eval ===")
    print(f"board {board_fs} Hz, {summary['n_valid_trials']}/{summary['n_trials']} valid trials, "
          f"{summary['n_windows']} windows, EA={summary['euclidean_align']}")
    print(f"window balanced accuracy: {summary['window_level']['balanced_accuracy']:.3f} (chance 0.50)")
    print(f"trial accuracy:           {tl.get('accuracy')} (95% CI {tl.get('ci95')}, "
          f"p<.05 if >= {tl.get('chance_threshold_p05')}, perm p={tl.get('permutation_p')}, n={tl.get('n')})")
    print(f"EMG-flagged trials: {summary['emg_guard']['n_flagged']}  ->  "
          f"trial accuracy without them: {eg.get('accuracy')} (95% CI {eg.get('ci95')}, n={eg.get('n')})")
    print("saved", out)
    return 0


def cmd_collect(args) -> int:
    meta = _load_model(args.model) if args.model else None
    trials, board_fs, order, run_ids = run_protocol(args.board, args.trials, args.cue_s, args.rest_s, args.seed,
                                                     args.headless, runs=args.runs, executed_per_run=args.executed,
                                                     model_meta=meta)
    summary, per = (evaluate_session(trials, meta) if meta else (None, None))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = save_session(Path(args.out_root) / f"{ts}_collect", trials, board_fs, order, run_ids, summary, per, meta)
    counts = {k: sum(1 for t in trials if t["label"] == k) for k in ("imagine", "rest", "executed")}
    print(f"\ncollected {len(trials)} trials {counts} at {board_fs} Hz over {args.runs} run(s)")
    print("saved", out)
    return 0


# --- cli ------------------------------------------------------------------------------------------------------------
def _add_protocol_args(p):
    p.add_argument("--board", choices=["muse2", "muse2-bled", "synthetic"], default="synthetic")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--runs", type=int, default=1, help="split the trials into this many randomized runs")
    p.add_argument("--executed", type=int, default=0, help="add this many REAL right-hand-clench trials per run")
    p.add_argument("--cue-s", type=float, default=4.0)
    p.add_argument("--rest-s", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="print cues instead of opening a window")
    p.add_argument("--out-root", default="data/eeg_live")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tb = sub.add_parser("train-base", help="base model on EEGMMIDB (Muse-proxy vs motor-8, + EOG control)")
    tb.add_argument("--subjects", default="1-20")
    tb.add_argument("--model", default="riemann_ts_lr")
    tb.add_argument("--channels", default="", help="comma list in Muse order (e.g. T9,AF7,AF8,T10,C3); default auto")
    tb.add_argument("--out", default="models/muse_proxy_mi.joblib")

    le = sub.add_parser("live-eval", help="cue protocol + run a model, with EMG guard and CI")
    le.add_argument("--model", required=True)
    _add_protocol_args(le)

    co = sub.add_parser("collect", help="cue protocol, save labelled raw trials")
    co.add_argument("--model", default="", help="optional: show live p while collecting")
    _add_protocol_args(co)

    tp = sub.add_parser("train-personal", help="train a Muse-native model on collected sessions")
    tp.add_argument("--sessions", nargs="+", required=True)
    tp.add_argument("--model", default="riemann_ts_lr")
    tp.add_argument("--win-s", type=float, default=1.0)
    tp.add_argument("--no-align", action="store_true", help="disable per-session Euclidean alignment")
    tp.add_argument("--out", default="models/muse_personal_mi.joblib")

    args = ap.parse_args(argv)
    return {"train-base": cmd_train_base, "live-eval": cmd_live_eval, "collect": cmd_collect,
            "train-personal": cmd_train_personal}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

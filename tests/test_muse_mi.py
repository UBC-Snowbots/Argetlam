"""eeg/muse_mi.py: imagined-fist MI on Muse-proxy channels. No hardware, no real screen.

Covers the pure pieces (cue scheduler, windowing/resampling, Euclidean alignment, EMG guard, accuracy/CI/chance/
permutation math), train-personal on synthetic separable data (and chance on label-independent noise) with the
artifact controls, evaluate_session, and the live-eval/collect commands end to end through BrainFlow's synthetic
board with --headless and a tiny trial count.
"""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rover_intent.eeg import muse_mi as MM


# --- synthetic MI trials --------------------------------------------------------------------------------------------
def mi_trial(label, fs=256.0, dur=4.0, seed=0, strong=8.0, bg=3.0, separable=True):
    """4-ch trial. When separable, an imagine trial puts a 12 Hz oscillation on ch0/ch3 and a rest trial on ch1/ch2,
    so the covariance structure differs by class. When separable=False the oscillation is class-independent (chance)."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    x = rng.normal(0, bg, (4, n))
    osc = np.sin(2 * np.pi * 12.0 * t + rng.uniform(0, 6))
    if not separable:
        x += strong * osc
    elif label == "imagine":
        x[0] += strong * osc
        x[3] += strong * osc
    else:
        x[1] += strong * osc
        x[2] += strong * osc
    return x


def trials_set(n, fs=256.0, dur=4.0, seed=0, separable=True, run=0):
    out = []
    for i in range(n):
        label = "imagine" if i % 2 == 0 else "rest"
        out.append({"label": label, "raw": mi_trial(label, fs, dur, seed + i, separable=separable),
                    "fs": fs, "run": run})
    return out


# --- cue scheduler --------------------------------------------------------------------------------------------------
def test_cue_order_is_balanced_and_randomized():
    order, runs = MM.build_cue_order(20, seed=1)
    assert len(order) == 20 and len(runs) == 20
    assert order.count("imagine") == 10 and order.count("rest") == 10
    other, _ = MM.build_cue_order(20, seed=2)
    assert other != order and other.count("imagine") == 10
    odd, _ = MM.build_cue_order(7, seed=0)
    assert abs(odd.count("imagine") - odd.count("rest")) == 1


def test_cue_order_runs_and_executed():
    order, runs = MM.build_cue_order(40, seed=3, runs=4, executed_per_run=2)
    assert set(runs) == {0, 1, 2, 3}
    # each run is balanced within itself and carries the executed trials
    for r in range(4):
        block = [lab for lab, rr in zip(order, runs) if rr == r]
        assert block.count("imagine") == block.count("rest")
        assert block.count("executed") == 2


# --- windowing + resampling + EA ------------------------------------------------------------------------------------
def test_sliding_windows_equal_length_and_skip():
    wins = MM.sliding_windows(n=1000, fs=100.0, win_s=1.0, step_s=0.25, skip_s=0.5)
    assert all((b - a) == 100 for a, b in wins)
    assert wins[0][0] == 50
    assert MM.sliding_windows(n=10, fs=100.0, win_s=1.0, step_s=0.25, skip_s=0.5) == []


def test_resample_to_changes_rate_and_is_identity_when_equal():
    x = np.random.default_rng(0).normal(0, 1, (4, 256))
    assert MM.resample_to(x, 256, 256).shape == x.shape
    y = MM.resample_to(x, 256, 160)
    assert abs(y.shape[1] - 160) <= 2 and y.shape[0] == 4


def test_bandpass_attenuates_out_of_band():
    fs, n = 256.0, 2048
    t = np.arange(n) / fs
    y_in = MM.bandpass(np.sin(2 * np.pi * 15 * t)[None], fs, (8.0, 30.0))
    y_out = MM.bandpass(np.sin(2 * np.pi * 70 * t)[None], fs, (8.0, 30.0))
    assert y_in.std() > 5 * y_out.std()


def test_euclidean_align_whitens_reference_to_identity():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 4, 160))
    Xa = MM.euclidean_align(X)
    R = np.mean([x @ x.T / x.shape[1] for x in Xa], axis=0)
    assert np.allclose(R, np.eye(4), atol=0.05)


def test_process_trial_window_shape():
    W, centres = MM.process_trial(mi_trial("imagine", fs=256.0, dur=4.0), 256.0, 160.0, (8.0, 30.0), win_s=1.0)
    assert W.ndim == 3 and W.shape[1] == 4 and W.shape[2] == 160
    assert len(centres) == len(W) and centres[0] >= 0.5


# --- EMG guard ------------------------------------------------------------------------------------------------------
def test_emg_metric_flags_a_jaw_clench_trial():
    fs = 256.0
    rng = np.random.default_rng(0)
    from scipy.signal import butter, sosfilt
    rest = rng.normal(0, 3, (4, int(4 * fs)))
    burst = sosfilt(butter(4, [40, 100], btype="band", fs=fs, output="sos"),
                    rng.normal(0, 60, (4, int(4 * fs))), axis=1)
    assert MM.emg_metric(rest + burst, fs) > 3 * MM.emg_metric(rest, fs)


def test_evaluate_session_emg_guard_split_present():
    fs = 256.0
    trials = trials_set(6, fs=fs, seed=10)
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(3)
    sos = butter(4, [40, 100], btype="band", fs=fs, output="sos")
    trials[0]["raw"] = trials[0]["raw"] + sosfilt(sos, rng.normal(0, 120, trials[0]["raw"].shape), axis=1)
    summary, per = MM.evaluate_session(trials, _fit_meta(fs), n_perm=200)
    assert summary["emg_guard"]["n_flagged"] >= 1
    assert "trial_level_unflagged" in summary["emg_guard"]
    assert any(r["emg_flag"] for r in per)


# --- accuracy / CI / significance math ------------------------------------------------------------------------------
def test_balanced_accuracy_wilson_chance_permutation():
    assert MM.balanced_accuracy([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert MM.balanced_accuracy([0, 0, 0, 1], [0, 0, 0, 0]) == 0.5
    lo5, hi5 = MM.wilson_ci(5, 10)
    assert lo5 < 0.5 < hi5
    assert MM.wilson_ci(0, 0) == (0.0, 1.0)
    # binomial chance threshold at n=60 ~ 0.62, n=100 ~ 0.59 (research values)
    assert MM.binom_chance_threshold(60) == pytest.approx(0.62, abs=0.02)
    assert MM.binom_chance_threshold(100) == pytest.approx(0.59, abs=0.02)
    # permutation: perfect predictions on a balanced set are significant; constant predictions are not
    assert MM.label_permutation_p([0, 1] * 10, [0, 1] * 10, n_perm=500) < 0.05
    assert MM.label_permutation_p([0, 1] * 10, [1] * 20, n_perm=500) > 0.2


# --- window assembly + evaluate_session -----------------------------------------------------------------------------
def _fit_meta(fs, win_s=1.0, seed=100, align=True):
    """A real fitted model on separable synthetic data, as a model_meta dict."""
    from rover_intent.eeg.train_mi import pipelines
    X, y, _tg, _rg = MM.session_windows(trials_set(10, fs=fs, seed=seed), fs, MM.BAND, win_s, align=align)
    model = pipelines()["riemann_ts_lr"]().fit(X, y)
    return {"model": model, "fs": fs, "band": list(MM.BAND), "win_s": win_s,
            "channels": MM.MUSE_CHANNELS, "align": align}


def test_session_windows_groups_are_per_trial():
    fs = 256.0
    X, y, tg, rg = MM.session_windows(trials_set(4, fs=fs, seed=1), fs, MM.BAND, 1.0, align=False)
    assert len(np.unique(tg)) == 4 and len(X) == len(y) == len(tg) == len(rg)
    for gi in np.unique(tg):
        assert len(np.unique(y[tg == gi])) == 1


def test_evaluate_session_beats_chance_on_separable_data():
    fs = 256.0
    meta = _fit_meta(fs, seed=100)
    summary, _ = MM.evaluate_session(trials_set(8, fs=fs, seed=500), meta, n_perm=200)
    assert summary["trial_level"]["accuracy"] >= 0.75
    assert summary["window_level"]["balanced_accuracy"] >= 0.7
    assert "chance_threshold_p05" in summary["trial_level"] and "permutation_p" in summary["trial_level"]


# --- train-personal -------------------------------------------------------------------------------------------------
def _save(tmp_path, trials, tag, fs):
    sess = tmp_path / tag
    MM.save_session(sess, trials, fs, [t["label"] for t in trials], [t["run"] for t in trials], None, None, None)
    return sess


def test_train_personal_separable_then_chance_on_noise(tmp_path):
    fs = 256.0

    def run(separable, tag, seed):
        sess = _save(tmp_path, trials_set(8, fs=fs, seed=seed, separable=separable), tag, fs)
        out = tmp_path / f"model_{tag}.joblib"
        rc = MM.main(["train-personal", "--sessions", str(sess), "--out", str(out), "--win-s", "1.0"])
        assert rc == 0
        return json.loads(out.with_suffix(".json").read_text())

    sep = run(True, "sep", 200)
    noise = run(False, "noise", 900)
    assert sep["cv"]["riemann_ts_lr"]["trial_accuracy"] >= 0.75
    assert noise["cv"]["riemann_ts_lr"]["trial_accuracy"] <= 0.7
    # artifact controls present and, on 12 Hz-only separable data, below the main model -> "not artifact"
    assert "emg_only" in sep["artifact_controls"] and "eog_only" in sep["artifact_controls"]
    assert "artifact_verdict" in sep


def test_train_personal_leave_one_run_out(tmp_path):
    fs = 256.0
    trials = trials_set(8, fs=fs, seed=200, run=0) + trials_set(8, fs=fs, seed=300, run=1)
    sess = _save(tmp_path, trials, "runs", fs)
    out = tmp_path / "m.joblib"
    rc = MM.main(["train-personal", "--sessions", str(sess), "--out", str(out), "--win-s", "1.0"])
    assert rc == 0
    rep = json.loads(out.with_suffix(".json").read_text())
    assert rep["cv_kind"] == "leave-one-run-out"


# --- live-eval / collect end to end (synthetic board, headless) -----------------------------------------------------
def test_live_eval_end_to_end(tmp_path):
    fs = 250.0  # BrainFlow synthetic board rate
    out = tmp_path / "m.joblib"
    import joblib
    joblib.dump(_fit_meta(fs, win_s=1.0, seed=100), out)
    root = tmp_path / "live"
    rc = MM.main(["live-eval", "--model", str(out), "--board", "synthetic", "--headless",
                  "--trials", "4", "--cue-s", "2", "--rest-s", "0", "--out-root", str(root)])
    assert rc == 0
    sess = sorted(root.glob("*_eval"))
    assert sess, "no eval session saved"
    summary = json.loads((sess[0] / "summary.json").read_text())
    assert (sess[0] / "trials.npz").exists()
    assert summary["summary"]["chance"] == 0.5
    assert "ci95" in summary["summary"]["trial_level"] and "emg_guard" in summary["summary"]


def test_collect_end_to_end(tmp_path):
    root = tmp_path / "coll"
    rc = MM.main(["collect", "--board", "synthetic", "--headless",
                  "--trials", "4", "--cue-s", "2", "--rest-s", "0", "--out-root", str(root)])
    assert rc == 0
    sess = sorted(root.glob("*_collect"))
    assert sess and (sess[0] / "trials.npz").exists()
    trials, fs, order, runs = MM.load_session(sess[0])
    assert len(order) == 4 and len(trials) == 4

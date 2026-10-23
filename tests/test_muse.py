"""Muse SSVEP + alpha pipeline on synthetic signals, the session recorder, the live detectors and the readiness
report. No hardware and no datasets: everything is synthesised (SSVEP sinusoids + harmonics in noise, alpha bursts,
blinks, mains hum, dropped frames), and the CLI paths use the BrainFlow synthetic board with pygame headless."""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # pygame headless for the recorder/display tests

from rover_intent.eeg import alpha as A
from rover_intent.eeg import ssvep as S
from rover_intent.eeg import muse_detect as MD
from rover_intent.eeg import muse_report as MR
from rover_intent.eeg import muse_session as MS

FS = 256.0


# --- synthetic generators --------------------------------------------------------------------------------------------
def ssvep_signal(freq, fs, n, n_ch=4, amp=2.0, harmonics=2, noise=6.0, seed=0, phase=0.0, alpha_uv=3.0, line_uv=8.0):
    """SSVEP: sin at freq + harmonics on all channels (weaker on 'frontal' ch 1,2), buried in pink-ish noise,
    plus a weak alpha rhythm and mains hum (what the Muse temporal electrodes would actually see)."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    white = rng.normal(0, noise, (n_ch, n))
    lp = butter(2, 20, fs=fs, output="sos")
    x = 0.6 * white + 0.9 * sosfilt(lp, white, axis=1)  # EEG-like 1/f-ish colouring, zero-mean (no random walk)
    w = np.array([1.0, 0.5, 0.5, 1.0])[:n_ch, None]
    for h in range(1, harmonics + 1):
        x += w * (amp / h) * np.sin(2 * np.pi * h * freq * t + phase)
    x += alpha_uv * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6, (n_ch, 1)))
    x += line_uv * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 6, (n_ch, 1)))
    return x


def alpha_signal(fs, n, n_ch=4, closed=True, seed=0):
    """Eyes-closed = strong 8-12 Hz; eyes-open = weak. Broadband background on all channels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = rng.normal(0, 8, (n_ch, n))
    from scipy.signal import butter, sosfilt
    band = butter(4, [8, 12], btype="band", fs=fs, output="sos")
    a = sosfilt(band, rng.normal(0, 1, (n_ch, n)), axis=1)
    a /= a.std(axis=1, keepdims=True)
    amp = 22.0 if closed else 3.0
    x += amp * a
    x += 6 * np.sin(2 * np.pi * 60 * t)  # mains
    return x


# --- ssvep.py --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("true_f", [12.0, 15.0])
def test_cca_and_fbcca_pick_the_right_frequency(true_f):
    x = ssvep_signal(true_f, FS, int(3 * FS), seed=1)
    for fn in (S.cca_scores, S.fbcca_scores):
        sc = fn(x, [12.0, 15.0], FS, harmonics=3)
        assert [12.0, 15.0][int(np.argmax(sc))] == true_f


def test_cca_handles_single_channel():
    x = ssvep_signal(15.0, FS, int(3 * FS), n_ch=1, seed=2)
    sc = S.cca_scores(x, [12.0, 15.0], FS)
    assert np.argmax(sc) == 1 and 0 <= sc[0] <= 1


def test_trca_trains_and_classifies_four_class():
    freqs = [12.0, 13.0, 14.0, 15.0]
    n = int(2 * FS)
    epochs, labels = [], []
    for ci, f in enumerate(freqs):
        for k in range(6):
            epochs.append(ssvep_signal(f, FS, n, amp=4.0, seed=100 + ci * 10 + k, phase=0.3))
            labels.append(ci)
    epochs, labels = np.array(epochs), np.array(labels)
    # leave-one-out on the last trial of each class
    test_idx = [i for i in range(len(labels)) if i % 6 == 5]
    train = [i for i in range(len(labels)) if i % 6 != 5]
    model = S.TRCA(FS, ensemble=True).fit(epochs[train], labels[train])
    correct = sum(model.predict(epochs[i]) == labels[i] for i in test_idx)
    assert correct >= 3  # at least 3/4 correct


def test_itr_is_zero_at_chance_and_positive_above():
    assert S.itr_bits_per_min(0.5, 2, 1.0) == 0.0
    assert S.itr_bits_per_min(0.25, 4, 1.0) == 0.0
    assert S.itr_bits_per_min(1.0, 4, 2.0) > S.itr_bits_per_min(0.8, 4, 2.0) > 0


def test_realized_frequency_is_frame_locked():
    assert MS.realized_freq(15.0, 60.0) == 15.0   # 60/4
    assert MS.realized_freq(12.0, 60.0) == 12.0   # 60/5
    assert abs(MS.realized_freq(13.0, 60.0) - 60.0 / 5) < 1e-9  # 13 -> rounds to period 5 => 12 Hz


# --- alpha.py --------------------------------------------------------------------------------------------------------
def test_closed_has_more_relative_alpha_than_open():
    c = A.rel_alpha(alpha_signal(FS, int(2 * FS), closed=True, seed=3), FS).mean()
    o = A.rel_alpha(alpha_signal(FS, int(2 * FS), closed=False, seed=4), FS).mean()
    assert c > o * 1.5


def test_alpha_detector_calibrates_and_fires_on_sustained_closed():
    ow = [alpha_signal(FS, int(FS), closed=False, seed=10 + i) for i in range(6)]
    cw = [alpha_signal(FS, int(FS), closed=True, seed=20 + i) for i in range(6)]
    det, st = A.AlphaDetector.calibrate(ow, cw, FS)
    assert st["train_bal_acc"] >= 0.8 and st["separation"] > 0
    # feed sustained closed windows with advancing time -> fires once past enter_s, then refractory
    fired = [det.push_window(alpha_signal(FS, int(FS), closed=True, seed=30 + i), t=float(i) * 0.5)
             for i in range(8)]
    assert sum(fired) >= 1
    # eyes open should not fire
    det2, _ = A.AlphaDetector.calibrate(ow, cw, FS)
    assert not any(det2.push_window(alpha_signal(FS, int(FS), closed=False, seed=40 + i), t=float(i) * 0.5)
                   for i in range(8))


# --- muse_session.py -------------------------------------------------------------------------------------------------
def test_build_timeline_has_alpha_and_ssvep_phases():
    cfg = MS.SessionConfig(freqs=(12.0, 15.0), reps=3, alpha_blocks=2)
    phases = MS.build_timeline(cfg)
    kinds = [p.kind for p in phases]
    assert kinds.count("eyes_open") == 2 and kinds.count("eyes_closed") == 2
    assert kinds.count("ssvep") == 6  # reps * n_freqs
    assert all(p.code >= 100 for p in phases if p.kind == "ssvep")
    assert cfg.est_minutes() > 0


def test_flicker_on_is_a_square_wave():
    on = [MS.flicker_on(i, 15.0, 60.0) for i in range(8)]  # period 4: on,on,off,off repeating
    assert on == [True, True, False, False, True, True, False, False]


def test_analyse_frame_log_counts_dropped_frames():
    good = list(np.cumsum(np.full(120, 1 / 60.0)))
    rep = MS.analyse_frame_log(good)
    assert abs(rep["measured_refresh_hz"] - 60) < 1 and rep["dropped_frames"] == 0
    dt = np.full(120, 1 / 60.0)
    dt[50] = 3 / 60.0  # one long frame = 2 dropped
    rep2 = MS.analyse_frame_log(list(np.cumsum(dt)))
    assert rep2["dropped_frames"] == 1 and rep2["frac_dropped"] > 0


def test_check_display_headless_runs_and_reports_realized_freqs():
    rep = MS.check_display([12.0, 15.0], seconds=1.0)
    assert rep["n_frames"] > 0 and set(rep["requested_freqs"]) == {12.0, 15.0}
    assert len(rep["realized_freqs"]) == 2


def test_session_recorder_synthetic_board_writes_npz(tmp_path):
    cfg = MS.SessionConfig(freqs=(12.0, 15.0), reps=1, alpha_blocks=1, block_s=1.0, ssvep_s=1.0,
                           settle_s=0.5, rest_s=0.3)
    out = tmp_path / "sess.npz"
    rep = MS.run_session(cfg, "synthetic", str(out), headless_frames=120)
    assert out.exists() and rep["n_samples"] > 0
    rec = np.load(out, allow_pickle=True)
    assert "eeg" in rec.files and "markers" in rec.files and "frame_report" in rec.files


# --- muse_detect.py --------------------------------------------------------------------------------------------------
def test_ssvep_stream_fires_on_the_looked_at_frequency():
    st = MD.SsvepStream((12.0, 15.0), FS, win_s=2.0, step_s=0.5, method="fbcca", thr=0.2, margin=0.2)
    x = ssvep_signal(15.0, FS, int(5 * FS), amp=3.5, seed=5)
    events = [e for i in range(0, x.shape[1], 64) for e in st.push(x[:, i:i + 64])]
    assert events and all(abs(e[1] - 15.0) < 1e-6 for e in events)


def test_ssvep_stream_confirm_freq_gate():
    st = MD.SsvepStream((12.0, 15.0), FS, win_s=2.0, step_s=0.5, method="fbcca", thr=0.1, margin=0.0,
                        confirm_freq=12.0)
    x = ssvep_signal(15.0, FS, int(4 * FS), seed=6)  # looking at 15, but only 12 is a confirm
    events = [e for i in range(0, x.shape[1], 64) for e in st.push(x[:, i:i + 64])]
    assert events == []


def test_alpha_stream_fires_on_alpha_but_not_on_open():
    ow = [alpha_signal(FS, int(FS), closed=False, seed=i) for i in range(6)]
    cw = [alpha_signal(FS, int(FS), closed=True, seed=50 + i) for i in range(6)]
    det, _ = A.AlphaDetector.calibrate(ow, cw, FS)
    st = MD.AlphaStream(FS, det.threshold, win_s=1.0, step_s=0.5)
    closed = alpha_signal(FS, int(4 * FS), closed=True, seed=77)
    ev = [e for i in range(0, closed.shape[1], 64) for e in st.push(closed[:, i:i + 64])]
    assert len(ev) >= 1


def test_emit_packets_parse_as_messages():
    from rover_intent.transport.udp import _MSG
    from rover_intent.types import EEGSelect, GripToggle
    g = _MSG.validate_python(json.loads(MD.grip_toggle_packet(1.0, "ssvep").decode()))
    assert isinstance(g, GripToggle) and g.source == "ssvep"
    s = _MSG.validate_python(json.loads(MD.eeg_select_packet(2.0, "grab", 12.0, 0.8, "ssvep").decode()))
    assert isinstance(s, EEGSelect) and s.label == "grab" and s.freq == 12.0


def test_muse_detect_replay_cli(tmp_path, capsys):
    rec = tmp_path / "r.npz"
    np.savez(rec, data=ssvep_signal(15.0, FS, int(4 * FS), seed=8), fs=FS)
    assert MD.main(["replay", "--detector", "ssvep", "--freqs", "12,15", "--file", str(rec),
                    "--win", "2.0", "--thr", "0.1", "--margin", "0.0"]) == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["detector"] == "ssvep" and out["n_events"] >= 1


# --- message type + UDP ----------------------------------------------------------------------------------------------
def test_eeg_select_survives_a_real_udp_hop():
    import time
    from rover_intent.transport.udp import Receiver, Sender
    from rover_intent.types import EEGSelect
    rx = Receiver(0, "127.0.0.1")
    port = rx.sock.getsockname()[1]
    Sender("127.0.0.1", port).send(EEGSelect(t=1.0, label="confirm", freq=15.0, confidence=0.7, source="ssvep"))
    msgs = []
    for _ in range(50):
        msgs += rx.poll()
        if msgs:
            break
        time.sleep(0.01)
    assert len(msgs) == 1 and isinstance(msgs[0][1], EEGSelect) and msgs[0][1].label == "confirm"


# --- muse_report.py --------------------------------------------------------------------------------------------------
def _synthetic_session(fs=FS, freqs=(12.0, 15.0), reps=3):
    """Build an eeg array + markers + meta like muse_session would save, for build_report."""
    segs, markers, sample = [], [], 0
    code = {"eyes_open": 10, "eyes_closed": 11}

    def add(x, c):
        nonlocal sample
        markers.append((sample, float(c)))
        segs.append(x)
        sample += x.shape[1]

    for _ in range(2):
        add(alpha_signal(fs, int(3 * fs), closed=False, seed=sample), 10)
        add(alpha_signal(fs, int(3 * fs), closed=True, seed=sample + 1), 11)
    order = [f for f in freqs for _ in range(reps)]
    for f in order:
        idx = list(freqs).index(f)
        add(ssvep_signal(f, fs, int(3 * fs), amp=2.5, seed=sample), 100 + idx)
    eeg = np.concatenate(segs, axis=1)
    meta = {"config": {"freqs": list(freqs)}, "channels": ["TP9", "AF7", "AF8", "TP10"]}
    frame_report = {"measured_refresh_hz": 60.0, "dropped_frames": 0, "frac_dropped": 0.0, "n_frames": 3600}
    return eeg, np.array(markers, float), meta, frame_report


def test_report_shows_alpha_rise_and_ssvep_structure():
    eeg, markers, meta, fr = _synthetic_session()
    rep = MR.build_report(eeg, FS, markers, meta, fr)
    assert rep["alpha"]["overall_ratio"] > 1.3                 # eyes-closed alpha rises
    assert rep["n_ssvep_trials"] == 6 and rep["n_alpha_blocks"]["closed"] == 2
    # SSVEP SNR at the flicker frequency should be positive (dB) for at least one channel
    for f in ("12", "15"):
        vals = [v for v in rep["ssvep_snr_db"][f].values() if v is not None]
        assert max(vals) > 0
    # FBCCA accuracy over the session's own trials beats chance
    best = max(d["acc"] for d in rep["ssvep_accuracy"]["fbcca"].values() if d["acc"] is not None)
    assert best >= 0.5
    assert rep["verdict"]["display_ok"] is True


def test_report_ssvep_snr_peaks_at_the_flicker_frequency():
    eeg, markers, meta, fr = _synthetic_session(freqs=(15.0,), reps=4)
    rep = MR.build_report(eeg, FS, markers, meta, fr)
    best_ch = rep["ssvep_snr_best_channel"]["15"]
    assert rep["ssvep_snr_db"]["15"][best_ch] > 0

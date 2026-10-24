"""eeg/sunday.py: the guided day-one Muse session. No hardware. The clench blocks reuse the Muse-like synthetic
EMG generator from test_clench; the alpha blocks reuse the alpha generator from test_muse; the end-to-end path uses
BrainFlow's synthetic board. Covers the protocol, resumable storage, the calibration/fit on labelled synthetic data,
the config write + backup, and the live-test wire format."""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rover_intent.eeg import sunday as SD
from test_clench import FS, double, simulate
from test_muse import alpha_signal


# --- protocol --------------------------------------------------------------------------------------------------------
def test_protocol_has_the_blocks_the_day_one_budget_needs():
    cfg = SD.SundayConfig()
    blocks = SD.build_protocol(cfg)
    labels = {b.name: b.label for b in blocks}
    # the two realistic fine-tune targets + the false-trigger negatives (the most important data)
    assert labels["rest"] == "rest"
    assert labels["clench_single"] == "clench" and labels["clench_double"] == "clench"
    assert [labels[n] for n in ("neg_chew", "neg_blink", "neg_head")] == ["negative"] * 3
    assert labels["eyes_open"] == "eyes_open" and labels["eyes_closed"] == "eyes_closed"
    assert cfg.clench_double_reps >= 6                       # enough doubles to hold some out
    assert 8 <= cfg.est_minutes() <= 30                     # lean, well within the ~25-30 min day-one budget
    assert set(SD.STAGES) == set(SD.RECORDING_BLOCKS) | {"train", "live"}


# --- resumable storage -----------------------------------------------------------------------------------------------
def test_save_block_and_manifest_roundtrip_and_resume(tmp_path):
    cfg = SD.SundayConfig(name="t")
    sd = tmp_path / "sess"
    manifest = SD.load_manifest(sd)
    block = SD.build_protocol(cfg)[0]
    eeg = simulate(3, seed=1)
    SD.save_block(sd, block, eeg, FS, ["TP9", "AF7", "AF8", "TP10"], {"ok": True}, "muse2", manifest)
    assert (sd / "raw" / "rest.npz").exists() and (sd / "session.json").exists()
    x, fs, chans = SD.load_block(sd, "rest")
    assert x.shape == eeg.shape and fs == FS and chans[0] == "TP9"
    # a fresh manifest load sees the block as done -> resume would skip it
    assert "rest" in SD.done_blocks(SD.load_manifest(sd))


def test_dated_session_dir():
    p = SD.session_dir_for("pranav", date="20261025")
    assert p.name == "pranav_20261025" and "eeg_sessions" in str(p)


# --- signal quality --------------------------------------------------------------------------------------------------
def test_channel_quality_flags_flat_and_mains_but_passes_a_clean_signal():
    cfg = SD.SundayConfig()
    rng = np.random.default_rng(0)
    n = int(6 * FS)
    clean = rng.normal(0, 15.0, (4, n))                     # 15 uV broadband, no dominant line
    q = SD.channel_quality(clean, FS, cfg)
    assert q["ok"] and all(d["status"] == "good" for d in q["channels"].values())
    bad = clean.copy()
    bad[0] *= 1e-3                                           # flat electrode (not on skin)
    t = np.arange(n) / FS
    bad[1] = 100 * np.sin(2 * np.pi * cfg.line_hz * t)      # swamped by mains (rms < drift cap, line power huge)
    q2 = SD.channel_quality(bad, FS, cfg)
    assert not q2["ok"] and "TP9" in q2["bad"] and "AF7" in q2["bad"]
    assert q2["channels"]["TP9"]["status"] == "flat" and q2["channels"]["AF7"]["status"] == "mains"
    assert q2["hints"]                                      # fix-it hints present


# --- clench fit on labelled synthetic data ---------------------------------------------------------------------------
def _clench_blocks(reps=8, seed=0):
    """Make rest / single / double / negative recordings the way the session would, from the Muse-like generator."""
    rest = simulate(16, blinks=np.arange(1.5, 16, 3.1), seed=seed + 1)
    single = np.concatenate([simulate(3, clenches=[(0.8, 0.5)], seed=seed + 10 + i) for i in range(reps)], axis=1)
    dbl = np.concatenate([simulate(4, clenches=double(0.8), seed=seed + 50 + i) for i in range(reps)], axis=1)
    chew = [(0.6 * i, 0.3, 55) for i in range(10)]
    neg = np.concatenate([simulate(10, talk=[(0.5, 9)], clenches=chew, seed=seed + 90),
                          simulate(8, blinks=np.arange(0.4, 8, 0.6), seed=seed + 91)], axis=1)
    return rest, single, dbl, neg


def test_fit_clench_detects_held_out_doubles_and_rejects_negatives():
    rest, single, dbl, neg = _clench_blocks(reps=8)
    profile, m = SD.fit_clench(rest, single, dbl, neg, FS, expected_doubles=8)
    assert profile["stats"]["quality"] in ("good", "marginal")
    assert m["calibrated"]["detection_rate"] >= 0.75             # catches most held-out double-clenches
    assert m["calibrated"]["false_per_min_negatives"] <= 2.0     # rarely fires on talk/chew/blink
    assert "synthetic_default" in m and m["deployed_on_ratio"] > 1.0
    json.dumps(profile)                                          # saveable


# --- alpha fit -------------------------------------------------------------------------------------------------------
def test_fit_alpha_beats_chance_on_held_out_windows():
    eo = np.concatenate([alpha_signal(FS, int(6 * FS), closed=False, seed=i) for i in range(4)], axis=1)
    ec = np.concatenate([alpha_signal(FS, int(6 * FS), closed=True, seed=20 + i) for i in range(4)], axis=1)
    thr, m = SD.fit_alpha(eo, ec, FS, win_s=2.0)
    assert m["ok"] and thr > 0
    assert m["heldout_bal_acc"] >= 0.7 and m["closed_over_open_ratio"] > 1.3


# --- config write + backup -------------------------------------------------------------------------------------------
def test_write_profiles_writes_both_and_backs_up_existing(tmp_path):
    rest, single, dbl, neg = _clench_blocks(reps=6)
    profile, _ = SD.fit_clench(rest, single, dbl, neg, FS, expected_doubles=6)
    pdir = tmp_path / "profiles"
    p1 = SD.write_profiles(profile, 0.42, {"ok": True}, "pranav", profiles_dir=str(pdir))
    assert (pdir / "clench_pranav.json").exists() and (pdir / "muse_pranav.json").exists()
    assert p1["clench_backup"] is None
    muse = json.loads((pdir / "muse_pranav.json").read_text())
    assert muse["alpha_threshold"] == 0.42
    # a second write backs up the old files rather than clobbering them
    p2 = SD.write_profiles(profile, 0.5, {"ok": True}, "pranav", profiles_dir=str(pdir))
    assert p2["clench_backup"] and p2["muse_backup"]
    assert list(pdir.glob("clench_pranav.json.bak.*")) and list(pdir.glob("muse_pranav.json.bak.*"))


# --- live test (step f): double clench -> GripToggle, SIM only -------------------------------------------------------
def test_live_test_dry_counts_toggles_from_replay():
    rest, single, dbl, neg = _clench_blocks(reps=8)
    profile, _ = SD.fit_clench(rest, single, dbl, neg, FS, expected_doubles=8)
    x = simulate(9, clenches=double(1.0) + double(5.0), seed=123)
    res = SD.live_test(profile, "127.0.0.1", SD.SPARE_SIM_PORT, replay_x=x, dry=True)
    assert res["toggles"] == 2 and res["port"] != 47100


def test_live_test_refuses_the_real_control_port():
    with pytest.raises(AssertionError):
        SD.live_test({}, "127.0.0.1", 47100, replay_x=np.zeros((4, 10)), dry=True)


def test_live_test_sends_parseable_grip_toggles_over_udp():
    import time

    from rover_intent.transport.udp import Receiver
    from rover_intent.types import GripToggle, Utterance
    rest, single, dbl, neg = _clench_blocks(reps=8)
    profile, _ = SD.fit_clench(rest, single, dbl, neg, FS, expected_doubles=8)
    rx = Receiver(0, "127.0.0.1")                       # ephemeral port, never the running app on 47100
    port = rx.sock.getsockname()[1]
    x = simulate(5, clenches=double(1.0), seed=7)
    SD.live_test(profile, "127.0.0.1", port, replay_x=x, dry=False, send_follow=True)
    msgs = []
    for _ in range(50):
        msgs += rx.poll()
        time.sleep(0.005)
    kinds = [type(m).__name__ for _, m, _ in msgs]
    assert "Utterance" in kinds and "GripToggle" in kinds   # follow-me + at least one toggle
    tog = [m for _, m, _ in msgs if isinstance(m, GripToggle)]
    assert tog and tog[0].source == "clench"


# --- end to end on the synthetic board -------------------------------------------------------------------------------
def test_run_session_synthetic_end_to_end_and_resume(tmp_path):
    cfg = SD.SundayConfig(name="e2e", rest_s=3, clench_single_reps=2, clench_single_s=1.0,
                          clench_double_reps=4, clench_double_s=1.0, neg_chew_s=3, neg_blink_s=2, neg_head_s=2,
                          alpha_block_s=8, q_check_s=2)
    cfg.serial = cfg.mac = cfg.name_target = ""
    sd = tmp_path / "e2e"
    res = SD.run_session(cfg, "synthetic", session_dir=sd, quiet=True, strict_quality=False)
    assert res["ok"]
    assert (sd / "report.json").exists() and set(res["done_blocks"]) == set(SD.RECORDING_BLOCKS)
    rep = json.loads((sd / "report.json").read_text())
    assert "clench" in rep and "alpha" in rep and "profiles" in rep
    # resume: with every block done, a second run records nothing new and still produces a report
    res2 = SD.run_session(cfg, "synthetic", session_dir=sd, quiet=True, strict_quality=False, resume=True)
    assert res2["ok"] and set(res2["done_blocks"]) == set(SD.RECORDING_BLOCKS)


def test_only_single_block_records_just_that_block(tmp_path):
    cfg = SD.SundayConfig(name="one", clench_double_reps=3, clench_double_s=1.0, q_check_s=1)
    cfg.serial = cfg.mac = cfg.name_target = ""
    sd = tmp_path / "one"
    res = SD.run_session(cfg, "synthetic", session_dir=sd, only="clench_double", quiet=True, strict_quality=False)
    assert res["ok"] and res["done_blocks"] == ["clench_double"]
    assert not (sd / "report.json").exists()                # training not run for a single-block redo


def test_preflight_passes_on_synthetic():
    out = SD.preflight()
    assert out["ok"] and out["seconds"] < 180
    assert out["checks"]["full dry run (synthetic board, end to end)"]["ok"]

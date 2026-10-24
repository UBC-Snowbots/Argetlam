"""Live tuning for PLANE body teleop (body/plane.py + app wiring + fast_path phrases).

Tomorrow's first live drive tunes the signs/gains/thresholds that were set from recordings, so every knob has a
bounded, clamped step, a sign flip must not jump, the tuning survives save/load/reset, the guided reach-threshold
check fits forward/back fracs from measured poses, and the voice phrases reach the mapper without stealing a normal
jog. The synthetic skeleton + app harness are reused from tests/test_plane.py.
"""
import json
import socket
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from rover_intent.body.plane import (DB_HI, DB_LO, FRAC_HI, FRAC_LO, FRAC_MIN_GAP, LAT_HI, LAT_LO, PlaneTuneCheck,
                                      RR_MAX_HI, RR_MAX_LO, VRT_HI, VRT_LO, fit_reach_thresholds, plane_coords)
from rover_intent.planner.fast_path import parse_fast

from test_plane import _App, _mapper, _pose, _settle


# --- bounded, clamped knob steps -----------------------------------------------

def test_reach_speed_clamps_both_ends():
    m, _ = _mapper()
    for _ in range(40):
        m.adjust("reach_faster")
    assert m.rr_max == pytest.approx(RR_MAX_HI)
    for _ in range(40):
        m.adjust("reach_slower")
    assert m.rr_max == pytest.approx(RR_MAX_LO)


def test_sensitivity_clamps_both_ends():
    m, _ = _mapper()
    for _ in range(40):
        m.adjust("more_sensitive")
    assert abs(m.gain[1]) == pytest.approx(LAT_HI) and abs(m.gain[2]) == pytest.approx(VRT_HI)
    for _ in range(40):
        m.adjust("less_sensitive")
    assert abs(m.gain[1]) == pytest.approx(LAT_LO) and abs(m.gain[2]) == pytest.approx(VRT_LO)


def test_deadzone_clamps_both_ends():
    m, _ = _mapper()
    for _ in range(40):
        m.adjust("bigger_deadzone")
    assert m.deadband == pytest.approx(DB_HI)
    for _ in range(40):
        m.adjust("smaller_deadzone")
    assert m.deadband == pytest.approx(DB_LO)


def test_reach_fracs_stay_in_bounds_and_ordered():
    m, _ = _mapper()
    for _ in range(40):                       # forward threshold pushed to one extreme
        m.adjust("reach_fwd_sooner")
    for _ in range(40):                       # back threshold pushed to the other
        m.adjust("reach_back_sooner")
    assert FRAC_LO <= m.rr_fwd_frac <= FRAC_HI and FRAC_LO <= m.rr_back_frac <= FRAC_HI
    assert m.rr_back_frac - m.rr_fwd_frac >= FRAC_MIN_GAP - 1e-9   # a non-zero dead zone survives
    # one step from the default moves the thresholds the intended way
    m2, _ = _mapper()
    f0, b0 = m2.rr_fwd_frac, m2.rr_back_frac
    m2.adjust("reach_fwd_sooner")
    assert m2.rr_fwd_frac > f0                 # forward begins sooner = lower (larger frac below shoulders)
    m3, _ = _mapper()
    m3.adjust("reach_back_sooner")
    assert m3.rr_back_frac < b0                 # back begins sooner = higher (smaller frac below shoulders)


# --- sign flips must not jump --------------------------------------------------

def test_flip_lateral_no_jump_then_reverses():
    m, tcp0 = _mapper()
    p = _settle(m, tcp0, [dict(rwx=0.25)] * 80)   # hand to the right: lateral driven away from tcp0
    before = p.copy()
    m.adjust("flip_lateral")
    after = m(_pose(0.033 * 160, rwx=0.25))        # same pose right after the flip
    assert np.max(np.abs(after - before)) < 2e-3   # continuous: no jump
    further = m(_pose(0.033 * 161, rwx=0.15))       # hand further right now drives lateral the OTHER way
    assert further[1] > after[1]


def test_flip_reach_no_jump_then_reverses():
    m, tcp0 = _mapper()
    _settle(m, tcp0, [dict(lwy=0.30)] * 120)        # driving reach forward
    x_before = m(_pose(0.033 * 200, lwy=0.30))[0]
    m.adjust("flip_reach")
    x_after = m(_pose(0.033 * 201, lwy=0.30))[0]
    assert abs(x_after - x_before) < 0.01           # one integration step at most, not a jump
    x_end = None
    for i in range(60):
        x_end = m(_pose(0.033 * (202 + i), lwy=0.30))[0]   # hand still up, but reach is now flipped -> goes back
    assert x_end < x_before - 0.02 and m.reach_rate < 0


# --- save / load / reset round trip --------------------------------------------

def test_save_load_reset_round_trip():
    m, _ = _mapper()
    for op in ("reach_faster", "reach_faster", "more_sensitive", "flip_vertical", "reach_back_sooner",
               "bigger_deadzone"):
        m.adjust(op)
    snap = m.snapshot()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "plane_tune_dev_arm_real.json"
        m.save(path)
        assert path.exists() and json.loads(path.read_text())["reach_max_speed"] > 0
        m2, _ = _mapper()
        assert m2.snapshot() != snap               # a fresh mapper is at the profile defaults
        assert m2.load(path) is True
        assert m2.snapshot() == snap               # load restores every knob
    m2.reset_tuning()
    base, _ = _mapper()
    assert m2.snapshot() == base.snapshot()        # reset returns to the profile baseline


def test_load_missing_file_is_false():
    m, _ = _mapper()
    assert m.load(Path(tempfile.gettempdir()) / "definitely_missing_plane_tune.json") is False


def test_load_clamps_out_of_range_values():
    m, _ = _mapper()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad.json"
        path.write_text(json.dumps({"reach_max_speed": 9.0, "lateral": 99.0, "vertical": -99.0,
                                    "deadband": 9.0, "forward_frac": 0.9, "back_frac": -0.9}))
        assert m.load(path) is True
    assert m.rr_max == pytest.approx(RR_MAX_HI)
    assert abs(m.gain[1]) == pytest.approx(LAT_HI) and abs(m.gain[2]) == pytest.approx(VRT_HI)
    assert m.gain[2] < 0                            # the stored sign is kept
    assert DB_LO <= m.deadband <= DB_HI
    assert FRAC_LO <= m.rr_fwd_frac < m.rr_back_frac <= FRAC_HI


# --- threshold fit on synthetic poses ------------------------------------------

def test_fit_reach_thresholds_from_poses():
    sh = plane_coords(_pose(0.0, lwy=0.50))         # left hand at shoulder height
    wa = plane_coords(_pose(0.0, lwy=0.80))         # left hand lowered toward the waist
    fit = fit_reach_thresholds({"shoulder": sh["reach_up"], "waist": wa["reach_up"],
                                "span": sh["hip_drop"], "right_x": 0.9})
    assert FRAC_LO <= fit["forward_frac"] < fit["back_frac"] <= FRAC_HI
    assert fit["measured"]["right_x"] == pytest.approx(0.9)
    # a lower waist sample widens the back threshold (more room before "back")
    wa2 = plane_coords(_pose(0.0, lwy=0.95))
    fit2 = fit_reach_thresholds({"shoulder": sh["reach_up"], "waist": wa2["reach_up"], "span": sh["hip_drop"]})
    assert fit2["back_frac"] >= fit["back_frac"]


def test_fit_reach_thresholds_degenerate_span():
    fit = fit_reach_thresholds({"shoulder": 0.0, "waist": 0.0, "span": 0.0})
    assert FRAC_LO <= fit["forward_frac"] < fit["back_frac"] <= FRAC_HI   # never a zero / inverted dead zone


def test_tune_check_flow_produces_a_fit():
    tc = PlaneTuneCheck()
    tc.start(0.0)
    assert tc.active and tc.pop_prompt() and tc.pop_prompt() is None     # the first prompt is served once

    def pose_for(t):
        # match the four stage holds by time: ready (neutral), left to shoulder, left to waist, right hand out
        if t < 2.5:
            return dict(lwy=0.62)
        if t < 7.0:
            return dict(lwy=0.50)       # left hand at shoulder height -> FORWARD reference
        if t < 11.5:
            return dict(lwy=0.85)       # left hand low -> BACK reference
        return dict(rwx=0.20)           # right hand out to the side

    prompts, t = 0, 0.0
    while tc.active and t < 20.0:
        t += 0.033
        tc.update(t, plane_coords(_pose(t, **pose_for(t))))
        if tc.pop_prompt():
            prompts += 1
    assert tc.finished and tc.result is not None
    r = tc.result
    assert FRAC_LO <= r["forward_frac"] < r["back_frac"] <= FRAC_HI
    assert prompts >= 3                              # prompted through shoulder / waist / right (and done)


# --- voice phrases via the fast path (in-tick) ---------------------------------

@pytest.mark.parametrize("text,op", [
    ("faster reach", "reach_faster"), ("reach slower", "reach_slower"),
    ("more sensitive", "more_sensitive"), ("less sensitive", "less_sensitive"),
    ("reach forward sooner", "reach_fwd_sooner"), ("reach back sooner", "reach_back_sooner"),
    ("bigger dead zone", "bigger_deadzone"), ("smaller dead zone", "smaller_deadzone"),
    ("flip left right", "flip_lateral"), ("flip up down", "flip_vertical"), ("flip reach", "flip_reach"),
    ("save tuning", "save"), ("reset tuning", "reset"), ("tune check", "tune_check"),
    ("save the tuning", "save"), ("check tuning", "tune_check")])
def test_fast_path_plane_tune_phrases(text, op):
    r = parse_fast(text)
    assert r is not None and len(r["steps"]) == 1
    assert r["steps"][0]["kind"] == "plane_tune" and r["steps"][0]["op"] == op


@pytest.mark.parametrize("text", ["move left 5 cm", "move right", "go forward 10 centimetres",
                                  "turn the base left 30 degrees", "up a bit", "slower", "faster", "go to base"])
def test_fast_path_does_not_steal_normal_commands(text):
    r = parse_fast(text, objects=[], presets=["base", "ready"])
    if r is not None:
        assert all(s["kind"] != "plane_tune" for s in r["steps"]), f"{text!r} wrongly parsed as plane tuning"


# --- app wiring ----------------------------------------------------------------

def _read_reply(a, want_text):
    end = time.time() + 3.0
    while time.time() < end:
        try:
            m = json.loads(a.s.recvfrom(65536)[0])
        except socket.timeout:
            break
        if m.get("kind") == "reply" and m.get("text") == want_text:
            return m
    return None


def test_app_plane_tune_message_and_voice():
    a = _App(47268, seconds=8.0)
    assert a.query()["mapping"] == "PlaneMapper"
    # typed PlaneTune message: reach_faster -> ok, snapshot carried back, speed rose above the 0.04 default
    a.send({"kind": "plane_tune", "t": time.time(), "op": "reach_faster"})
    rep = _read_reply(a, "plane_tune")
    assert rep is not None and rep["ok"] and rep["plane_tune"]["max_speed"] > 0.04
    # spoken phrase: "flip left right" reaches the mapper and replies the flip confirmation
    a.send({"kind": "utterance", "t": time.time(), "text": "flip left right"})
    rep2 = _read_reply(a, "flip left right")
    assert rep2 is not None and "left and right" in (rep2.get("say") or "").lower()
    # the knob snapshot now appears in the 10 Hz status (what the HUD / body client draw)
    a.send_pose(); time.sleep(0.05)
    st = a.query()
    assert st.get("reach_state") in (">>", "<<", "hold")
    a.join()

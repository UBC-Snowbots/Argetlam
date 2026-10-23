"""Voice -> named poses with a spoken confirm.

Covers: the keyword fallback parser mapping pose words onto preset Intents (profile-gated), the Nemotron
`normalize_preset` post-processing, the `is_confirm` vocabulary, and the full app confirm flow over UDP on the
mock backend: accept, cancel, timeout, stop-bypass, that voice "home" drives the sim RETURN preset and NEVER the
hardware home button, and the confirm.skills flag (off for dev_arm so the sim gate still scripts skills, on for
dev_arm_real).
"""
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from rover_intent.config import load_config
from rover_intent.control.presets import RETURN
from rover_intent.planner.nemotron import fallback_parse, is_confirm, normalize_preset
from rover_intent.types import Action, Intent

ROOT = Path(__file__).resolve().parents[1]
BASE = [0.104, -1.415, -0.997, -0.229, -0.482, -3.142]   # dev_arm "base" (stow) preset


# --- parser: pose words -> preset Intent -------------------------------------------------------------------------
@pytest.mark.parametrize("text,name", [
    ("stow", "base"), ("stow the arm please", "base"), ("rest", "base"), ("go to base", "base"),
    ("ready", "ready"), ("ready position", "ready"),
    ("go back", RETURN), ("return", RETURN), ("back to start", RETURN), ("home", RETURN), ("go home", RETURN),
])
def test_fallback_maps_pose_words_to_presets(text, name):
    i = fallback_parse(text, ["cup", "bottle"], ["ready", "base"])
    assert i.action == Action.preset and i.preset == name


def test_voice_home_is_sim_return_never_hardware():
    i = fallback_parse("home", [], ["ready", "base"])
    assert i.action == Action.preset and i.preset == RETURN     # not Action.home, not "home_hw"
    assert i.preset != "home_hw"


def test_named_preset_gated_by_profile_but_return_always_ok():
    # a profile without "ready" must not emit a ready preset from the keyword parser...
    assert fallback_parse("ready", [], ["base"]).action == Action.unknown
    # ...but RETURN is always valid even with no presets listed, and the 2-arg call (no list) still works
    assert fallback_parse("go back", []).preset == RETURN
    assert fallback_parse("stow", []).preset == "base"          # unknown-to-this-profile names are refused downstream


def test_normalize_preset_folds_home_and_aliases():
    assert normalize_preset(Intent(action=Action.home)).action == Action.preset
    assert normalize_preset(Intent(action=Action.home)).preset == RETURN
    assert normalize_preset(Intent(action=Action.preset, preset="Stow")).preset == "base"
    assert normalize_preset(Intent(action=Action.preset, preset="ready")).preset == "ready"


@pytest.mark.parametrize("text,ok", [
    ("yes", True), ("yeah do it", True), ("confirm", True), ("okay", True), ("ok", True), ("sure", True),
    ("go ahead", True), ("affirmative", True),
    ("no", False), ("cancel", False), ("stop", False), ("go back", False), ("go home", False), ("banana", False),
])
def test_is_confirm_vocabulary(text, ok):
    assert is_confirm(text) is ok


# --- app confirm flow over UDP (mock backend) --------------------------------------------------------------------
def _app(cfg, port, seconds):
    cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
    cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"   # force the offline keyword parser, no network
    from rover_intent import app
    th = threading.Thread(target=app.run, args=(cfg, "mock", seconds, False), daemon=True)
    th.start()
    time.sleep(0.4)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.3)
    return app, th, s


def _drain(s):
    s.settimeout(0.05)
    try:
        while True:
            s.recvfrom(65536)
    except (socket.timeout, BlockingIOError):
        pass
    s.settimeout(1.5)


def _recv(s, kind, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            m = json.loads(s.recvfrom(65536)[0])
        except (socket.timeout, BlockingIOError):
            continue
        if m.get("kind") == kind:
            return m
    return None


def _say(s, port, text, offline=True):
    _drain(s)
    s.sendto(json.dumps({"kind": "utterance", "t": time.time(), "text": text, "offline": offline}).encode(),
             ("127.0.0.1", port))
    return _recv(s, "reply")


def _query(s, port):
    _drain(s)
    s.sendto(json.dumps({"kind": "query", "t": time.time()}).encode(), ("127.0.0.1", port))
    return _recv(s, "state")


def test_confirm_accept_cancel_stop_and_home(monkeypatch):
    """One app instance: a yes moves to the pose; a non-yes or a stop cancels; voice home drives RETURN and never
    calls the hardware home."""
    app, th, s = _app(load_config(arm="dev_arm"), 47211, 12.0)
    calls = []
    monkeypatch.setattr(app, "request_hw_home", lambda *a, **k: (calls.append(a), (False, "patched"))[1])
    try:
        # 1) accept: "stow" asks first (no move), "yes" executes via the preset path and drives the arm to base
        r = _say(s, 47211, "stow")
        assert r["confirm"] is True and "confirm" in r["say"].lower()
        assert _query(s, 47211)["preset"] is None               # nothing moving yet
        r = _say(s, 47211, "yes")
        assert r["ok"] is True and "base" in r["say"].lower()
        reached = False
        for _ in range(40):
            st = _query(s, 47211)
            if st and max(abs(a - b) for a, b in zip(st["q_meas"], BASE)) < 0.02:
                reached = True
                break
            time.sleep(0.1)
        assert reached, "the confirmed preset never drove the arm to the base pose"
        _say(s, 47211, "reset")

        # 2) cancel: a non-confirm utterance cancels the armed confirm; nothing moves
        r = _say(s, 47211, "ready")
        assert r["confirm"] is True
        _say(s, 47211, "banana")
        time.sleep(0.2)
        assert _query(s, 47211)["preset"] is None
        assert _say(s, 47211, "yes") is not None                 # a lone yes now is just an unrecognised command
        assert _query(s, 47211)["preset"] is None

        # 3) stop bypasses the confirm instantly
        r = _say(s, 47211, "stow")
        assert r["confirm"] is True
        r = _say(s, 47211, "stop")
        assert r["mode"] == "hold"
        assert _query(s, 47211)["preset"] is None

        # 4) voice "home" = sim RETURN, confirmed; the hardware home function is never called
        r = _say(s, 47211, "home")
        assert r["confirm"] is True and "start pose" in r["say"].lower()
        r = _say(s, 47211, "confirm")
        assert r["ok"] is True and RETURN in r["say"].lower()
        assert _query(s, 47211)["preset"] in (RETURN, None)       # returning to / already at the start pose
        assert calls == [], "voice home must never call request_hw_home"
    finally:
        s.close()
        th.join(timeout=14)
    assert not th.is_alive()


def test_confirm_times_out():
    cfg = load_config(arm="dev_arm")
    cfg.setdefault("confirm", {})["window_s"] = 0.5
    app, th, s = _app(cfg, 47212, 6.0)
    try:
        r = _say(s, 47212, "stow")
        assert r["confirm"] is True
        time.sleep(1.0)                      # let the window lapse
        _drain(s)
        _say(s, 47212, "yes")                # too late: no pending confirm, so nothing moves
        assert _query(s, 47212)["preset"] is None
    finally:
        s.close()
        th.join(timeout=8)
    assert not th.is_alive()


def test_skills_confirm_flag_off_for_sim_on_for_real():
    # dev_arm (sim): skills execute immediately, no confirm -> the existing gate keeps scripting pick/place
    app, th, s = _app(load_config(arm="dev_arm"), 47213, 5.0)
    try:
        r = _say(s, 47213, "pick up the cup")
        assert r.get("confirm") is not True and r["mode"] == "auto"
    finally:
        s.close()
        th.join(timeout=7)

    # dev_arm_real: skills ask first; a yes then runs them
    app, th, s = _app(load_config(arm="dev_arm_real"), 47214, 6.0)
    try:
        r = _say(s, 47214, "pick up the cup")
        assert r["confirm"] is True and "confirm" in r["say"].lower() and r["mode"] != "auto"
        r = _say(s, 47214, "yes")
        assert r["mode"] == "auto"
    finally:
        s.close()
        th.join(timeout=8)
    assert not th.is_alive()

"""Voice fixes (branch voice-fixes, 2026-10-23 real-mic follow-up):

  fix 1 CONTEXT: a short motion history lets the fast path resolve "again" / "repeat" (redo the last motion) and
                 "the other way" / "opposite" / "reverse" / "undo" (the last motion reversed; a preset reversed =
                 go back to the previous pose). A compact context string is also passed to Nemotron.
  fix 5 SPEED:   a client timeout no longer hangs or sounds like an error; the app says "didn't catch that in time".

Covers the pure helpers (reverse_step, _parse_context, parse_fast with history), the Nemotron prompt carrying the
context + defaults, the timeout path with a fake client, and the app over UDP (mock, dev_arm_real): repeat moves the
same way again, "the other way" reverses a jog, "undo" returns to the previous pose, and STOP does not wipe history.
"""
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from rover_intent.config import load_config
from rover_intent.planner.nemotron import IntentParser
from rover_intent.planner.fast_path import make_cart_step, make_joint_step, make_preset_step, parse_fast, reverse_step
from rover_intent.types import Action

ROOT = Path(__file__).resolve().parents[1]
OBJ, PRE = ["cup", "bottle", "block"], ["ready", "base"]


# --- reverse_step ------------------------------------------------------------------------------------------------
def test_reverse_step_cart_joint_preset():
    assert reverse_step(make_cart_step("left", 0.05))["direction"] == "right"
    assert reverse_step(make_cart_step("up", 0.03))["direction"] == "down"
    assert reverse_step(make_cart_step("forward", 0.1))["direction"] == "back"
    j = make_joint_step(1, +1, 0.5, "left")                  # base left
    rj = reverse_step(j)
    assert rj["joint"] == 1 and rj["angle_rad"] == pytest.approx(-0.5) and rj["dir_word"] == "right"
    pstep = dict(make_preset_step("ready"), from_pose="base")
    assert reverse_step(pstep)["name"] == "base"             # undo a preset -> the previous pose
    assert reverse_step(make_preset_step("ready")) is None   # no known previous pose -> can't reverse


# --- parse_fast with history (context) ---------------------------------------------------------------------------
def _hist(step):
    return [{"steps": [step]}]


def test_repeat_replays_last_motion():
    hist = _hist(make_cart_step("left", 0.05))
    for text in ("again", "repeat", "once more", "do it again"):
        r = parse_fast(text, OBJ, PRE, history=hist)
        assert r is not None and r["steps"][0]["direction"] == "left"


@pytest.mark.parametrize("text", ["the other way", "opposite", "reverse", "reverse it", "undo", "undo that",
                                  "back the other way", "go the opposite way"])
def test_reverse_phrases_flip_last_jog(text):
    hist = _hist(make_cart_step("left", 0.05))
    r = parse_fast(text, OBJ, PRE, history=hist)
    assert r is not None and r["steps"][0]["direction"] == "right"


def test_spin_around_the_other_way_reverses_base_rotation():
    # the live bug: "Spin around the other way." had no memory and Nemotron guessed a jog. With the prior base
    # rotation in history the fast path reverses it locally.
    hist = _hist(make_joint_step(1, +1, 0.785, "left"))
    r = parse_fast("Spin around the other way.", OBJ, PRE, history=hist)
    assert r is not None and r["steps"][0]["joint"] == 1 and r["steps"][0]["angle_rad"] < 0


def test_undo_preset_returns_to_previous_pose():
    hist = [{"steps": [dict(make_preset_step("ready"), from_pose="base")]}]
    r = parse_fast("undo", OBJ, PRE, history=hist)
    assert r is not None and r["steps"][0]["kind"] == "preset" and r["steps"][0]["name"] == "base"


def test_context_without_history_defers():
    assert parse_fast("the other way", OBJ, PRE, history=[]) is None
    assert parse_fast("again", OBJ, PRE, history=None) is None


def test_context_with_a_number_is_a_fresh_command_not_a_reference_back():
    # "move left 5 cm" carries its own magnitude, so it is never read as "repeat/reverse" even with history present
    hist = _hist(make_cart_step("up", 0.05))
    r = parse_fast("move left 5 cm", OBJ, PRE, history=hist)
    assert r is not None and r["steps"][0]["direction"] == "left" and r["steps"][0]["dist_m"] == pytest.approx(0.05)


# --- Nemotron prompt carries the context + defaults, and the timeout path -----------------------------------------
class _FakeComp:
    last = None

    def create(self, **kw):
        _FakeComp.last = kw
        content = '{"action":"jog","direction":"left","say":"ok"}'
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()]})()


class _TimeoutComp:
    def create(self, **kw):
        raise TimeoutError("slow endpoint")


def _parser_with(comp):
    os.environ["FAKE_VOICE_KEY"] = "x"
    p = IntentParser("m", "http://unused", api_key_env="FAKE_VOICE_KEY")
    p.client = type("Client", (), {"chat": type("Chat", (), {"completions": comp})()})()
    return p


def test_prompt_includes_context_and_defaults():
    p = _parser_with(_FakeComp())
    p.parse("nudge the other way", ["cup"], None, ["ready", "base"], context="Moving left 5 centimetres. [at the base pose]")
    sysmsg = _FakeComp.last["messages"][0]["content"]
    assert "Moving left 5 centimetres" in sysmsg and "at the base pose" in sysmsg
    assert "joint_jog the base" in sysmsg and "shoulder" in sysmsg     # the defaults (fix 2) are stated to the model
    assert "the other way" in sysmsg                                   # context resolution guidance


def test_timeout_returns_friendly_message_not_an_error():
    p = _parser_with(_TimeoutComp())
    i = p.parse("move the cup somewhere", ["cup"])
    assert i.action == Action.unknown and "didn't catch that in time" in i.say


# --- app over UDP (mock backend, dev_arm_real = no confirm): history end to end ------------------------------------
def _app(cfg, port, seconds):
    cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
    cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"   # force the offline parser; the fast path is local
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


def _say(s, port, text):
    _drain(s)
    s.sendto(json.dumps({"kind": "utterance", "t": time.time(), "text": text, "offline": True}).encode(),
             ("127.0.0.1", port))
    return _recv(s, "reply")


def _query(s, port):
    _drain(s)
    s.sendto(json.dumps({"kind": "query", "t": time.time()}).encode(), ("127.0.0.1", port))
    return _recv(s, "state")


def _wait_tcp(s, port, pred, tries=60):
    for _ in range(tries):
        st = _query(s, port)
        if st and pred(st):
            return st
        time.sleep(0.1)
    return None


def test_app_repeat_and_reverse_a_jog():
    app, th, s = _app(load_config(arm="dev_arm_real"), 47241, 20.0)
    try:
        _say(s, 47241, "go to ready")
        _wait_tcp(s, 47241, lambda st: st["preset"] is None)
        y0 = _query(s, 47241)["tcp"][1]
        # move left (+y)
        assert _say(s, 47241, "move left 5 cm")["ok"] is True
        st = _wait_tcp(s, 47241, lambda st: st["preset"] is None and st["tcp"][1] > y0 + 0.03)
        assert st is not None, "left jog did not move +y"
        y1 = _query(s, 47241)["tcp"][1]
        # "again" repeats the left jog -> +y again
        r = _say(s, 47241, "again")
        assert r["ok"] is True and "left" in r["say"].lower()
        st = _wait_tcp(s, 47241, lambda st: st["preset"] is None and st["tcp"][1] > y1 + 0.03)
        assert st is not None, "'again' did not repeat the left jog"
        y2 = _query(s, 47241)["tcp"][1]
        # "the other way" reverses -> -y (moves right)
        r = _say(s, 47241, "the other way")
        assert r["ok"] is True and "right" in r["say"].lower()
        st = _wait_tcp(s, 47241, lambda st: st["preset"] is None and st["tcp"][1] < y2 - 0.03)
        assert st is not None, "'the other way' did not reverse the jog"
    finally:
        s.close()
        th.join(timeout=22)
    assert not th.is_alive()


def test_app_undo_preset_and_stop_keeps_history():
    app, th, s = _app(load_config(arm="dev_arm_real"), 47242, 20.0)
    ready = [0.144, -1.131, -1.007, -0.003, 0.679, -2.957]   # dev_arm_real "ready"
    home = load_config(arm="dev_arm_real")["arm"]["home"]
    try:
        # go to ready (from the start pose), then "undo" -> back to the start pose
        _say(s, 47242, "go to ready")
        st = _wait_tcp(s, 47242, lambda st: st["preset"] is None
                       and max(abs(a - b) for a, b in zip(st["q_meas"], ready)) < 0.03)
        assert st is not None, "did not reach ready"
        r = _say(s, 47242, "undo")
        assert r["ok"] is True
        st = _wait_tcp(s, 47242, lambda st: st["preset"] is None
                       and max(abs(a - b) for a, b in zip(st["q_meas"], home)) < 0.03)
        assert st is not None, "'undo' did not return to the start pose"
        # STOP must not wipe the history: a jog, STOP, then "again" still repeats the jog
        y0 = _query(s, 47242)["tcp"][1]
        _say(s, 47242, "move left 4 cm")
        _wait_tcp(s, 47242, lambda st: st["tcp"][1] > y0 + 0.02)
        _say(s, 47242, "stop")
        time.sleep(0.3)
        y1 = _query(s, 47242)["tcp"][1]
        r = _say(s, 47242, "again")
        assert r["ok"] is True and "left" in r["say"].lower()
        st = _wait_tcp(s, 47242, lambda st: st["tcp"][1] > y1 + 0.02)
        assert st is not None, "history was lost after STOP"
    finally:
        s.close()
        th.join(timeout=22)
    assert not th.is_alive()

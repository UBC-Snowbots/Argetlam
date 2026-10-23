"""Session recording + replay round trip, the Query skip and size cap, and the arbiter say_log (issue 9)."""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from rover_intent.session import SessionRecorder
from rover_intent.transport.udp import Receiver, _MSG
from rover_intent.types import BodyPose, ClutchMsg, EEGState, GripToggle, Query, Utterance

ROOT = Path(__file__).resolve().parents[1]


def _msgs():
    return [
        BodyPose(t=1.0, wrist=(0.1, 0.2, 0.3), shoulder=(0, 0, 0), seq=1, hand_open=0.2),
        Utterance(t=1.1, text="follow me"),
        ClutchMsg(t=1.2, action="toggle"),
        EEGState(t=1.3, p_grasp=0.9),
        GripToggle(t=1.4, source="clench"),
    ]


def test_recorder_skips_query_and_round_trips(tmp_path):
    rec = SessionRecorder(tmp_path, enabled=True)
    for i, m in enumerate(_msgs()):
        rec.record(10.0 + i, m)
    rec.record(99.0, Query(t=5.0))          # must NOT be recorded
    rec.close()
    lines = [json.loads(x) for x in rec.path.read_text().splitlines() if x.strip()]
    assert len(lines) == len(_msgs())        # the Query was skipped
    assert [l["t"] for l in lines] == [10.0, 11.0, 12.0, 13.0, 14.0]
    # each recorded line reconstructs to the same message the app would parse
    for line, original in zip(lines, _msgs()):
        parsed = _MSG.validate_python(line["msg"])
        assert parsed.model_dump() == original.model_dump()
    assert not any(l["msg"]["kind"] == "query" for l in lines)


def test_recorder_size_cap(tmp_path):
    rec = SessionRecorder(tmp_path, enabled=True, size_cap_mb=0.001)   # ~1 KB
    for i in range(500):
        rec.record(float(i), Utterance(t=float(i), text="x" * 50))
    assert rec._capped and rec.bytes <= rec.size_cap
    n = len([x for x in rec.path.read_text().splitlines() if x.strip()])
    assert 0 < n < 500


def test_recorder_disabled_writes_nothing(tmp_path):
    rec = SessionRecorder(tmp_path, enabled=False)
    rec.record(1.0, Utterance(t=1.0, text="hi"))
    rec.close()
    assert rec.path is None and list(tmp_path.iterdir()) == []


def _load_replay():
    spec = importlib.util.spec_from_file_location("replay_session", ROOT / "scripts/replay_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replay_reproduces_timing_and_content(tmp_path):
    rec = SessionRecorder(tmp_path, enabled=True)
    for i, m in enumerate(_msgs()):
        rec.record(10.0 + i * 0.5, m)         # 0.5 s between messages
    rec.close()
    replay = _load_replay()
    sent, gaps = [], []
    replay.replay(str(rec.path), rate=2.0, sleep=gaps.append, send=sent.append)
    assert len(sent) == len(_msgs())
    assert [s["kind"] for s in sent] == ["body", "utterance", "clutch", "eeg", "grip_toggle"]
    assert all(abs(g - 0.25) < 1e-6 for g in gaps)   # 0.5 s gaps / rate 2 = 0.25 s


def test_replay_round_trips_into_a_receiver(tmp_path):
    rec = SessionRecorder(tmp_path, enabled=True)
    for i, m in enumerate(_msgs()):
        rec.record(10.0 + i, m)
    rec.close()
    rx = Receiver(0)
    port = rx.sock.getsockname()[1]
    replay = _load_replay()
    replay.replay(str(rec.path), host="127.0.0.1", port=port, rate=1000.0, sleep=lambda s: None)
    time.sleep(0.2)
    got = [m for _, m, _ in rx.poll()]
    kinds = [m.model_dump()["kind"] for m in got]
    assert kinds == ["body", "utterance", "clutch", "eeg", "grip_toggle"]


def test_arbiter_say_log_collects_and_is_drainable():
    from rover_intent.control.arbiter import Arbiter
    from rover_intent.planner.scene import Scene
    from rover_intent.types import Action, Intent
    a = Arbiter(Scene({"cup": {"pos": [0.55, 0.15, 0.10]}}))
    a.on_intent(Intent(action=Action.follow))
    a.on_intent(Intent(action=Action.pick, object="nope"))   # SkillError -> a say
    assert any("Can't do that" in s for s in a.say_log)
    n = len(a.say_log)
    a.last_say = a.last_say                                   # re-setting the current value does not double-push
    assert len(a.say_log) == n
    a.say_log.clear()                                         # the app drains it each tick
    assert a.say_log == []

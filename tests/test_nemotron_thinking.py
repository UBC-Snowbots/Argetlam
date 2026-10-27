"""nemotron.thinking: off by default -> enable_thinking False is sent to the API (2026-10-26: 4x faster, more accurate)."""
import json

from rover_intent.planner.nemotron import IntentParser


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeClient:
    def __init__(self):
        self.kwargs = None
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw):
        self.kwargs = kw
        return _Resp(json.dumps({"action": "pick", "object": "cup", "say": "Picking up the cup."}))


def _parser(**kw):
    p = IntentParser("m", "http://x", api_key_env="NO_SUCH_KEY_FOR_TESTS", **kw)
    p.client = _FakeClient()
    return p


def test_thinking_off_by_default_sends_enable_thinking_false():
    p = _parser()
    intent = p.parse("pick up the cup", ["cup"])
    assert intent.action.value == "pick"
    assert p.client.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_on_sends_no_override():
    p = _parser(thinking=True)
    p.parse("pick up the cup", ["cup"])
    assert p.client.kwargs["extra_body"] == {}

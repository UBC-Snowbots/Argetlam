"""Natural language -> Intent via NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible API).

Only the *decision* goes to the cloud. Motion (IK, safety, the control loop) stays local.
Stop words never wait for the cloud (a Token Factory call takes 1.4-3.9 s): `urgent_stop()` checks them locally first.
Without NEBIUS_API_KEY, `parse()` falls back to a tiny keyword parser so the loop still runs offline.
"""
from __future__ import annotations

import json
import os
import re

from ..types import Action, Intent

SYSTEM = """You control a 6-axis robot arm with a two-finger gripper, standing at a table.
Objects on the table (name: description): {objects}.
Turn what the user says into one JSON command. Actions:
pick (grasp and lift an object), place (put the held object relative to a reference), move (pick + place),
grasp (close the gripper), release (open it), home, stop, follow (mirror the user's arm), reset,
clarify (you are not sure what the user means: ask them in "say", and nothing moves), unknown (not a robot command).
Use your own judgement. "say" is spoken back to the user; keep it short."""


class IntentParser:
    def __init__(self, model: str, base_url: str, api_key_env: str = "NEBIUS_API_KEY", timeout_s: float = 8.0):
        self.model = model
        key = os.environ.get(api_key_env)
        self.client = None
        if key:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=key, timeout=timeout_s)

    @property
    def online(self) -> bool:
        return self.client is not None

    def parse(self, text: str, objects: list[str], descriptions: dict[str, str] | None = None) -> Intent:
        stop = urgent_stop(text)
        if stop is not None:
            return stop
        if not self.online:
            return fallback_parse(text, objects)
        desc = descriptions or {}
        listing = "; ".join(f"{o}: {desc[o]}" if desc.get(o) else o for o in objects)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM.format(objects=listing)},
                      {"role": "user", "content": text}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "intent", "schema": Intent.model_json_schema()}},
        )
        try:
            return Intent.model_validate(json.loads(resp.choices[0].message.content))
        except Exception:
            return Intent(action=Action.unknown, say="Sorry, I didn't get that.")


_STOP = re.compile(r"\b(stop|freeze|halt|hold on|wait|whoa|emergency|abort)\b")


_RESET = re.compile(r"\b(reset|restart|start over)\b")


def urgent_stop(text: str) -> Intent | None:
    """Local, instant commands, checked before (and instead of) any LLM call: stop, then reset."""
    t = text.lower()
    if _STOP.search(t):
        return Intent(action=Action.stop, say="Stopping.")
    if _RESET.search(t):
        return Intent(action=Action.reset, say="Resetting.")
    return None


_REL = {"beside": "beside", "next to": "beside", "left of": "left_of", "right of": "right_of",
        "in front of": "in_front_of", "behind": "behind", "on": "on", "onto": "on", "into": "into", "in": "into"}


def fallback_parse(text: str, objects: list[str]) -> Intent:
    t = text.lower()
    found = sorted((m.start(), o) for o in objects for m in [re.search(rf"\b{re.escape(o)}\b", t)] if m)
    names = [o for _, o in found]
    if re.search(r"\b(follow|mirror|take over)\b", t):
        return Intent(action=Action.follow, say="Following you.")
    if re.search(r"\b(release|let go|drop|open)\b", t):
        return Intent(action=Action.release, say="Releasing.")
    if re.search(r"\bhome\b", t):
        return Intent(action=Action.home, say="Going home.")
    for phrase, rel in sorted(_REL.items(), key=lambda kv: -len(kv[0])):
        if f" {phrase} " in f" {t} " and len(names) >= 2:
            return Intent(action=Action.move, object=names[0], relation=rel, reference=names[1],
                          say=f"Moving the {names[0]}.")
    if re.search(r"\b(pick|grab|take|get)\b", t) and names:
        return Intent(action=Action.pick, object=names[0], say=f"Picking up the {names[0]}.")
    if re.search(r"\b(close|grip)\b", t):
        return Intent(action=Action.grasp, say="Closing.")
    return Intent(action=Action.unknown, say="Sorry, I didn't get that.")

"""Natural language -> Intent via NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible API).

Only the *decision* goes to the cloud. Motion (IK, safety, the control loop) stays local.
Without NEBIUS_API_KEY, `parse()` falls back to a tiny keyword parser so the loop still runs offline.
"""
from __future__ import annotations

import json
import os
import re

from ..types import Action, Intent

SYSTEM = """You turn spoken commands for a 6-axis robot arm with a gripper into JSON.
Known objects in the scene: {objects}.
Rules: use only known object names (map synonyms, e.g. 'mug' -> 'cup'). If the user wants something
moved next to/onto/into something, use action "move" with object, relation and reference.
"grab/pick up X" -> "pick". "let go/drop it/open" -> "release". "close/hold" -> "grasp".
"stop/freeze/halt" -> "stop". "follow me/mirror me/take over" -> "follow". Otherwise "unknown".
Put a short confirmation (<= 8 words) in "say"."""


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

    def parse(self, text: str, objects: list[str]) -> Intent:
        if not self.online:
            return fallback_parse(text, objects)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM.format(objects=", ".join(objects))},
                      {"role": "user", "content": text}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "intent", "schema": Intent.model_json_schema()}},
        )
        try:
            return Intent.model_validate(json.loads(resp.choices[0].message.content))
        except Exception:
            return Intent(action=Action.unknown, say="Sorry, I didn't get that.")


_REL = {"beside": "beside", "next to": "beside", "left of": "left_of", "right of": "right_of",
        "in front of": "in_front_of", "behind": "behind", "on": "on", "onto": "on", "into": "into", "in": "into"}


def fallback_parse(text: str, objects: list[str]) -> Intent:
    t = text.lower()
    found = sorted(((t.find(o), o) for o in objects if o in t))
    names = [o for _, o in found]
    if re.search(r"\b(stop|freeze|halt)\b", t):
        return Intent(action=Action.stop, say="Stopping.")
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
    if re.search(r"\b(close|hold|grip)\b", t):
        return Intent(action=Action.grasp, say="Closing.")
    return Intent(action=Action.unknown, say="Sorry, I didn't get that.")

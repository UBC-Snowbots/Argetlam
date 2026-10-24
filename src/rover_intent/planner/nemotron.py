"""Natural language -> Intent via NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible API).

Only the *decision* goes to the cloud. Motion (IK, safety, the control loop) stays local.
Stop words never wait for the cloud (a Token Factory call takes 1.4-3.9 s): `urgent_stop()` checks them locally first.
Without NEBIUS_API_KEY, `parse()` falls back to a tiny keyword parser so the loop still runs offline.
"""
from __future__ import annotations

import json
import os
import re

from ..control.presets import RETURN
from ..types import Action, Intent

SYSTEM = """You control a 6-axis robot arm with a two-finger gripper, standing at a table.
Objects on the table (name: description): {objects}.
Turn what the user says into one JSON command. Actions:
pick (grasp and lift an object), place (put the held object relative to a reference), move (pick + place),
grasp (close the gripper), release (open it),
preset (move to a named stored pose: set "preset" to one of {presets}; "stow"/"rest" mean "base", and "go back" means "{ret}"),
home (treat as preset "{ret}": go back to the start pose in the simulation - this NEVER homes the physical hardware),
jog (nudge the tool a short distance WITHOUT a camera: set "direction" to one of left/right/up/down/forward/back in the
  robot base frame as the operator faces it, and "distance_cm" to the centimetres asked for - omit it for a small default),
joint_jog (turn ONE joint: set "joint" to base/shoulder/elbow/forearm/wrist/tool, "direction" to a word like
  left/right/up/down/in/out/clockwise, and "angle_deg" to the degrees - omit it for a small default),
speed (set "speed" to slower/faster/normal to change how fast voice moves run),
stop, follow (mirror the user's arm), reset,
clarify (you are not sure what the user means: ask them in "say", and nothing moves), unknown (not a robot command).
Only use jog/joint_jog for a direct nudge with no object; to move an OBJECT use pick/place/move.
Defaults when a turn is asked in degrees with NO joint named: "rotate"/"turn"/"spin" (or "left"/"right") by N degrees
means joint_jog the base by N degrees; "up"/"down" by N degrees means joint_jog the shoulder by N degrees.
Recent commands, oldest to newest (use these if the user refers back): {context}.
"again"/"repeat" means redo the last command; "the other way"/"opposite"/"reverse"/"undo" means the last motion
reversed (a reversed preset = go back to the previous pose). "say" is spoken back to the user; keep it short.
Use your own judgement."""

# Spoken aliases -> preset name. The targets must be real profile presets (RETURN is always valid; presets.py
# refuses an unknown name). Voice "home" means RETURN (the sim start pose), NEVER the hardware home_hw button.
_POSE_ALIASES = {"stow": "base", "rest": "base", "base": "base", "ready": "ready",
                 "return": RETURN, "go back": RETURN, "back to start": RETURN, "home": RETURN}


class IntentParser:
    def __init__(self, model: str, base_url: str, api_key_env: str = "NEBIUS_API_KEY", timeout_s: float = 6.0):
        self.model = model
        self.timeout_s = timeout_s
        key = os.environ.get(api_key_env)
        self.client = None
        if key:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=key, timeout=timeout_s)

    @property
    def online(self) -> bool:
        return self.client is not None

    def parse(self, text: str, objects: list[str], descriptions: dict[str, str] | None = None,
              presets: list[str] | None = None, context: str = "") -> Intent:
        presets = list(presets or [])
        stop = urgent_stop(text)
        if stop is not None:
            return stop
        if not self.online:
            return fallback_parse(text, objects, presets)
        desc = descriptions or {}
        listing = "; ".join(f"{o}: {desc[o]}" if desc.get(o) else o for o in objects)
        preset_list = ", ".join([RETURN] + [p for p in presets if p != RETURN]) or RETURN
        system = SYSTEM.format(objects=listing, presets=preset_list, ret=RETURN, context=context or "(none)")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": text}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "intent", "schema": Intent.model_json_schema()}},
            )
        except Exception as e:  # a timeout (the client's timeout_s) must not hang the loop or sound like an error
            if "timeout" in type(e).__name__.lower():
                return Intent(action=Action.unknown, say="Sorry, I didn't catch that in time.")
            raise
        try:
            return normalize_preset(Intent.model_validate(json.loads(resp.choices[0].message.content)))
        except Exception:
            return Intent(action=Action.unknown, say="Sorry, I didn't get that.")


def normalize_preset(intent: Intent) -> Intent:
    """Fold Nemotron's pose output onto a `preset` action with a canonical name. "home" from the model means the
    sim start pose (RETURN), never the hardware home button; a spoken alias in `preset` is resolved via _POSE_ALIASES."""
    if intent.action == Action.home:
        intent.action, intent.preset = Action.preset, RETURN
    elif intent.action == Action.preset:
        intent.preset = _POSE_ALIASES.get((intent.preset or "").strip().lower(), intent.preset)
    return intent


_CONFIRM = re.compile(r"\b(yes|yeah|yep|yup|confirm|confirmed|okay|ok|sure|affirmative|proceed|do it|"
                      r"go ahead|go for it)\b")


def is_confirm(text: str) -> bool:
    """A short 'yes' to the app's confirm prompt. Bare 'go' is deliberately NOT a confirm word: it collides with
    the pose commands 'go back' / 'go home', so only the phrases 'go ahead' / 'go for it' count."""
    return bool(_CONFIRM.search(text.lower()))


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


# Common faster-whisper (base.en) mishearings of the short pose words, folded onto the real word BEFORE matching.
# Applied only inside quick_pose, which first defers any clause that names a scene object, so "store the cup" (a
# skill) still goes to Nemotron and only an object-free clause like "store it" folds onto "stow". Kept OUT of
# _POSE_ALIASES so the offline fallback / Nemotron-normalize path can never fold an object command onto a pose.
_POSE_MISHEARS = {"store": "stow", "stove": "stow", "stowe": "stow", "stoe": "stow", "stoic": "stow",
                  "reddy": "ready", "readi": "ready"}


def quick_pose(text: str, objects: list[str], presets: list[str] | None = None) -> Intent | None:
    """Instant local match for SHORT named-pose commands ("go to the ready pose", "stow it", "go back"), so they skip
    the 2-4 s cloud call. Anything longer, naming a scene object, or chaining steps ("then") returns None and goes
    to Nemotron as before. Common whisper mishearings (_POSE_MISHEARS) are folded first. Returns a preset Intent
    or None."""
    t = text.lower().strip()
    words = re.findall(r"[a-z']+", t)
    if not words or len(words) > 6 or "then" in words:
        return None
    if any(re.search(rf"\b{re.escape(o.lower())}\b", t) for o in objects):
        return None
    norm = " ".join(_POSE_MISHEARS.get(w, w) for w in words)   # object-free here, so folding a mishear is safe
    intent = fallback_parse(norm, [], presets)
    return intent if intent.action == Action.preset else None


def fallback_parse(text: str, objects: list[str], presets: list[str] | None = None) -> Intent:
    t = text.lower()
    found = sorted((m.start(), o) for o in objects for m in [re.search(rf"\b{re.escape(o)}\b", t)] if m)
    names = [o for _, o in found]
    if re.search(r"\b(follow|mirror|take over)\b", t):
        return Intent(action=Action.follow, say="Following you.")
    if re.search(r"\b(release|let go|drop|open)\b", t):
        return Intent(action=Action.release, say="Releasing.")
    # named poses (longest phrase first so "go back"/"back to start" beat "base"). RETURN is always valid;
    # a named target must be a real profile preset when the list is known. "home" = RETURN (sim start, not hardware).
    for phrase in sorted(_POSE_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            name = _POSE_ALIASES[phrase]
            if name == RETURN or not presets or name in presets:
                say = "Going back to the start pose." if name == RETURN else f"Moving to {name}."
                return Intent(action=Action.preset, preset=name, say=say)
    for phrase, rel in sorted(_REL.items(), key=lambda kv: -len(kv[0])):
        if f" {phrase} " in f" {t} " and len(names) >= 2:
            return Intent(action=Action.move, object=names[0], relation=rel, reference=names[1],
                          say=f"Moving the {names[0]}.")
    if re.search(r"\b(pick|grab|take|get)\b", t) and names:
        return Intent(action=Action.pick, object=names[0], say=f"Picking up the {names[0]}.")
    if re.search(r"\b(close|grip)\b", t):
        return Intent(action=Action.grasp, say="Closing.")
    return Intent(action=Action.unknown, say="Sorry, I didn't get that.")

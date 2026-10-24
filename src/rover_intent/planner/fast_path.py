"""Instant, strict LOCAL parser for short voice commands, run in the control loop BEFORE the Nemotron call.

Why: a real-mic test (2026-10-23) showed named poses taking a Nemotron round trip (~2 s) plus a spoken-confirm
window, so a slightly late "yes" timed out. For the short no-camera commands below we don't need the cloud at all:
this parser turns them into motion steps in the same tick (< 50 ms from receipt), and the app executes them with no
confirm (see configs confirm.presets / confirm.motion). Anything it does not FULLY understand returns None and the
app falls back to Nemotron (or the offline keyword parser), so it can never misfire on a complex sentence like
"put the cup left of the bottle". STOP is handled even earlier, by nemotron.urgent_stop, and never reaches here.

Grammar (each clause of a "X then Y then Z" sequence, up to 5 clauses):
  - named pose:   a _POSE_ALIASES phrase (stow/rest/ready/go back/return/home/...) + only filler words.
  - Cartesian jog: a direction {left,right,up,down,forward,back} in the ROBOT BASE FRAME as the operator faces it
                   (+x forward, +y left, +z up) + an optional distance. Default 5 cm; "a bit"/"a little" = 3 cm;
                   clamped to <= 15 cm. Units: cm/centimetre(s), mm/millimetre(s). The app solves IK for TCP+delta
                   holding the current tool orientation.
  - joint jog:    a joint name + a direction word + an optional angle. Default 10 deg; "a bit"/"a little" = 5 deg;
                  clamped to <= 45 deg. Units: degree(s)/deg. Joint names and the per-joint sign map (MEASURED on
                  dev_arm_real at its home pose, 2026-10-23) are below.
  - speed:        "slower" / "faster" / "normal speed" -> scale the velocity cap (app clamps to 0.25x .. 1.0x).

Sign map (dev_arm_real, measured): a direction word always moves the TCP the way the operator means, i.e. "up"
raises the TCP even though that is -q on the shoulder/wrist here:
  J1 base/turn/rotate : left = +q1, right = -q1 (left = counterclockwise seen from above).
  J2 shoulder         : up = -q2, down = +q2   (+q2 lowers the TCP).
  J3 elbow            : up = +q3, down = -q3.
  J4 forearm/roll     : counterclockwise/in = +q4, clockwise/out = -q4 (a roll; the TCP barely moves).
  J5 wrist/tilt       : up = -q5, down = +q5   (+q5 tilts the TCP down).
  J6 tool/twist       : counterclockwise/in = +q6, clockwise/out = -q6 (a roll).
"""
from __future__ import annotations

import re
from math import radians

from .nemotron import quick_pose
from ..control.presets import RETURN

# ---- distance / angle defaults and clamps ----------------------------------------------------------------------
MAX_DIST_M = 0.15       # per-command Cartesian clamp
DEFAULT_DIST_M = 0.05   # a direction with no distance
SMALL_DIST_M = 0.03     # "a bit" / "a little"
MAX_ANGLE = radians(45)
DEFAULT_ANGLE = radians(10)
SMALL_ANGLE = radians(5)

MAX_STEPS = 5

# ---- Cartesian directions (robot base frame as the operator faces the arm): (axis index, sign) -----------------
CART = {"left": (1, +1), "right": (1, -1), "up": (2, +1), "down": (2, -1),
        "forward": (0, +1), "forwards": (0, +1), "ahead": (0, +1),
        "back": (0, -1), "backward": (0, -1), "backwards": (0, -1)}

# ---- joint names -> index (1..6) -------------------------------------------------------------------------------
J_WORDS = {"base": 1, "turn": 1, "rotate": 1, "waist": 1,
           "shoulder": 2,
           "elbow": 3,
           "forearm": 4, "roll": 4,
           "wrist": 5, "tilt": 5,
           "tool": 6, "twist": 6, "gripper_rot": 6}

J_NAMES = {1: "base", 2: "shoulder", 3: "elbow", 4: "forearm", 5: "wrist", 6: "tool"}

# per-joint direction word -> sign (see the module docstring; measured on dev_arm_real at home)
J_SIGN = {
    1: {"left": +1, "right": -1, "counterclockwise": +1, "anticlockwise": +1, "ccw": +1,
        "clockwise": -1, "cw": -1},
    2: {"up": -1, "raise": -1, "lift": -1, "down": +1, "lower": +1, "drop": +1},
    3: {"up": +1, "raise": +1, "lift": +1, "down": -1, "lower": -1, "drop": -1},
    4: {"counterclockwise": +1, "anticlockwise": +1, "ccw": +1, "in": +1, "left": +1,
        "clockwise": -1, "cw": -1, "out": -1, "right": -1},
    5: {"up": -1, "raise": -1, "lift": -1, "down": +1, "lower": +1, "drop": +1},
    6: {"counterclockwise": +1, "anticlockwise": +1, "ccw": +1, "in": +1, "left": +1,
        "clockwise": -1, "cw": -1, "out": -1, "right": -1},
}
DIR_WORDS = set(CART) | {w for d in J_SIGN.values() for w in d}

NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
             "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
             "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90}

# words that may surround a command without changing it (coverage guard: anything NOT consumed must be filler)
FILLER = {"move", "jog", "go", "nudge", "shift", "slide", "the", "a", "an", "by", "to", "please",
          "now", "arm", "over", "just", "and", "degrees", "degree", "deg", "of", "little"}
# (number/unit/direction/joint tokens are consumed explicitly; "bit"/"little" come from the "a bit" rule)

_WORD = re.compile(r"[a-z0-9.\-]+")
_SEQ_SPLIT = re.compile(r"\b(?:and\s+then|then)\b|,")

# Common faster-whisper (base.en) mishearings of the MOTION words, folded before parsing. Per-token fixes plus a
# few two-word splits ("for word" -> "forward"). Conservative: only words that are not themselves commands.
_WORD_FIX = {"write": "right", "rite": "right", "wright": "right", "rights": "right",
             "foward": "forward", "fourward": "forward", "forwards": "forward", "backwards": "backward"}
_PHRASE_FIX = [(re.compile(r"\bfor\s+ward\b"), "forward"), (re.compile(r"\bfor\s+word\b"), "forward"),
               (re.compile(r"\bfore\s+ward\b"), "forward"), (re.compile(r"\bback\s+ward\b"), "backward"),
               (re.compile(r"\bclock\s+wise\b"), "clockwise"),
               (re.compile(r"\bcounter\s+clockwise\b"), "counterclockwise")]


def _fix_motion(s: str) -> str:
    s = s.lower()
    for rx, rep in _PHRASE_FIX:
        s = rx.sub(rep, s)
    return s


def _tokens(s: str) -> list[str]:
    return [_WORD_FIX.get(t, t) for t in _WORD.findall(s.lower())]


# ---- step builders (single source of the spoken text + clamps; shared with app's Nemotron path) ----------------
def make_preset_step(name: str) -> dict:
    label = "the start pose" if name == RETURN else name
    return {"kind": "preset", "name": name, "say": f"Moving to {label}."}


def make_cart_step(direction: str, dist_m: float) -> dict:
    dist_m = min(MAX_DIST_M, max(0.0, float(dist_m)))
    cm = dist_m * 100
    return {"kind": "cartjog", "direction": direction, "dist_m": dist_m,
            "say": f"Moving {direction} {cm:g} centimetre{'' if cm == 1 else 's'}."}


def make_joint_step(joint_idx: int, sign: int, angle_rad: float, dir_word: str | None = None) -> dict:
    angle_rad = min(MAX_ANGLE, abs(float(angle_rad)))
    deg = angle_rad * 180.0 / 3.141592653589793
    dw = f" {dir_word}" if dir_word else ""
    return {"kind": "jointjog", "joint": joint_idx, "angle_rad": sign * angle_rad,
            "say": f"Moving the {J_NAMES[joint_idx]}{dw} {deg:g} degree{'' if round(deg, 3) == 1 else 's'}."}


def make_speed_step(mode: str) -> dict:
    return {"kind": "speed", "mode": mode,
            "say": {"slower": "Slower.", "faster": "Faster.", "normal": "Normal speed."}[mode]}


# ---- per-clause parsers (each returns a step dict or None) -----------------------------------------------------
_SPEED = {"slower": "slower", "slow": "slower", "faster": "faster", "fast": "faster",
          "normal": "normal", "default": "normal"}


def _parse_speed(toks: list[str]) -> dict | None:
    # "slower", "faster", "normal speed", "full speed", "slow down", "speed up"
    words = [t for t in toks if t not in ("speed", "down", "up", "please", "the")]
    if len(words) == 1 and words[0] in _SPEED:
        return make_speed_step(_SPEED[words[0]])
    if toks in (["full", "speed"], ["speed", "full"]):
        return make_speed_step("normal")
    return None


def _match_pose(clause: str, objects, presets) -> dict | None:
    """Named-pose clause. WRAPS nemotron.quick_pose (the coordinator's short-pose matcher) so there is one pose
    grammar: quick_pose rejects object-naming / long / chained text; here each clause is already a single step."""
    intent = quick_pose(clause, list(objects or []), list(presets or []))
    return make_preset_step(intent.preset) if intent is not None and intent.preset else None


def _extract_number(toks: list[str]) -> tuple[float | None, str | None, set[int]]:
    """First number (digit or number word) + an adjacent unit. Returns (value, unit, consumed indices)."""
    for i, t in enumerate(toks):
        val = None
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            val = float(t)
        elif t in NUM_WORDS:
            val = float(NUM_WORDS[t])
        if val is None:
            continue
        consumed = {i}
        unit = None
        for j in (i + 1, i - 1):
            if 0 <= j < len(toks) and toks[j] in _UNITS:
                unit = _UNITS[toks[j]]
                consumed.add(j)
                break
        return val, unit, consumed
    return None, None, set()


_UNITS = {"cm": "cm", "centimetre": "cm", "centimetres": "cm", "centimeter": "cm", "centimeters": "cm",
          "mm": "mm", "millimetre": "mm", "millimetres": "mm", "millimeter": "mm", "millimeters": "mm",
          "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m",
          "degree": "deg", "degrees": "deg", "deg": "deg", "degs": "deg"}


def _has_bit(toks: list[str]) -> bool:
    s = " ".join(toks)
    return "a bit" in s or "a little" in s or "little" in toks or "slightly" in toks


def _covered(toks: list[str], consumed: set[int]) -> bool:
    """Every token is either consumed by the parse or a harmless filler word -> the whole clause is understood."""
    return all(i in consumed or toks[i] in FILLER or toks[i] in ("bit", "little", "slightly")
               for i in range(len(toks)))


def _parse_joint_jog(toks: list[str]) -> dict | None:
    jidx = [(i, J_WORDS[t]) for i, t in enumerate(toks) if t in J_WORDS]
    joints = {j for _, j in jidx}
    if len(joints) != 1:
        return None                     # need exactly one joint referenced
    joint = next(iter(joints))
    consumed = {i for i, _ in jidx}
    # direction word for this joint
    sign = dir_word = None
    for i, t in enumerate(toks):
        if t in J_SIGN[joint]:
            sign, dir_word = J_SIGN[joint][t], t
            consumed.add(i)
            break
    if sign is None:
        return None                     # a joint jog must say which way
    val, unit, num_consumed = _extract_number(toks)
    consumed |= num_consumed
    if val is not None and unit not in (None, "deg"):
        return None                     # a joint jog is in degrees, not cm/mm
    if val is not None:
        angle = radians(val)
    elif _has_bit(toks):
        angle = SMALL_ANGLE
    else:
        angle = DEFAULT_ANGLE
    if not _covered(toks, consumed):
        return None
    return make_joint_step(joint, sign, angle, dir_word)


def _parse_cart_jog(toks: list[str]) -> dict | None:
    dirs = [(i, t) for i, t in enumerate(toks) if t in CART]
    if len(dirs) != 1:
        return None
    di, direction = dirs[0]
    consumed = {di}
    val, unit, num_consumed = _extract_number(toks)
    consumed |= num_consumed
    if val is not None and unit == "deg":
        return None                     # degrees is a joint jog, not a Cartesian one
    bit = _has_bit(toks)
    verbs = {"move", "jog", "go", "nudge", "shift", "slide"}
    has_verb = any(t in verbs for t in toks)
    if val is None and not bit and not has_verb:
        return None                     # a bare direction word with no cue -> defer (avoid noise misfires)
    if val is not None:
        dist = val / 1000.0 if unit == "mm" else val if unit == "m" else val / 100.0  # default cm
    elif bit:
        dist = SMALL_DIST_M
    else:
        dist = DEFAULT_DIST_M
    if not _covered(toks, consumed):
        return None
    return make_cart_step(direction, dist)


def scale_speed(factor: float, mode: str) -> float:
    """New velocity-cap factor for a speed command, clamped to [0.25, 1.0] of the profile. slower halves, faster
    doubles, normal resets to the profile limit. Never above 1.0 (the profile's own limit)."""
    if mode == "normal":
        return 1.0
    if mode == "slower":
        return max(0.25, round(factor / 2, 4))
    if mode == "faster":
        return min(1.0, round(factor * 2, 4))
    return factor


def _parse_clause(clause: str, objects, presets) -> dict | None:
    toks = _tokens(_fix_motion(clause))   # fold motion mishearings ("write"->"right", "for word"->"forward")
    if not toks:
        return None
    # A pose name ("base") can also be a joint alias, so when the clause CLEARLY describes a jog (a joint keyword
    # with a direction, or a direction with a magnitude) try the jogs first; otherwise poses win (keeps "go back").
    has_jointkw = any(t in J_WORDS for t in toks)
    has_dir = any(t in DIR_WORDS for t in toks)
    has_mag = _extract_number(toks)[0] is not None
    clearly_motion = (has_jointkw and has_dir) or (has_dir and has_mag)
    jogs = (_parse_joint_jog, _parse_cart_jog)
    order = ((_parse_speed, *jogs, lambda t: _match_pose(clause, objects, presets)) if clearly_motion
             else (_parse_speed, lambda t: _match_pose(clause, objects, presets), *jogs))
    for parse in order:
        step = parse(toks)
        if step is not None:
            return step
    return None


def parse_fast(text: str, objects=None, presets=None) -> dict | None:
    """Return {"steps": [...], "say": "..."} if the WHOLE utterance is covered by the grammar, else None.

    `objects` are the scene object names (a clause naming one defers to Nemotron, via quick_pose) and `presets`
    the arm profile's preset names (poses gated to these + the RETURN aliases). A sequence is split on
    "then" / "and then" / commas, capped at MAX_STEPS clauses; if any clause is not fully understood the WHOLE
    utterance defers to Nemotron (never half-execute a mixed command like "move left then pick up the cup")."""
    objects = list(objects or [])
    presets = list(presets or [])
    clauses = [c.strip() for c in _SEQ_SPLIT.split(text) if c.strip()]
    if not clauses or len(clauses) > MAX_STEPS:
        return None
    steps = []
    for c in clauses:
        step = _parse_clause(c, objects, presets)
        if step is None:
            return None
        steps.append(step)
    return {"steps": steps, "say": " ".join(s["say"] for s in steps)}

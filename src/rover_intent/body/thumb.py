"""Left-hand THUMB reach gesture for PLANE body teleop.

Forward/back reach is driven by the operator's LEFT (reach) hand thumb:
    thumb UP   -> reach forward
    thumb DOWN -> reach back
    anything else (open hand, wave, fist, sideways thumb, missing/low-confidence hand) -> hold

`classify_thumb` is a PURE function of the 21 MediaPipe Hands landmarks (normalized image
coords, x to the right, y DOWN), so it is testable without a camera. It only reads x, y (never
the hand z). `ThumbReach` adds a short dwell so flicker does not jog the arm: a direction must
hold continuously for ~0.2 s before it commits, and the output drops to "none" (hold) at once
when the gesture is lost or becomes ambiguous (so a reversal passes through "none").

Why compute here (VM) and not on the laptop: the body client sends the reach-hand landmarks as
BodyPose.reach_hand2d (optional, backwards compatible), so the single, unit-tested classifier
lives in the repo instead of being duplicated inside the standalone client.
"""
from __future__ import annotations

import math

# MediaPipe Hands landmark indices (21 points).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
MIDDLE_MCP = 9
# (mcp, pip, tip) for the four non-thumb fingers (index, middle, ring, pinky)
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]

# Gate thresholds, all relative to the palm scale (wrist -> middle MCP distance), so they are
# independent of how large the hand looks in the image.
THUMB_EXTEND = 0.55   # thumb tip-to-MCP span, as a fraction of palm scale, to count as extended
CURL_RATIO = 1.1      # a finger is curled when its tip-to-wrist < CURL_RATIO * its mcp-to-wrist
MIN_CURLED = 3        # at least this many of the 4 fingers curled (an open hand / wave is NOT a thumb)
THUMB_DIR = 0.35      # thumb vertical span, as a fraction of palm scale, for a clear up/down read
THUMB_DWELL_S = 0.2   # default dwell before a direction commits (anti-flicker)


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify_thumb(landmarks) -> str:
    """Map LEFT-hand landmarks to "up" / "down" / "none". Pure: no state, no camera.

    `landmarks` is a sequence of 21 points, each (x, y) or (x, y, z) in normalized image coords
    (y points DOWN), or None. Missing / too-short / degenerate input returns "none" (hold).
    A thumb reads up/down only when it is extended, the other fingers are curled, and it points
    clearly along the image y axis (not sideways)."""
    if landmarks is None or len(landmarks) < 21:
        return "none"
    P = [(float(p[0]), float(p[1])) for p in landmarks]
    scale = _d(P[WRIST], P[MIDDLE_MCP])
    if scale < 1e-6:
        return "none"
    # 1) the thumb must be extended (tip well away from the thumb MCP, not folded into the palm)
    if _d(P[THUMB_TIP], P[THUMB_MCP]) < THUMB_EXTEND * scale:
        return "none"
    # 2) the other four fingers must be curled (an open hand or a wave is NOT a thumbs-up)
    curled = sum(1 for mcp, _pip, tip in FINGERS
                 if _d(P[tip], P[WRIST]) < CURL_RATIO * _d(P[mcp], P[WRIST]))
    if curled < MIN_CURLED:
        return "none"
    # 3) the thumb must point clearly up or down in IMAGE y, not sideways
    dx = P[THUMB_TIP][0] - P[THUMB_MCP][0]
    dy = P[THUMB_TIP][1] - P[THUMB_MCP][1]
    if abs(dy) <= abs(dx) or abs(dy) < THUMB_DIR * scale:
        return "none"
    return "up" if dy < 0 else "down"   # image y points DOWN: tip above the MCP = thumb up


class ThumbReach:
    """Dwell + immediate-release wrapper over `classify_thumb`, so flicker does not jog the arm.

    A direction ("up"/"down") must hold continuously for `dwell_s` before it is committed to the
    output `state`; a "none" reading (gesture lost or ambiguous) drops `state` to "none" at once.
    A reversal therefore passes through "none" until the new direction has held `dwell_s`."""

    def __init__(self, dwell_s: float = THUMB_DWELL_S):
        self.dwell_s = float(dwell_s)
        self.state = "none"
        self._cand = "none"
        self._t_cand = None

    def reset(self):
        self.state = "none"
        self._cand = "none"
        self._t_cand = None

    def update(self, raw: str, now: float) -> str:
        """Feed one raw classifier label and the frame time; return the committed state."""
        if raw not in ("up", "down"):
            self._cand, self._t_cand, self.state = "none", None, "none"
            return self.state
        if raw != self._cand:                    # a new direction restarts the dwell timer
            self._cand, self._t_cand = raw, now
        held = self._t_cand is not None and (now - self._t_cand) >= self.dwell_s
        self.state = raw if held else "none"
        return self.state

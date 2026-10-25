"""LEFT-hand THUMB reach classifier (body/thumb.py).

Pure, camera-free: classify_thumb maps 21 synthetic MediaPipe Hands landmarks (normalized image
coords, x right, y DOWN) to "up" / "down" / "none", and ThumbReach adds an anti-flicker dwell
(a direction must hold ~0.2 s to start; "none" drops out at once). thumb_hand() builds the
synthetic hands and is reused by tests/test_plane.py to drive PlaneMapper reach.
"""
import pytest

from rover_intent.body.thumb import ThumbReach, classify_thumb

# Finger (mcp -> pip, dip, tip) chains, and the finger MCP anchor points (palm pointing up, y DOWN).
_FINGER_CHAIN = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
_FINGER_MCP = {5: (0.44, 0.60), 9: (0.50, 0.60), 13: (0.56, 0.60), 17: (0.61, 0.61)}
# thumb tip (4) + ip (3) placements for each thumb pose (mcp(2) = (0.37, 0.67), palm scale ~0.20)
_THUMB_POSE = {"up": [(0.33, 0.59), (0.29, 0.49)], "down": [(0.37, 0.75), (0.38, 0.85)],
               "side": [(0.27, 0.67), (0.17, 0.66)], "tuck": [(0.41, 0.65), (0.45, 0.63)]}


def thumb_hand(thumb="up", fingers="curled"):
    """A synthetic 21-point hand. thumb in {up, down, side, tuck}; fingers in {curled, extended}."""
    P = [(0.0, 0.0)] * 21
    P[0] = (0.50, 0.80)                    # wrist
    P[1] = (0.41, 0.73); P[2] = (0.37, 0.67)   # thumb CMC, MCP
    P[3], P[4] = _THUMB_POSE[thumb]            # thumb IP, TIP
    for mcp, pip, dip, tip in _FINGER_CHAIN:
        mx, my = _FINGER_MCP[mcp]
        P[mcp] = (mx, my)
        if fingers == "extended":              # finger points up, away from the wrist
            P[pip], P[dip], P[tip] = (mx, my - 0.08), (mx, my - 0.14), (mx, my - 0.20)
        else:                                  # curled: tip folds back toward the palm / wrist
            P[pip], P[dip], P[tip] = (mx, my - 0.03), (mx, my + 0.02), (mx, my + 0.08)
    return P


# --- classify_thumb ------------------------------------------------------------

def test_thumb_up():
    assert classify_thumb(thumb_hand("up", "curled")) == "up"


def test_thumb_down():
    assert classify_thumb(thumb_hand("down", "curled")) == "down"


def test_open_hand_is_none():
    # thumb up but all fingers extended (an open hand / wave) must NOT read as a thumbs-up
    assert classify_thumb(thumb_hand("up", "extended")) == "none"


def test_fist_is_none():
    # thumb tucked across the palm (not extended) -> none
    assert classify_thumb(thumb_hand("tuck", "curled")) == "none"


def test_sideways_thumb_is_none():
    assert classify_thumb(thumb_hand("side", "curled")) == "none"


def test_missing_or_short_hand_is_none():
    assert classify_thumb(None) == "none"
    assert classify_thumb(thumb_hand("up")[:10]) == "none"       # too few points
    assert classify_thumb([(0.5, 0.5)] * 21) == "none"           # degenerate (zero scale)


def test_classify_reads_xy_only():
    # a z coordinate on every landmark must not change the result (only x, y are read)
    hand3d = [(x, y, 0.123) for (x, y) in thumb_hand("down", "curled")]
    assert classify_thumb(hand3d) == "down"


# --- ThumbReach dwell + immediate release --------------------------------------

def test_dwell_before_commit():
    tr = ThumbReach(dwell_s=0.2)
    t = 0.0
    # within the dwell window the output is still "none"
    for _ in range(5):
        assert tr.update("up", t) == "none"
        t += 0.033
    # past the dwell it commits to "up"
    committed = False
    for _ in range(5):
        if tr.update("up", t) == "up":
            committed = True
        t += 0.033
    assert committed and tr.state == "up"


def test_flicker_does_not_commit():
    tr = ThumbReach(dwell_s=0.2)
    t = 0.0
    for _ in range(3):              # a brief flick of "up" (~0.1 s) then it is gone
        assert tr.update("up", t) == "none"
        t += 0.033
    assert tr.update("none", t) == "none"   # released: dwell resets, never commits
    t += 0.033
    assert tr.update("up", t) == "none"     # a fresh "up" starts the dwell over


def test_release_is_immediate():
    tr = ThumbReach(dwell_s=0.2)
    t = 0.0
    while tr.update("up", t) != "up":       # commit to up
        t += 0.033
    t += 0.033
    assert tr.update("none", t) == "none"   # a single "none" drops out at once


def test_reversal_passes_through_none():
    tr = ThumbReach(dwell_s=0.2)
    t = 0.0
    while tr.update("up", t) != "up":
        t += 0.033
    t += 0.033
    assert tr.update("down", t) == "none"   # opposite direction is not committed instantly
    # it only commits once "down" has held the dwell
    committed = False
    for _ in range(10):
        t += 0.033
        if tr.update("down", t) == "down":
            committed = True
    assert committed

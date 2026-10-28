"""Clutch pause-on-tracking-loss + fast plane engage (2026-10-28).

Real sessions (logs/app_20261028_*.log) showed palm-up -> engaged took 3-20 s because the stillness countdown kept
restarting, and EVERY following session ended in "tracking lost" (mostly the hand cropping out of frame when Pranav
reached down), each one gliding the arm to base. The fixes:
  1. Plane mode engages fast (short calib_s + loose still_m, from the config).
  2. While following in plane mode, losing the arm pauses (holds the target, no on_drop glide) and resumes without a
     re-calibration when the arm is back; a long absence still falls through to the "tracking lost" stow.

Unit tests drive the Clutch directly; the app test drives dev_arm_real (plane) over a spare port with a mock backend.
"""
import json
import socket
import threading
import time

import numpy as np
import pytest

from rover_intent import app
from rover_intent.body.clutch import Clutch
from rover_intent.config import load_config

from test_plane import _pose


# --- unit: fast engage ----------------------------------------------------------
def test_fast_engage_plane_config():
    """With the plane config (calib_s 0.5, loose still_m), a steady hand engages in ~0.5 s, and a small wobble inside
    still_m does NOT restart the countdown (that was the 3-20 s finicky start)."""
    c = Clutch(calib_s=0.5, still_m=0.08)
    rel = np.zeros(3)
    assert c.toggle(0.0).kind == "calibrating"
    assert c.update(rel, False, False, 0.1) is None
    assert c.update(rel + [0.05, 0, 0], False, False, 0.3) is None   # 5 cm < still_m 0.08: countdown keeps running
    ev = c.update(rel, False, False, 0.55)                            # 0.5 s steady -> engaged
    assert ev is not None and ev.kind == "engaged" and c.state == "following"


# --- unit: pause / resume / release ---------------------------------------------
def _engage(c):
    """Toggle on and run the calibration to 'following'."""
    c.toggle(0.0)
    rel = np.zeros(3)
    t = 0.0
    while c.state != "following":
        t += 0.05
        c.update(rel, False, False, t)
        assert t < 5.0
    return t


def test_pause_after_tracking_loss_holds_not_released():
    c = Clutch(calib_s=0.3, still_m=0.08, pause_on_lost=True, track_s=0.5, pause_max_s=8.0)
    _engage(c)
    # arm not visible for > track_s while following -> PAUSED (not released, no reason set -> no on_drop)
    assert c.watchdog(0.8, last_packet_t=0.79, last_visible_t=0.5) is None   # 0.3 s invisible < track_s 0.5
    ev = c.watchdog(1.2, last_packet_t=1.19, last_visible_t=0.5)             # 0.7 s invisible > track_s 0.5
    assert ev is not None and ev.kind == "paused" and c.state == "paused"
    assert c.release_reason is None   # a pause is NOT a release: on_drop must not fire


def test_resume_needs_continuous_visibility_then_resumes():
    c = Clutch(calib_s=0.3, still_m=0.08, pause_on_lost=True, track_s=0.5, resume_s=0.3)
    _engage(c)
    c.watchdog(1.2, last_packet_t=1.19, last_visible_t=0.4)
    assert c.state == "paused"
    rel = np.zeros(3)
    assert c.update(rel, False, False, 2.0) is None          # visible again, resume timer starts
    assert c.update(None, False, False, 2.1) is None          # flicker: not visible -> resume timer resets
    assert c.state == "paused"
    assert c.update(rel, False, False, 2.2) is None           # restart
    ev = c.update(rel, False, False, 2.55)                    # 0.35 s >= resume_s 0.3 of continuous visibility
    assert ev is not None and ev.kind == "resumed" and c.state == "following"


def test_long_pause_releases_tracking_lost():
    c = Clutch(calib_s=0.3, still_m=0.08, pause_on_lost=True, track_s=0.5, pause_max_s=1.0)
    _engage(c)
    c.watchdog(1.2, last_packet_t=1.19, last_visible_t=0.4)
    assert c.state == "paused"
    assert c.watchdog(2.0, last_packet_t=1.99, last_visible_t=0.4) is None   # still within pause_max_s (0.8 s paused)
    ev = c.watchdog(2.3, last_packet_t=2.29, last_visible_t=0.4)             # > pause_max_s 1.0 -> full release
    assert ev is not None and ev.kind == "released"
    assert c.release_reason == "tracking lost" and c.state == "idle"         # so arm.on_drop still stows


def test_link_lost_releases_immediately_from_paused():
    c = Clutch(calib_s=0.3, still_m=0.08, pause_on_lost=True, track_s=0.5, pause_max_s=8.0, link_s=0.5)
    _engage(c)
    c.watchdog(1.2, last_packet_t=1.19, last_visible_t=0.4)
    assert c.state == "paused"
    ev = c.watchdog(2.0, last_packet_t=1.2, last_visible_t=0.4)              # 0.8 s without packets > link_s
    assert ev is not None and ev.kind == "released" and c.release_reason == "link lost" and c.state == "idle"


def test_toggle_and_stop_release_from_paused():
    c = Clutch(calib_s=0.3, still_m=0.08, pause_on_lost=True, track_s=0.5)
    _engage(c)
    c.watchdog(1.2, last_packet_t=1.19, last_visible_t=0.4)
    assert c.state == "paused"
    ev = c.toggle(2.0)                      # F toggle from paused -> release (reason None -> app tags it later)
    assert ev.kind == "released" and c.state == "idle"
    # a direct stop release (voice STOP path) also works from paused
    c2 = Clutch(calib_s=0.3, still_m=0.08, pause_on_lost=True, track_s=0.5)
    _engage(c2)
    c2.watchdog(1.2, last_packet_t=1.19, last_visible_t=0.4)
    assert c2.state == "paused"
    assert c2._release().kind == "released" and c2.state == "idle"


def test_pause_disabled_releases_as_before():
    """pause_on_lost off (arm_mirror / the default): a tracking loss releases with 'tracking lost' (on_drop stows)."""
    c = Clutch(calib_s=0.3, still_m=0.03, pause_on_lost=False, track_s=1.0)
    _engage(c)
    ev = c.watchdog(2.0, last_packet_t=1.99, last_visible_t=0.5)             # 1.5 s invisible > track_s 1.0
    assert ev is not None and ev.kind == "released" and c.release_reason == "tracking lost"


def test_calibrating_still_releases_on_tracking_loss():
    """Pause is a FOLLOWING feature: a tracking loss during calibration still releases (no half-calibrated pause)."""
    c = Clutch(calib_s=2.0, still_m=0.08, pause_on_lost=True, track_s=0.5)
    c.toggle(0.0)
    c.update(np.zeros(3), False, False, 0.1)
    assert c.state == "calibrating"
    ev = c.watchdog(1.0, last_packet_t=0.99, last_visible_t=0.3)             # 0.7 s invisible > track_s 0.5
    assert ev is not None and ev.kind == "released" and c.release_reason == "tracking lost"


# --- app level: plane mode, pause holds then resumes, long loss stows -----------
class _App:
    def __init__(self, port, seconds):
        cfg = load_config(arm="dev_arm_real")
        assert cfg["body"]["mode"] == "plane"
        cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
        # fast engage + pause on, but a short pause_max_s / track_s so the test is quick
        cfg["body"]["clutch"] = {"calib_s": 0.5, "still_m": 0.08, "palm_s": 0.6, "raise_m": 0.05,
                                 "pause_on_lost": True, "resume_s": 0.3, "pause_max_s": 2.0, "track_s": 0.5}
        cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"
        self.base = np.array(cfg["arm"]["presets"]["base"], float)
        self.th = threading.Thread(target=app.run, args=(cfg, "mock", seconds), kwargs={"record": False}, daemon=True)
        self.th.start(); time.sleep(0.5)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.s.settimeout(3)
        self.addr = ("127.0.0.1", port)

    def send_pose(self, **kw):
        self.s.sendto(json.dumps(_pose(time.time(), **kw).model_dump()).encode(), self.addr)

    def send(self, m):
        self.s.sendto(json.dumps(m).encode(), self.addr)

    def drain(self):
        self.s.settimeout(0.001)
        try:
            while True:
                self.s.recvfrom(65536)
        except (socket.timeout, BlockingIOError):
            pass
        self.s.settimeout(3)

    def query(self):
        self.send({"kind": "query", "t": time.time()})
        while True:
            m = json.loads(self.s.recvfrom(65536)[0])
            if m.get("kind") == "state":
                return m

    def feed(self, secs, **kw):
        t_end = time.time() + secs
        while time.time() < t_end:
            self.send_pose(**kw); time.sleep(1 / 30); self.drain()

    def join(self):
        self.th.join(timeout=14)


def test_app_plane_pause_holds_resumes_then_stows():
    a = _App(47280, seconds=20.0)
    assert a.query()["mapping"] == "PlaneMapper"
    a.send({"kind": "utterance", "t": time.time(), "text": "follow me"})
    # hold still to calibrate, then swing the control hand right + up so the TCP leaves base
    t_end = time.time() + 4.0
    i = 0
    while time.time() < t_end:
        k = max(0, i - 15)
        a.send_pose(rwx=0.35 - min(0.12, k * 0.004), rwy=0.50 - min(0.12, k * 0.004))
        time.sleep(1 / 30); a.drain(); i += 1
    moved = a.query()
    assert moved["clutch"] == "following" and moved["mode"] == "follow"
    q_moved = np.asarray(moved["q_meas"])
    off_base = float(np.max(np.abs(q_moved - a.base)))
    assert off_base > 0.03, f"TCP should have tracked the hand off base, only {off_base:.3f} rad"

    # HIDE the arm (packets still flow, visible=False) for ~1.2 s: > track_s but < pause_max_s + resume gap.
    # This must PAUSE (hold), not glide to base.
    a.feed(1.2, visible=False)
    paused = a.query()
    assert paused["clutch"] == "paused", f"expected paused, got {paused['clutch']}"
    q_pause = np.asarray(paused["q_meas"])
    held = float(np.max(np.abs(q_pause - q_moved)))
    assert held < 0.02, f"the arm must HOLD its target while paused, moved {held:.3f} rad"
    to_base = float(np.max(np.abs(q_pause - a.base)))
    assert to_base > 0.03, f"paused arm must NOT glide to base (on_drop), only {to_base:.3f} rad from base"

    # SHOW the arm again at the SAME hand position: it resumes without a re-calibration and WITHOUT a jump.
    a.feed(1.0, rwx=0.35 - 0.12, rwy=0.50 - 0.12)
    resumed = a.query()
    assert resumed["clutch"] == "following", f"expected following after resume, got {resumed['clutch']}"
    jump = float(np.max(np.abs(np.asarray(resumed["q_meas"]) - q_pause)))
    assert jump < 0.03, f"resume must be continuous (no jump), moved {jump:.3f} rad"

    # HIDE for longer than pause_max_s: now the full 'tracking lost' release fires -> on_drop glides to base.
    a.feed(6.0, visible=False)
    end = a.query()
    a.join()
    assert end["clutch"] == "idle", f"a long absence must release, got {end['clutch']}"
    err = float(np.max(np.abs(np.asarray(end["q_meas"]) - a.base)))
    assert err < 0.03, f"after a long absence the arm should stow at base, still {err:.3f} rad away"

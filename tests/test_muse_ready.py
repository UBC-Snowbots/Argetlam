"""muse_ready: the put-on / calibrate / confirm / go routine, end to end on the synthetic board.

No hardware. A localhost fake hub on an OS-chosen high port (never the live-system ports). Input is injected (a
scripted fake stdin), so the whole flow (fit check -> calibration -> Enter -> a short stream) runs unattended.
"""
import json
import socket
import threading
import time

import numpy as np
import pytest

from rover_intent.eeg import muse_ready
from rover_intent.eeg.muse_ready import ReadyConfig, fit_check
from rover_intent.eeg.sunday import SundayConfig
from rover_intent.eeg.stream_hub import LineReader


class FakeHub:
    """Minimal TCP hub: accepts one sender, collects every newline-JSON record it sends. Spare OS-chosen port."""

    def __init__(self):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.records: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        self.srv.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self.srv.accept()
            except (OSError, socket.timeout):
                continue
            for rec in LineReader(conn):
                self.records.append(rec)
            conn.close()

    def close(self):
        self._stop.set()
        try:
            self.srv.close()
        except OSError:
            pass

    @property
    def headers(self):
        return [r for r in self.records if r.get("kind") == "eeg_header"]

    @property
    def frames(self):
        return [r for r in self.records if r.get("kind") == "eeg"]


def _tiny_cfg():
    # shrink every duration so the synthetic smoke test is a few seconds, not ~2 minutes
    # fit_window_s must give >= 0.5 s of samples (the quality check needs that many) and fit_pass_s forces at least
    # two good windows in a row, so the synthetic smoke test still exercises the "good for N s in a row" path.
    return ReadyConfig(fit_window_s=0.6, fit_pass_s=0.5, fit_max_s=15.0, rest_s=1.0, double_reps=2, double_s=1.0,
                       neg_chew_s=1.0, neg_blink_s=1.0)


def _free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_fit_check_passes_on_synthetic():
    """The synthetic board reads 'good' on all four sensors, so the fit check returns without needing Enter."""
    from rover_intent.eeg.clench import open_board, _read
    board, fs, chans = open_board("synthetic")
    _read(board, chans)
    try:
        q = fit_check(board, fs, chans, SundayConfig(), _tiny_cfg(), quiet=True, poll_enter=lambda: False)
    finally:
        board.stop_stream()
        board.release_session()
    assert q["ok"] is True
    assert not q.get("skipped")
    assert set(q["channels"]) == {"TP9", "AF7", "AF8", "TP10"}


def test_end_to_end_synthetic(tmp_path):
    """Full routine on synthetic: fit check -> calibration (profile written to tmp) -> Enter -> short stream to a
    fake hub. Verify the profile file and that a header + at least one frame reached the hub."""
    hub = FakeHub()
    profiles_dir = tmp_path / "profiles"

    # scripted stdin: 's' continues a poor calibration fit and also releases the READY-CHECK Enter.
    def fake_input(prompt=""):
        return "s"

    try:
        res = muse_ready.run(
            "tester", "synthetic", cfg=_tiny_cfg(), quiet=True,
            profiles_dir=str(profiles_dir), ask=fake_input, poll_enter=lambda: False,
            hub_host="127.0.0.1", hub_port=hub.port,
            app_host="127.0.0.1", app_port=_free_udp_port(),
            stream_seconds=1.5)
    finally:
        time.sleep(0.3)
        hub.close()

    # profile written to the tmp dir (never the real data/profiles)
    prof_path = profiles_dir / "clench_tester.json"
    assert prof_path.exists(), "clench profile was not written"
    profile = json.loads(prof_path.read_text())
    assert "rest_level" in profile and "config" in profile and "stats" in profile
    assert len(profile["rest_level"]) == 4

    assert res.get("streamed") is True
    assert res["fit"]["ok"] is True

    # header + frames reached the hub
    assert len(hub.headers) >= 1, "no eeg_header reached the hub"
    hdr = hub.headers[0]
    assert hdr["channels"] == ["TP9", "AF7", "AF8", "TP10"]
    assert hdr["board"] == "synthetic"
    assert len(hub.frames) >= 1, "no eeg frames reached the hub"
    # each frame carries the 4 channels
    assert len(hub.frames[0]["x"]) == 4


def test_skip_calib_uses_existing_profile(tmp_path):
    """--skip-calib loads a given profile and does fit check + Enter + a short stream, writing no new profile."""
    from rover_intent.eeg.clench import open_board, _read, calibrate
    # make a profile from a quick synthetic capture
    board, fs, chans = open_board("synthetic")
    _read(board, chans)
    time.sleep(3.0)  # calibrate() skips a 1 s settle window, so capture well over 1 s of samples
    x = _read(board, chans)
    board.stop_stream()
    board.release_session()
    profile = calibrate(x, x, x, fs)
    prof_path = tmp_path / "clench_given.json"
    prof_path.write_text(json.dumps(profile))

    hub = FakeHub()
    try:
        res = muse_ready.run(
            "given", "synthetic", cfg=_tiny_cfg(), quiet=True,
            skip_calib=True, profile_path=str(prof_path),
            profiles_dir=str(tmp_path / "profiles"), ask=lambda prompt="": "s", poll_enter=lambda: False,
            hub_host="127.0.0.1", hub_port=hub.port,
            app_host="127.0.0.1", app_port=_free_udp_port(),
            stream_seconds=1.0)
    finally:
        time.sleep(0.3)
        hub.close()
    assert res.get("streamed") is True
    # no new profile was written in --skip-calib mode
    assert not (tmp_path / "profiles" / "clench_given.json").exists()
    assert len(hub.headers) >= 1 and len(hub.frames) >= 1


def test_no_stream_stops_after_enter(tmp_path):
    """--no-stream stops after the Enter: a profile is written, nothing is streamed."""
    res = muse_ready.run(
        "nos", "synthetic", cfg=_tiny_cfg(), quiet=True, no_stream=True,
        profiles_dir=str(tmp_path / "profiles"), ask=lambda prompt="": "s", poll_enter=lambda: False)
    assert res.get("streamed") is None
    assert (tmp_path / "profiles" / "clench_nos.json").exists()

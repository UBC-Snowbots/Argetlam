"""Guided ~10 min Muse 2 session recorder: eyes-open/closed blocks + full-screen SSVEP flicker, with EEG-clock
markers and a frame-timing log. Standalone on the laptop, same spirit as laptop/body_client.py and eeg/clench.py.

What it records, so docs/muse_readiness.md numbers can be reproduced on the real headband:
  - alpha blocks: alternating EYES OPEN / EYES CLOSED with text + beep cues (the eyes-closed fallback control)
  - SSVEP blocks: N full-screen boxes each flickering at a chosen frequency, one cued per trial (look at it)
  - flicker is FRAME-LOCKED to the display: each box is on when (frame_index * f / refresh) mod 1 < 0.5, so the
    real frequency is exactly refresh/round(refresh/f); we log the measured refresh and every dropped frame
  - markers go in via BrainFlow insert_marker, i.e. on the SAME clock as the EEG samples (no cross-clock drift)
  - saves an npz: eeg (n_ch, n), markers (sample_index, code), fs, channels, frame_log, and the full plan/meta

Usage (laptop with the Muse; Bluetooth on):
  python -m rover_intent.eeg.muse_session record --out data/eeg_recordings/session_pranav.npz --board muse2
  python -m rover_intent.eeg.muse_session record --board synthetic --out /tmp/s.npz   # dry pipeline check
  python -m rover_intent.eeg.muse_session --check-display                              # flicker timing only, no board
  --freqs 12,15 (2-class) | 12,13,14,15 (4-class); --ssvep-s 5 --reps 6 --alpha-blocks 3 --block-s 20
Tests run pygame headless with SDL_VIDEODRIVER=dummy.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MARK = {"settle": 1, "rest": 2, "eyes_open": 10, "eyes_closed": 11, "ssvep_base": 100}  # ssvep: 100 + freq_index


@dataclass
class SessionConfig:
    freqs: tuple[float, ...] = (12.0, 15.0)
    ssvep_s: float = 5.0
    rest_s: float = 1.5
    reps: int = 6               # SSVEP trials per frequency
    alpha_blocks: int = 3       # pairs of open/closed
    block_s: float = 20.0
    settle_s: float = 5.0
    seed: int = 0

    def ssvep_trials(self) -> float:
        return self.reps * len(self.freqs)

    def est_minutes(self) -> float:
        alpha = self.settle_s + self.alpha_blocks * 2 * self.block_s
        ssvep = self.settle_s + self.ssvep_trials() * (self.ssvep_s + self.rest_s)
        return (alpha + ssvep) / 60.0


@dataclass
class Phase:
    kind: str          # "settle" | "eyes_open" | "eyes_closed" | "ssvep" | "rest"
    seconds: float
    code: int
    cued_freq: float | None = None   # for ssvep phases: which box to look at
    text: str = ""


def build_timeline(cfg: SessionConfig) -> list[Phase]:
    """Deterministic plan (pure; testable without pygame). Alpha blocks first, then SSVEP trials in a fixed order."""
    rng = np.random.default_rng(cfg.seed)
    phases = [Phase("settle", cfg.settle_s, MARK["settle"], text="Relax, sit still. Starting soon.")]
    for _ in range(cfg.alpha_blocks):
        phases.append(Phase("eyes_open", cfg.block_s, MARK["eyes_open"], text="EYES OPEN - look at the cross"))
        phases.append(Phase("eyes_closed", cfg.block_s, MARK["eyes_closed"], text="EYES CLOSED - relax (beep to open)"))
    phases.append(Phase("settle", cfg.settle_s, MARK["settle"], text="SSVEP next: look at the highlighted box."))
    order = [f for f in cfg.freqs for _ in range(cfg.reps)]
    rng.shuffle(order)
    for f in order:
        idx = list(cfg.freqs).index(f)
        phases.append(Phase("rest", cfg.rest_s, MARK["rest"], text="rest"))
        phases.append(Phase("ssvep", cfg.ssvep_s, MARK["ssvep_base"] + idx, cued_freq=float(f),
                            text=f"LOOK at the {f:g} Hz box"))
    return phases


def flicker_on(frame_index: int, freq: float, refresh: float) -> bool:
    """Frame-locked square-wave: real frequency = refresh / round(refresh/freq). On for the first half of each
    frame period, so it never depends on wall-clock timing (only the frame counter)."""
    period = max(2, round(refresh / freq)) if freq > 0 else 2
    return (frame_index % period) < period / 2


def realized_freq(freq: float, refresh: float) -> float:
    period = max(2, round(refresh / freq)) if freq > 0 else 2
    return refresh / period


def analyse_frame_log(times: list[float]) -> dict:
    """times: perf_counter at each rendered frame. Returns measured refresh and dropped-frame count."""
    t = np.asarray(times, float)
    if len(t) < 3:
        return {"n_frames": len(t), "measured_refresh_hz": 0.0, "dropped_frames": 0, "frac_dropped": 0.0}
    dt = np.diff(t)
    med = float(np.median(dt))
    refresh = 1.0 / med if med > 0 else 0.0
    dropped = int(np.sum(dt > 1.5 * med))
    return {"n_frames": int(len(t)), "measured_refresh_hz": round(refresh, 2),
            "nominal_frame_ms": round(med * 1000, 3), "dropped_frames": dropped,
            "frac_dropped": round(dropped / len(dt), 4), "max_frame_ms": round(float(dt.max()) * 1000, 2)}


# --- board I/O (BrainFlow) ------------------------------------------------------------------------------------------
def open_board(kind: str, serial: str = "", mac: str = "", name: str = ""):
    """Reuses the board map from eeg.clench (muse2 / muse2-bled / synthetic)."""
    from rover_intent.eeg.clench import open_board as _open
    return _open(kind, serial, mac, name)


# --- flicker rendering (pygame) -------------------------------------------------------------------------------------
def _box_rects(screen_w, screen_h, n):
    """Lay out n flicker boxes across the screen."""
    import pygame
    bw, bh = int(screen_w / (n + 1)), int(screen_h * 0.4)
    gap = (screen_w - n * bw) / (n + 1)
    y = int(screen_h * 0.3)
    return [pygame.Rect(int(gap + i * (bw + gap)), y, bw, bh) for i in range(n)]


def check_display(freqs, seconds: float = 6.0, windowed: bool = True) -> dict:
    """Measure real flicker timing with no board. Returns the frame-log analysis plus realized frequencies."""
    import os
    import pygame
    pygame.init()
    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    size = (800, 600)
    flags = 0 if (windowed or headless) else pygame.FULLSCREEN
    try:
        screen = pygame.display.set_mode(size, flags, vsync=1)
    except Exception:  # noqa: BLE001
        screen = pygame.display.set_mode(size, flags)
    clock = pygame.time.Clock()
    # first second: estimate refresh
    probe = []
    for _ in range(90):
        screen.fill((0, 0, 0))
        pygame.display.flip()
        probe.append(time.perf_counter())
        clock.tick(240)
    refresh = analyse_frame_log(probe)["measured_refresh_hz"] or 60.0
    rects = _box_rects(size[0], size[1], len(freqs))
    frame_times, fi, t_end = [], 0, time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                t_end = 0
        screen.fill((0, 0, 0))
        for r, f in zip(rects, freqs):
            if flicker_on(fi, f, refresh):
                pygame.draw.rect(screen, (255, 255, 255), r)
        pygame.display.flip()
        frame_times.append(time.perf_counter())
        fi += 1
        clock.tick(240)
        if headless and fi > int(seconds * 60):
            break
    pygame.quit()
    rep = analyse_frame_log(frame_times)
    rep["requested_freqs"] = list(map(float, freqs))
    rep["realized_freqs"] = [round(realized_freq(f, rep["measured_refresh_hz"] or 60.0), 3) for f in freqs]
    return rep


def _read(board, chans):
    from rover_intent.eeg.clench import _read as r
    return r(board, chans)


def run_session(cfg: SessionConfig, board_kind: str, out: str, serial="", mac="", name="",
                windowed: bool = True, headless_frames: int | None = None) -> dict:
    """Execute the timeline on screen while streaming the board; insert markers on the EEG clock; save npz."""
    import os
    import pygame
    from brainflow.board_shim import BoardShim

    phases = build_timeline(cfg)
    board, fs, chans = open_board(board_kind, serial, mac, name)
    marker_row = BoardShim.get_marker_channel(board.board_id) if hasattr(board, "board_id") else None
    pygame.init()
    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy" or headless_frames is not None
    size = (800, 600)
    flags = 0 if (windowed or headless) else pygame.FULLSCREEN
    try:
        screen = pygame.display.set_mode(size, flags, vsync=1)
    except Exception:  # noqa: BLE001
        screen = pygame.display.set_mode(size, flags)
    font = pygame.font.Font(None, 48)
    clock = pygame.time.Clock()
    rects = _box_rects(size[0], size[1], len(cfg.freqs))

    # estimate refresh before starting
    probe = []
    for _ in range(90):
        screen.fill((0, 0, 0)); pygame.display.flip(); probe.append(time.perf_counter()); clock.tick(240)
    refresh = analyse_frame_log(probe)["measured_refresh_hz"] or 60.0

    _read(board, chans)  # drop startup junk
    t_start = time.perf_counter()
    frame_times, fi = [], 0
    markers = []  # (sample_index_estimate, code) also inserted into the stream
    n_seen = 0

    def beep():
        pass  # audio optional; text cue is primary. (A real build can add a short tone here.)

    try:
        for ph in phases:
            board.insert_marker(float(ph.code))
            # record an approximate sample index for the marker (stream marker row is the ground truth)
            markers.append((n_seen, ph.code, ph.cued_freq))
            t_end = time.perf_counter() + ph.seconds
            while time.perf_counter() < t_end:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                        raise KeyboardInterrupt
                screen.fill((0, 0, 0))
                if ph.kind == "ssvep":
                    for r, f in zip(rects, cfg.freqs):
                        if flicker_on(fi, f, refresh):
                            pygame.draw.rect(screen, (255, 255, 255), r)
                        if f == ph.cued_freq:
                            pygame.draw.rect(screen, (0, 200, 0), r, 6)  # cue border
                else:
                    msg = font.render(ph.text, True, (230, 230, 230))
                    screen.blit(msg, msg.get_rect(center=(size[0] // 2, size[1] // 2)))
                    if ph.kind in ("eyes_open", "settle"):
                        pygame.draw.line(screen, (120, 120, 120), (size[0]//2-15, size[1]//2+80),
                                         (size[0]//2+15, size[1]//2+80), 3)
                pygame.display.flip()
                frame_times.append(time.perf_counter())
                fi += 1
                x = _read(board, chans)
                n_seen += x.shape[1]
                clock.tick(240)
                if headless and len(frame_times) >= (headless_frames or 240):
                    raise KeyboardInterrupt  # tests: stop quickly
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(0.1)
        full = board.get_board_data()
        board.stop_stream()
        board.release_session()
        pygame.quit()

    eeg = full[chans] if full.shape[1] else np.zeros((len(chans), 0))
    marker_stream = full[marker_row] if (marker_row is not None and full.shape[1]) else np.zeros(eeg.shape[1])
    mk_idx = np.flatnonzero(marker_stream != 0)
    stream_markers = [(int(i), float(marker_stream[i])) for i in mk_idx]
    frame_rep = analyse_frame_log(frame_times)
    frame_rep["estimated_refresh_hz"] = refresh
    meta = {"config": asdict(cfg), "channels": _muse_names(len(chans)), "board": board_kind,
            "realized_freqs": {str(f): realized_freq(f, refresh) for f in cfg.freqs},
            "plan": [asdict(p) for p in phases]}
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, eeg=eeg.astype(np.float32), fs=fs, markers=np.array(stream_markers, dtype=float),
                        frame_times=np.array(frame_times), frame_report=json.dumps(frame_rep),
                        meta=json.dumps(meta))
    return {"out": str(out_p), "n_samples": int(eeg.shape[1]), "fs": fs, "stream_markers": len(stream_markers),
            "frame_report": frame_rep, "seconds": round(eeg.shape[1] / fs, 1) if fs else 0}


def _muse_names(n):
    from rover_intent.eeg.clench import MUSE_CHANNELS
    return MUSE_CHANNELS[:n] if n <= 4 else [f"ch{i}" for i in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="record", choices=["record"])
    ap.add_argument("--check-display", action="store_true", help="measure flicker timing only, no board")
    ap.add_argument("--board", choices=["muse2", "muse2-bled", "synthetic"], default="muse2")
    ap.add_argument("--serial", default="")
    ap.add_argument("--mac", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="data/eeg_recordings/session.npz")
    ap.add_argument("--freqs", default="12,15", help="comma-separated flicker frequencies in Hz")
    ap.add_argument("--ssvep-s", type=float, default=5.0)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--alpha-blocks", type=int, default=3)
    ap.add_argument("--block-s", type=float, default=20.0)
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args(argv)
    freqs = tuple(float(x) for x in args.freqs.split(","))
    if args.check_display:
        print(json.dumps(check_display(freqs, windowed=not args.fullscreen), indent=1))
        return 0
    cfg = SessionConfig(freqs=freqs, ssvep_s=args.ssvep_s, reps=args.reps,
                        alpha_blocks=args.alpha_blocks, block_s=args.block_s)
    print(f"plan: ~{cfg.est_minutes():.1f} min, {cfg.ssvep_trials()} SSVEP trials, freqs {freqs}", flush=True)
    rep = run_session(cfg, args.board, args.out, args.serial, args.mac, args.name, windowed=not args.fullscreen)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

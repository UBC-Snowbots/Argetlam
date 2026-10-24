"""One guided Muse 2 session, start to finish, for day one (Sunday) with minimal time on the headband.

ONE command on the laptop records a short labelled session, fits Pranav's own clench + eyes-closed-alpha
calibration, writes it where eeg/clench.py and eeg/muse_detect.py already load it (old one backed up), and runs a
short live test that double-clench -> gripper toggle in the SIM only. A Saturday --preflight runs the whole thing on
BrainFlow's synthetic board with NO Muse, in under 3 minutes, so day one is a known quantity.

What actually gets calibrated (be honest, see docs/muse_readiness.md):
  - CLENCH detector (eeg/clench.py): EMG (jaw muscle), the chosen hands-free control. We refit the per-channel
    relaxed level + the on_ratio threshold on Pranav's real rest / clench / negative recordings, replacing the
    synthetic-calibrated defaults. This is the primary fine-tune target.
  - ALPHA eyes-closed detector (eeg/alpha.py): genuine EEG. We fit the per-user relative-alpha threshold on his
    eyes-open vs eyes-closed blocks. Secondary, dependable but slow.
  - SSVEP (eeg/ssvep.py) and the motor-imagery decoder (eeg/train_mi.py, eeg/hybrid.py) are NOT fit here: SSVEP is
    near chance on the bare Muse (no occipital electrode) and motor imagery needs motor-strip electrodes the Muse
    does not have. --with-ssvep records the optional flicker block via eeg/muse_session for a readiness report only.

Pranav's EEG is personal biometric data: everything is written to local disk under data/eeg_sessions/ (gitignored),
never uploaded, never committed.

Usage (laptop with the Muse; Bluetooth on that machine):
  python -m rover_intent.eeg.sunday --preflight                 # Saturday, NO Muse, synthetic board, < 3 min
  python -m rover_intent.eeg.sunday --board muse2 --name pranav  # Sunday, the real session
  python -m rover_intent.eeg.sunday --board muse2 --name pranav --only clench_double   # re-record one block
  python -m rover_intent.eeg.sunday --board muse2 --name pranav --only train           # refit from saved blocks
  python -m rover_intent.eeg.sunday --board muse2 --name pranav --only live --host <vm> # live test to the sim
  --board muse2 (native BLE) | muse2-bled (BLED112 dongle, --serial COM3) | synthetic (dry run)
  Live test goes to a SPARE sim port (default 47195), never 47100 and never the real arm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rover_intent.eeg.alpha import AlphaConfig, AlphaDetector, rel_alpha
from rover_intent.eeg.clench import (
    MUSE_CHANNELS,
    ClenchConfig,
    ClenchDetector,
    _read,
    _record,
    calibrate,
    open_board,
    toggle_packet,
)

PROTOCOL_VERSION = "1.0"
SPARE_SIM_PORT = 47195  # live test goes here, NEVER 47100 (that is the real control path) and never the real arm


# --- protocol (pure, testable) --------------------------------------------------------------------------------------
@dataclass
class Block:
    name: str              # unique stage id
    label: str             # training category: rest | clench | negative | eyes_open | eyes_closed
    kind: str              # emg | alpha
    seconds: float         # per-rep seconds (reps>1) or total seconds (reps==1)
    cue: str               # what to tell the operator
    reps: int = 1          # >1: prompt the action this many times, each `seconds` long, concatenated
    hint: str = ""         # fix-it hint shown before the block


@dataclass
class SundayConfig:
    name: str = "pranav"
    line_hz: float = 60.0            # mains: 60 Canada/US, 50 elsewhere
    rest_s: float = 30.0
    clench_single_reps: int = 8
    clench_single_s: float = 3.0
    clench_double_reps: int = 10
    clench_double_s: float = 4.0
    neg_chew_s: float = 30.0
    neg_blink_s: float = 20.0
    neg_head_s: float = 20.0
    alpha_block_s: float = 60.0
    # signal quality gate (heuristic, per-channel; tune on the day):
    q_rms_lo_uv: float = 2.0         # below -> flat / not on the skin
    q_rms_hi_uv: float = 120.0       # above -> drifting / huge movement / bad contact
    q_line_ratio_max: float = 0.6    # 60 Hz power / broadband; above -> mains pickup, press the band down
    q_check_s: float = 6.0

    def est_minutes(self) -> float:
        rec = (self.rest_s
               + self.clench_single_reps * (self.clench_single_s + 1.5)
               + self.clench_double_reps * (self.clench_double_s + 1.5)
               + self.neg_chew_s + self.neg_blink_s + self.neg_head_s
               + 2 * self.alpha_block_s)
        overhead = self.q_check_s + 60 + 90 + 60  # signal check, instructions, train, live test
        return round((rec + overhead) / 60.0, 1)


def build_protocol(cfg: SundayConfig) -> list[Block]:
    """Ordered recording blocks. Negatives (false-trigger data) come right after the clenches on purpose: they are
    the most important data for keeping the gripper from toggling while Pranav talks, chews or looks around."""
    return [
        Block("rest", "rest", "emg", cfg.rest_s,
              "RELAX: jaw loose, eyes open, blink normally, sit still.",
              hint="This sets the relaxed baseline every threshold is measured against."),
        Block("clench_single", "clench", "emg", cfg.clench_single_s,
              "CLENCH your jaw firmly for about half a second, then relax.",
              reps=cfg.clench_single_reps,
              hint="One firm bite each time the counter shows. Not a long squeeze."),
        Block("clench_double", "clench", "emg", cfg.clench_double_s,
              "DOUBLE CLENCH: two quick bites in a row (like a double click), then relax.",
              reps=cfg.clench_double_reps,
              hint="This is the real control gesture. Two fast bites, then rest a second."),
        Block("neg_chew", "negative", "emg", cfg.neg_chew_s,
              "TALK and CHEW normally (read aloud, chew gum). Do NOT double-clench.",
              hint="False-trigger data: the detector must ignore this."),
        Block("neg_blink", "negative", "emg", cfg.neg_blink_s,
              "BLINK and move your eyes around. No clenching.",
              hint="Blinks hit the forehead electrodes; they must not toggle the gripper."),
        Block("neg_head", "negative", "emg", cfg.neg_head_s,
              "TURN and TILT your head, look left/right/up/down. No clenching.",
              hint="Head turns move the band; must not toggle."),
        Block("eyes_open", "eyes_open", "alpha", cfg.alpha_block_s,
              "EYES OPEN, look softly at a fixed point. Relax.",
              hint="Alpha baseline (eyes open)."),
        Block("eyes_closed", "eyes_closed", "alpha", cfg.alpha_block_s,
              "EYES CLOSED, relax (I will tell you when to open).",
              hint="Alpha rises when the eyes close (Berger effect)."),
    ]


RECORDING_BLOCKS = [b.name for b in build_protocol(SundayConfig())]
STAGES = RECORDING_BLOCKS + ["train", "live"]


# --- resumable storage ----------------------------------------------------------------------------------------------
def session_dir_for(name: str, root: str = "data/eeg_sessions", date: str | None = None) -> Path:
    date = date or time.strftime("%Y%m%d")
    return Path(root) / f"{name}_{date}"


def load_manifest(session_dir: Path) -> dict:
    p = session_dir / "session.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"protocol_version": PROTOCOL_VERSION, "blocks": {}}


def write_manifest(session_dir: Path, manifest: dict) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(json.dumps(manifest, indent=1))


def save_block(session_dir: Path, block: Block, eeg: np.ndarray, fs: float, channels: list[str],
               quality: dict, board: str, manifest: dict) -> dict:
    """Write one block's raw EEG (npz, the repo's existing format, MNE-loadable) + update the manifest. Called right
    after each block so a crash or a bad block never loses the earlier ones."""
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{block.name}.npz"
    np.savez_compressed(path, eeg=eeg.astype(np.float32), fs=fs, channels=np.array(channels),
                        label=block.label, kind=block.kind)
    manifest.setdefault("blocks", {})[block.name] = {
        "done": True, "label": block.label, "kind": block.kind, "seconds": round(eeg.shape[1] / fs, 1),
        "samples": int(eeg.shape[1]), "quality": quality, "path": str(path.relative_to(session_dir)),
        "t_saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    manifest["board"], manifest["fs"], manifest["channels"] = board, fs, channels
    write_manifest(session_dir, manifest)
    return manifest["blocks"][block.name]


def load_block(session_dir: Path, name: str) -> tuple[np.ndarray, float, list[str]]:
    rec = np.load(session_dir / "raw" / f"{name}.npz", allow_pickle=True)
    return np.asarray(rec["eeg"], float), float(rec["fs"]), [str(c) for c in rec["channels"]]


def done_blocks(manifest: dict) -> set[str]:
    return {k for k, v in manifest.get("blocks", {}).items() if v.get("done")}


# --- signal quality -------------------------------------------------------------------------------------------------
def channel_quality(eeg: np.ndarray, fs: float, cfg: SundayConfig) -> dict:
    """Per-channel contact check from a short relaxed capture. Heuristic amplitudes (dry Muse electrodes); the
    thresholds live in SundayConfig so they can be tuned on the day. Returns per-channel status + overall verdict."""
    eeg = np.asarray(eeg, float)
    n_ch, n = eeg.shape
    channels = (MUSE_CHANNELS[:n_ch] if n_ch <= 4 else [f"ch{i}" for i in range(n_ch)])
    out = {"channels": {}, "ok": True, "bad": [], "hints": []}
    if n < int(0.5 * fs):
        out["ok"], out["hints"] = False, ["capture too short to judge contact"]
        return out
    x = eeg - eeg.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(np.fft.rfft(x * np.hanning(n), axis=1)) ** 2
    broad = (freqs >= 2) & (freqs <= 40)
    line = (freqs >= cfg.line_hz - 1) & (freqs <= cfg.line_hz + 1)
    for i, ch in enumerate(channels):
        rms = float(np.std(x[i]))
        line_ratio = float(P[i, line].sum() / max(P[i, broad].sum(), 1e-12))
        flat = rms < cfg.q_rms_lo_uv
        drift = rms > cfg.q_rms_hi_uv
        noisy = line_ratio > cfg.q_line_ratio_max
        status = "good" if not (flat or drift or noisy) else ("flat" if flat else "drift" if drift else "mains")
        out["channels"][ch] = {"rms_uv": round(rms, 1), "line_ratio": round(line_ratio, 2), "status": status}
        if status != "good":
            out["ok"] = False
            out["bad"].append(ch)
    if out["bad"]:
        out["hints"] = [
            "Wet the electrodes (a dab of water behind the ears on TP9/TP10 and on the forehead for AF7/AF8).",
            "Push the band down so the sensors sit on skin, not hair; part the hair under TP9/TP10.",
            "Sit still for the check; a drifting channel usually settles after ~10 s of contact.",
            f"Bad channels: {', '.join(out['bad'])}.",
        ]
    return out


# --- training / calibration (pure, reuses clench.calibrate + AlphaDetector.calibrate) -------------------------------
def _count_toggles(detector_profile: dict, x: np.ndarray, chunk: int = 32) -> int:
    d = ClenchDetector.from_profile(detector_profile)
    return sum(len(d.push(x[:, i:i + chunk])) for i in range(0, x.shape[1], chunk))


def fit_clench(rest: np.ndarray, single: np.ndarray, double: np.ndarray, neg: np.ndarray, fs: float,
               expected_doubles: int) -> tuple[dict, dict]:
    """Deployed profile = clench.calibrate on ALL the data. Then an honest held-out check: refit on a train split
    and measure detection rate on held-out double-clenches + false triggers/min on held-out negatives, and compare
    against the synthetic-calibrated default threshold on the same held-out data."""
    clench_all = np.concatenate([single, double], axis=1) if single.shape[1] else double
    profile = calibrate(rest, clench_all, neg if neg.shape[1] else None, fs)

    # held-out split of the double-clench block and the negatives
    dh = double.shape[1] // 2
    nh = neg.shape[1] // 2
    train_doubles, test_doubles = double[:, :dh], double[:, dh:]
    train_neg, test_neg = neg[:, :nh], neg[:, nh:]
    test_double_count = max(1, expected_doubles - expected_doubles // 2)
    train_clench = np.concatenate([single, train_doubles], axis=1) if single.shape[1] else train_doubles
    train_profile = calibrate(rest, train_clench, train_neg if train_neg.shape[1] else None, fs)

    # default (synthetic) comparison: same relaxed level, but the library default threshold, no per-user fit
    default_profile = {**train_profile, "config": {**train_profile["config"], "on_ratio": ClenchConfig().on_ratio}}

    def fp_per_min(prof, seg):
        secs = seg.shape[1] / fs
        return round(_count_toggles(prof, seg) / (secs / 60.0), 2) if secs > 0 else 0.0

    detected = _count_toggles(train_profile, test_doubles)
    detected_default = _count_toggles(default_profile, test_doubles)
    metrics = {
        "deployed_on_ratio": profile["config"]["on_ratio"],
        "deployed_quality": profile["stats"]["quality"],
        "deployed_margin": profile["stats"]["margin"],
        "heldout_double_clenches": test_double_count,
        "calibrated": {
            "on_ratio": train_profile["config"]["on_ratio"],
            "detected": int(min(detected, test_double_count)),
            "detection_rate": round(min(detected, test_double_count) / test_double_count, 2),
            "raw_toggle_count": int(detected),
            "false_per_min_negatives": fp_per_min(train_profile, test_neg),
        },
        "synthetic_default": {
            "on_ratio": ClenchConfig().on_ratio,
            "detected": int(min(detected_default, test_double_count)),
            "detection_rate": round(min(detected_default, test_double_count) / test_double_count, 2),
            "raw_toggle_count": int(detected_default),
            "false_per_min_negatives": fp_per_min(default_profile, test_neg),
        },
    }
    return profile, metrics


def _slice_windows(x: np.ndarray, fs: float, win_s: float) -> list[np.ndarray]:
    n = int(win_s * fs)
    return [x[:, i:i + n] for i in range(0, x.shape[1] - n + 1, n)] if n > 0 else []


def fit_alpha(eyes_open: np.ndarray, eyes_closed: np.ndarray, fs: float, win_s: float = 2.0) -> tuple[float, dict]:
    """Fit the per-user eyes-closed relative-alpha threshold (AlphaDetector.calibrate) on a TRAIN split of the
    windows, report balanced accuracy on a held-out split, and compare to a fixed cold threshold (no calibration)."""
    ow = _slice_windows(eyes_open, fs, win_s)
    cw = _slice_windows(eyes_closed, fs, win_s)
    if len(ow) < 2 or len(cw) < 2:
        return 0.0, {"ok": False, "reason": "not enough alpha windows"}
    oh, ch = max(1, int(len(ow) * 0.6)), max(1, int(len(cw) * 0.6))
    det, st = AlphaDetector.calibrate(ow[:oh], cw[:ch], fs, AlphaConfig())
    mo = np.array([rel_alpha(w, fs).mean() for w in ow[oh:]])
    mc = np.array([rel_alpha(w, fs).mean() for w in cw[ch:]])
    bal = 0.5 * (float((mo < det.threshold).mean()) + float((mc >= det.threshold).mean())) if len(mo) and len(mc) else None
    # cold / no-calibration baseline: a user-independent threshold of rel-alpha = 1.0 is meaningless; instead use the
    # midpoint of the GLOBAL open/closed means as the naive guess an un-tuned detector would use.
    all_o = np.array([rel_alpha(w, fs).mean() for w in ow])
    all_c = np.array([rel_alpha(w, fs).mean() for w in cw])
    cold_thr = float((all_o.mean() + all_c.mean()) / 2)
    cold_bal = 0.5 * (float((mo < cold_thr).mean()) + float((mc >= cold_thr).mean())) if len(mo) and len(mc) else None
    ratio = float(all_c.mean() / max(all_o.mean(), 1e-9))
    metrics = {"ok": True, "threshold": round(det.threshold, 4), "heldout_bal_acc": round(bal, 3) if bal else None,
               "cold_bal_acc": round(cold_bal, 3) if cold_bal else None,
               "closed_over_open_ratio": round(ratio, 2), "train_separation": round(st["separation"], 2),
               "n_windows": {"open": len(ow), "closed": len(cw)}, "win_s": win_s}
    return det.threshold, metrics


# --- write the calibration where the app / tools load it (step e), backing up the old one ---------------------------
def _backup_and_write(path: Path, payload: dict) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        backup = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}")
        path.rename(backup)
    path.write_text(json.dumps(payload, indent=1))
    return str(backup) if backup else None


def write_profiles(clench_profile: dict, alpha_threshold: float, alpha_metrics: dict, name: str,
                   profiles_dir: str = "data/profiles") -> dict:
    """Clench profile -> data/profiles/clench_<name>.json (what eeg/clench.py run/replay loads with --profile).
    Alpha threshold + optional slot -> data/profiles/muse_<name>.json (what eeg/muse_detect.py loads). Old files are
    renamed to *.bak.<timestamp>, never overwritten."""
    pdir = Path(profiles_dir)
    clench_path = pdir / f"clench_{name}.json"
    muse_path = pdir / f"muse_{name}.json"
    clench_backup = _backup_and_write(clench_path, clench_profile)
    muse_payload = {"alpha_threshold": alpha_threshold, "alpha_metrics": alpha_metrics,
                    "fs": clench_profile.get("fs"), "channels": clench_profile.get("channels"),
                    "source": "eeg/sunday.py", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    muse_backup = _backup_and_write(muse_path, muse_payload)
    return {"clench_profile": str(clench_path), "clench_backup": clench_backup,
            "muse_profile": str(muse_path), "muse_backup": muse_backup}


# --- terminal prompts + progress bar --------------------------------------------------------------------------------
def _say(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def _bar(frac: float, width: int = 36) -> str:
    frac = max(0.0, min(1.0, frac))
    return "#" * int(width * frac) + "-" * (width - int(width * frac))


def _countdown_prompt(label: str):
    def prompt(remaining: float, total: float):
        done = 1.0 - remaining / total if total else 1.0
        print(f"\r  {label:<22} |{_bar(done)}| {remaining:4.1f}s ", end="", flush=True)
    return prompt


# --- board capture of one block -------------------------------------------------------------------------------------
def record_block(board, chans, block: Block, quiet: bool = False) -> np.ndarray:
    """Capture one block from the board. reps>1 prompts the action that many times and concatenates."""
    if block.reps > 1:
        parts = []
        for i in range(block.reps):
            if not quiet:
                print(f"  [{i + 1}/{block.reps}] {block.name.replace('_', ' ').upper()}", flush=True)
            prompt = None if quiet else (lambda rem, tot=block.seconds, lb=f"rep {i + 1}/{block.reps}":
                                         _countdown_prompt(lb)(rem, tot))
            parts.append(_record(board, chans, block.seconds, prompt))
            if not quiet:
                print()
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(chans), 0))
    prompt = None if quiet else (lambda rem, tot=block.seconds: _countdown_prompt(block.name)(rem, tot))
    x = _record(board, chans, block.seconds, prompt)
    if not quiet:
        print()
    return x


# --- live test (step f): double clench -> GripToggle -> SIM only ----------------------------------------------------
def replay_toggles(profile: dict, x: np.ndarray, chunk: int = 32) -> list[float]:
    d = ClenchDetector.from_profile(profile)
    return [t for i in range(0, x.shape[1], chunk) for t in d.push(x[:, i:i + chunk])]


def live_test(profile: dict, host: str, port: int, board=None, chans=None, replay_x: np.ndarray | None = None,
              seconds: float = 20.0, dry: bool = False, send_follow: bool = True) -> dict:
    """Send GripToggle to the SIM at host:port (default a SPARE port, never 47100, never the real arm). With a live
    board it streams for `seconds`; otherwise it replays `replay_x` (recorded double clenches) through the detector.
    Set dry=True to print toggles without sending."""
    import socket

    assert port != 47100, "live test must use a spare SIM port, not the real control port 47100"
    tx = None if dry else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    if send_follow and tx is not None:
        # put the arbiter in FOLLOW so the toggle actually moves the sim gripper (offline keyword parser, no cloud)
        tx.sendto(json.dumps({"kind": "utterance", "t": time.time(), "text": "follow me", "offline": True}).encode(),
                  (host, port))
        time.sleep(0.5)
    if replay_x is not None:
        for _t in replay_toggles(profile, replay_x):
            sent += 1
            if tx is not None:
                tx.sendto(toggle_packet(time.time()), (host, port))
            print(f"  TOGGLE #{sent}" + (" (sent to sim)" if tx is not None else " (dry)"), flush=True)
            time.sleep(0.3)
    elif board is not None:
        det = ClenchDetector.from_profile(profile, t0=time.time())
        print(f"  Double-clench to toggle the sim gripper ({seconds:.0f}s). Watch the sim window.", flush=True)
        t_end = time.time() + seconds
        while time.time() < t_end:
            x = _read(board, chans)
            if x.shape[1]:
                for _t in det.push(x):
                    sent += 1
                    if tx is not None:
                        tx.sendto(toggle_packet(time.time()), (host, port))
                    print(f"\n  TOGGLE #{sent}" + (" (sent to sim)" if tx is not None else " (dry)"), flush=True)
            time.sleep(0.02)
    return {"toggles": sent, "host": host, "port": port, "dry": dry}


# --- orchestration --------------------------------------------------------------------------------------------------
def _train_and_report(session_dir: Path, scfg: SundayConfig, manifest: dict) -> dict:
    """Fit clench + alpha from the saved blocks, print plain-English metrics, write the profiles (old ones backed up)."""
    fs = float(manifest["fs"])
    rest, _, _ = load_block(session_dir, "rest")
    single, _, _ = load_block(session_dir, "clench_single")
    double, _, _ = load_block(session_dir, "clench_double")
    neg_parts = [load_block(session_dir, n)[0] for n in ("neg_chew", "neg_blink", "neg_head")
                 if (session_dir / "raw" / f"{n}.npz").exists()]
    neg = np.concatenate(neg_parts, axis=1) if neg_parts else np.zeros((rest.shape[0], 0))
    clench_profile, cm = fit_clench(rest, single, double, neg, fs, scfg.clench_double_reps)

    eo, _, _ = load_block(session_dir, "eyes_open")
    ec, _, _ = load_block(session_dir, "eyes_closed")
    alpha_thr, am = fit_alpha(eo, ec, fs)

    paths = write_profiles(clench_profile, alpha_thr, am, scfg.name)
    report = {"clench": cm, "alpha": am, "profiles": paths}
    (session_dir / "report.json").write_text(json.dumps(report, indent=1))

    # plain-English summary
    _say("Results (plain English)")
    c = cm["calibrated"]
    d = cm["synthetic_default"]
    print(f"  Double-clench detection (held-out): caught {c['detected']}/{cm['heldout_double_clenches']} "
          f"({int(100 * c['detection_rate'])}%).")
    print(f"  False triggers while talking/chewing/blinking/head-turning: {c['false_per_min_negatives']} per minute.")
    print(f"  Your threshold: {c['on_ratio']}x relaxed  (synthetic default was {d['on_ratio']}x).")
    print(f"  Vs the synthetic default on the same data: detection {int(100 * d['detection_rate'])}%, "
          f"false {d['false_per_min_negatives']}/min.")
    print(f"  Deployed profile quality: {cm['deployed_quality'].upper()} (margin {cm['deployed_margin']}x).")
    if am.get("ok"):
        print(f"  Eyes-closed vs eyes-open accuracy (held-out): {int(100 * (am['heldout_bal_acc'] or 0))}% "
              f"(chance 50%); alpha rises {am['closed_over_open_ratio']}x when you close your eyes.")
        if am.get("cold_bal_acc") is not None:
            print(f"    (a no-calibration threshold would get {int(100 * am['cold_bal_acc'])}% on the same data.)")
    print(f"\n  Clench profile written: {paths['clench_profile']}" +
          (f"  (old -> {paths['clench_backup']})" if paths["clench_backup"] else "  (no previous profile)"))
    print(f"  Alpha profile written:  {paths['muse_profile']}" +
          (f"  (old -> {paths['muse_backup']})" if paths["muse_backup"] else "  (no previous profile)"))
    manifest["report"] = report
    write_manifest(session_dir, manifest)
    return report


def run_session(scfg: SundayConfig, board_kind: str, host: str = "127.0.0.1", port: int = SPARE_SIM_PORT,
                only: str | None = None, resume: bool = True, session_dir: Path | None = None,
                quiet: bool | None = None, live_seconds: float = 20.0, strict_quality: bool | None = None,
                date: str | None = None) -> dict:
    """The whole day-one session. Resumable: each block is saved as it finishes; re-run with the same --name to
    continue, or --only <block> to redo one. `only` can be a recording block, 'train' or 'live'."""
    session_dir = session_dir or session_dir_for(scfg.name, date=date)
    quiet = (board_kind == "synthetic") if quiet is None else quiet
    strict_quality = (board_kind != "synthetic") if strict_quality is None else strict_quality
    protocol = build_protocol(scfg)
    manifest = load_manifest(session_dir)
    manifest["line_hz"] = scfg.line_hz
    manifest["sunday_config"] = asdict(scfg)
    manifest["created"] = manifest.get("created") or time.strftime("%Y-%m-%d %H:%M:%S")
    write_manifest(session_dir, manifest)

    stages = [only] if only else STAGES
    recording_stages = [b for b in protocol if b.name in stages]

    board = fs = chans = None
    if recording_stages or (not only and board_kind != "synthetic"):
        _say(f"Day-one Muse session for '{scfg.name}'. Board: {board_kind}. Plan ~{scfg.est_minutes():.0f} min.")
        _say("Connecting to the Muse. Make sure it is on and Bluetooth is enabled on this laptop.")
        board, fs, chans = open_board(board_kind, scfg.serial, scfg.mac, scfg.name_target)
        manifest["fs"], manifest["channels"] = fs, (MUSE_CHANNELS[:len(chans)] if len(chans) <= 4
                                                     else [f"ch{i}" for i in range(len(chans))])
        _read(board, chans)  # drop startup junk

        # (a) signal quality gate, unless resuming past it
        try:
            _say(f"Signal check: sit still, relax, for {scfg.q_check_s:.0f} s.")
            time.sleep(1.0 if not quiet else 0)
            qcap = _record(board, chans, scfg.q_check_s,
                           None if quiet else (lambda rem, tot=scfg.q_check_s: _countdown_prompt("signal check")(rem, tot)))
            if not quiet:
                print()
            quality = channel_quality(qcap, fs, scfg)
            for ch, d in quality["channels"].items():
                print(f"    {ch}: {d['status']:<5} (rms {d['rms_uv']} uV, 60Hz {d['line_ratio']})", flush=True)
            manifest["signal_check"] = quality
            write_manifest(session_dir, manifest)
            if not quality["ok"]:
                for h in quality["hints"]:
                    print(f"    - {h}", flush=True)
                if strict_quality:
                    _say("Signal is not good enough yet. Fix the contact above and re-run; refusing to record.")
                    board.stop_stream(); board.release_session()
                    return {"ok": False, "stage": "signal_check", "quality": quality,
                            "session_dir": str(session_dir)}
            else:
                _say("Signal looks good on all channels.")

            # (b)+(c) record each block (skip the ones already done when resuming)
            done = done_blocks(manifest) if (resume and not only) else set()
            for block in recording_stages:
                if block.name in done:
                    print(f"  (already recorded: {block.name}, skipping. --only {block.name} to redo.)", flush=True)
                    continue
                _say(f"{block.cue}")
                if block.hint and not quiet:
                    print(f"    [{block.hint}]", flush=True)
                    time.sleep(2.0)
                x = record_block(board, chans, block, quiet=quiet)
                bq = channel_quality(x, fs, scfg) if block.label in ("rest", "eyes_open") else {"skipped": True}
                save_block(session_dir, block, x, fs, manifest["channels"], bq, board_kind, manifest)
                print(f"    saved {block.name}: {x.shape[1]} samples ({x.shape[1] / fs:.1f}s)", flush=True)
        finally:
            board.stop_stream(); board.release_session()

    # (d)+(e) train + write profiles
    report = None
    if (not only) or only == "train":
        if done_blocks(manifest) >= set(RECORDING_BLOCKS):
            report = _train_and_report(session_dir, scfg, manifest)
        elif only == "train":
            missing = set(RECORDING_BLOCKS) - done_blocks(manifest)
            _say(f"Cannot train: missing blocks {sorted(missing)}. Record them first.")
            return {"ok": False, "stage": "train", "missing": sorted(missing), "session_dir": str(session_dir)}

    # (f) live test
    live = None
    if (not only) or only == "live":
        prof_path = Path("data/profiles") / f"clench_{scfg.name}.json"
        if prof_path.exists():
            profile = json.loads(prof_path.read_text())
            _say(f"Live test: double-clench toggles the SIM gripper (host {host}:{port}). SIM only, never the arm.")
            if board_kind == "synthetic":
                # dry run: replay the recorded double-clench block through the detector, nothing sent on the wire
                dpath = session_dir / "raw" / "clench_double.npz"
                replay_x = load_block(session_dir, "clench_double")[0] if dpath.exists() else None
                live = live_test(profile, host, port, replay_x=replay_x, dry=True)
            else:
                lboard, _, lchans = open_board(board_kind, scfg.serial, scfg.mac, scfg.name_target)
                try:
                    _read(lboard, lchans)
                    live = live_test(profile, host, port, board=lboard, chans=lchans, seconds=live_seconds)
                finally:
                    lboard.stop_stream(); lboard.release_session()
            manifest["live_test"] = live
            write_manifest(session_dir, manifest)
        elif only == "live":
            _say(f"No clench profile at {prof_path}; run training first.")

    return {"ok": True, "session_dir": str(session_dir), "report": report, "live": live,
            "done_blocks": sorted(done_blocks(manifest))}


# --- Saturday preflight (no Muse) -----------------------------------------------------------------------------------
def preflight(session_dir: Path | None = None) -> dict:
    """Saturday check on the laptop with NO Muse: Python, deps, BrainFlow import, Bluetooth presence (Windows if
    detectable), then the WHOLE session end to end on BrainFlow's synthetic board, in under 3 minutes."""
    t0 = time.time()
    out = {"checks": {}, "ok": True}

    def check(name, ok, detail="", fatal=True):
        out["checks"][name] = {"ok": bool(ok), "detail": detail, "fatal": fatal}
        if fatal:
            out["ok"] = out["ok"] and bool(ok)
        tag = "OK " if ok else ("FAIL" if fatal else "WARN")
        print(f"  [{tag}] {name}: {detail}", flush=True)

    _say("Saturday preflight (no Muse needed).")
    check("python>=3.10", sys.version_info >= (3, 10), f"{sys.version.split()[0]}")
    for mod in ("numpy", "brainflow"):  # the Sunday session needs only these (+ pygame for --with-ssvep)
        try:
            __import__(mod)
            check(f"import {mod}", True, "present")
        except Exception as e:  # noqa: BLE001
            check(f"import {mod}", False, f"MISSING ({e}); run the pip install below")
    # optional: Windows Smart App Control can block SciPy's DLLs. scipy.signal loads lazily, so "import scipy.signal"
    # can succeed while welch/sosfilt fail; report what the session code actually ended up using.
    from rover_intent.eeg import alpha as _alpha, clench as _clench
    impl = {_alpha.welch.__module__.split(".")[0], _clench.sosfilt.__module__.split(".")[0]}
    if impl == {"scipy"}:
        check("signal filters", True, "SciPy")
    else:
        check("signal filters", False, "SciPy blocked or missing: using the numpy-only filters (eeg/np_signal.py)",
              fatal=False)
    try:                                          # sklearn is only for the offline MI studies, not the session
        import sklearn  # noqa: F401
        check("import sklearn (offline studies only)", True, "present", fatal=False)
    except Exception:  # noqa: BLE001
        check("import sklearn (offline studies only)", False, "missing; not needed for the Sunday session",
              fatal=False)
    try:
        import pygame  # noqa: F401
        check("import pygame (only needed for --with-ssvep)", True, "present", fatal=False)
    except Exception:  # noqa: BLE001
        check("import pygame (only needed for --with-ssvep)", False, "missing; optional unless --with-ssvep",
              fatal=False)

    # Bluetooth presence (best effort; only meaningful on the real laptop, which is Windows)
    bt = _bluetooth_present()
    check("bluetooth adapter", bt is not False, bt if isinstance(bt, str) else
          ("detected" if bt else "could not detect here (expected on the VM; check on the Windows laptop)"),
          fatal=False)

    # BrainFlow synthetic board usable
    try:
        from brainflow.board_shim import BoardIds, BoardShim
        fs = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD)
        check("brainflow synthetic board", True, f"fs={fs} Hz")
    except Exception as e:  # noqa: BLE001
        check("brainflow synthetic board", False, str(e))

    # full session end to end on the synthetic board (short), NO Muse
    sd = session_dir or session_dir_for("preflight", root="data/eeg_sessions", date="dryrun")
    import shutil
    if sd.exists():
        shutil.rmtree(sd)
    scfg = SundayConfig(name="preflight", rest_s=3, clench_single_reps=2, clench_single_s=1.0,
                        clench_double_reps=4, clench_double_s=1.0, neg_chew_s=3, neg_blink_s=2, neg_head_s=2,
                        alpha_block_s=8, q_check_s=2)
    scfg.serial = scfg.mac = scfg.name_target = ""
    try:
        res = run_session(scfg, "synthetic", host="127.0.0.1", port=SPARE_SIM_PORT, session_dir=sd,
                          quiet=True, strict_quality=False, date="dryrun")
        ran = res.get("ok") and (sd / "report.json").exists()
        check("full dry run (synthetic board, end to end)", ran,
              f"session={res.get('session_dir')}, toggles={(res.get('live') or {}).get('toggles')}")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("full dry run (synthetic board, end to end)", False, str(e))

    elapsed = time.time() - t0
    check("finished under 3 minutes", elapsed < 180, f"{elapsed:.1f}s")
    out["seconds"] = round(elapsed, 1)
    _say("PREFLIGHT PASS - ready for Sunday." if out["ok"] else "PREFLIGHT FOUND PROBLEMS - fix the FAIL lines above.")
    if not out["checks"].get("import brainflow", {}).get("ok", True):
        print("\n  Install everything with:\n    pip install -e '.[muse]'\n  "
              "(or: pip install brainflow scipy numpy scikit-learn pygame)", flush=True)
    return out


def _bluetooth_present():
    """Best-effort Bluetooth adapter detection. Returns True/False/str. Only meaningful on the real Windows laptop."""
    import platform
    import subprocess
    try:
        if platform.system() == "Windows":
            # the radio itself sits on USB or PCI; paired devices' profiles (BTHENUM\..., e.g. a headset's
            # "Avrcp Transport") are also in the Bluetooth class and must not count as the adapter
            ps = ("$d = Get-PnpDevice -Class Bluetooth -Status OK; "
                  "$a = $d | Where-Object { $_.InstanceId -match '^(USB|PCI)\\\\' } | Select-Object -First 1; "
                  "if ($a) { $a.FriendlyName } elseif ($d) { 'NOADAPTER' }")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=15)
            name = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
            if name == "NOADAPTER":
                return False
            return name or False
        if platform.system() == "Linux":
            r = subprocess.run(["hciconfig"], capture_output=True, text=True, timeout=5)
            return "hci" in r.stdout
        if platform.system() == "Darwin":
            return "Bluetooth" in subprocess.run(["system_profiler", "SPBluetoothDataType"],
                                                 capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        return False
    return False


# --- optional SSVEP readiness block (gated on the AUX occipital electrode) -------------------------------------------
def ssvep_block(scfg: SundayConfig, board_kind: str, session_dir: Path, freqs=(12.0, 15.0)) -> dict:
    """Record the flicker block with eeg/muse_session and score it with eeg/muse_report. SSVEP is only usable with an
    occipital electrode on the AUX port (see docs/muse_readiness.md); this just measures whether it is in play."""
    from rover_intent.eeg import muse_report as MR
    from rover_intent.eeg import muse_session as MS
    out = session_dir / "ssvep_session.npz"
    cfg = MS.SessionConfig(freqs=tuple(freqs), reps=6, alpha_blocks=1, block_s=20.0)
    MS.run_session(cfg, board_kind, str(out), scfg.serial, scfg.mac, scfg.name_target, windowed=True)
    rec = np.load(out, allow_pickle=True)
    rep = MR.build_report(rec["eeg"], float(rec["fs"]), rec["markers"], json.loads(str(rec["meta"])),
                          json.loads(str(rec["frame_report"])) if "frame_report" in rec.files else {})
    (session_dir / "ssvep_report.json").write_text(json.dumps(rep, indent=1))
    print(f"  SSVEP verdict: {rep['verdict']}", flush=True)
    return rep


# --- CLI ------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true", help="Saturday: no Muse, synthetic end-to-end, < 3 min")
    ap.add_argument("--board", choices=["muse2", "muse2-bled", "synthetic"], default="muse2")
    ap.add_argument("--serial", default="", help="BLED112 dongle port for muse2-bled (COM3, /dev/ttyACM0)")
    ap.add_argument("--mac", default="", help="Muse MAC address (optional)")
    ap.add_argument("--muse-name", default="", help="specific Muse, e.g. Muse-1A2B")
    ap.add_argument("--name", default="pranav", help="subject name; profiles go to data/profiles/clench_<name>.json")
    ap.add_argument("--line-hz", type=float, default=60.0, help="mains frequency (60 NA, 50 EU)")
    ap.add_argument("--host", default="127.0.0.1", help="SIM host for the live test")
    ap.add_argument("--port", type=int, default=SPARE_SIM_PORT, help=f"SIM port for the live test (default {SPARE_SIM_PORT}; never 47100)")
    ap.add_argument("--only", choices=STAGES, default=None, help="run/redo a single stage")
    ap.add_argument("--no-resume", action="store_true", help="re-record even blocks already saved")
    ap.add_argument("--with-ssvep", action="store_true", help="also record the optional SSVEP flicker block (AUX electrode)")
    ap.add_argument("--freqs", default="12,15", help="SSVEP flicker frequencies for --with-ssvep")
    args = ap.parse_args(argv)

    if args.port == 47100:
        print("refusing: 47100 is the real control port; the live test must use a spare SIM port.", file=sys.stderr)
        return 2
    if args.preflight:
        return 0 if preflight()["ok"] else 1

    scfg = SundayConfig(name=args.name, line_hz=args.line_hz)
    scfg.serial, scfg.mac, scfg.name_target = args.serial, args.mac, args.muse_name
    res = run_session(scfg, args.board, host=args.host, port=args.port, only=args.only,
                      resume=not args.no_resume)
    if args.with_ssvep and not args.only:
        ssvep_block(scfg, args.board, Path(res["session_dir"]), tuple(float(x) for x in args.freqs.split(",")))
    return 0 if res.get("ok") else 1


# runtime attributes attached to SundayConfig that are not part of the dataclass defaults
SundayConfig.serial = ""
SundayConfig.mac = ""
SundayConfig.name_target = ""


if __name__ == "__main__":
    # allow `python src/rover_intent/eeg/sunday.py ...` as well as `python -m rover_intent.eeg.sunday`
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())

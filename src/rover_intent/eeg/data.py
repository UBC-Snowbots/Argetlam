"""Motor-imagery data: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB, 109 subjects, 64 ch, 160 Hz).

Binary task for the demo: *imagined hand clench* (T1/T2 in imagery runs 4, 8, 12) vs *rest* (T0).
We restrict to 8 motor-strip channels so the model matches what an 8-channel OpenBCI Cyton over
the motor cortex can see. Subjects are loaded one at a time (the box has ~17 GB RAM).
"""
from __future__ import annotations

import numpy as np

# 10-10 names as they appear in EEGMMIDB after eegbci.standardize()
MOTOR_8 = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Pz"]
IMAGERY_RUNS = [4, 8, 12]
EPOCH = (0.5, 2.5)   # s after cue; skip the first 0.5 s (visual evoked response)
BAND = (8.0, 30.0)   # mu + beta


def load_subject(subject: int, channels=MOTOR_8, runs=IMAGERY_RUNS, sfreq: float = 160.0):
    """Returns X (n_epochs, n_ch, n_times) float32 and y (1 = imagined clench, 0 = rest)."""
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("WARNING")
    files = eegbci.load_data(subject, runs, update_path=True)  # non-interactive (tsp jobs have no stdin)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)
    raw.pick(channels)
    raw.filter(*BAND, fir_design="firwin")
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq)
    events, ev_id = mne.events_from_annotations(raw, event_id={"T0": 0, "T1": 1, "T2": 2})
    ep = mne.Epochs(raw, events, ev_id, tmin=EPOCH[0], tmax=EPOCH[1], baseline=None, preload=True)
    X = ep.get_data().astype(np.float32)
    y = (ep.events[:, 2] > 0).astype(int)
    return X, y


def balance(X, y, seed: int = 0):
    """EEGMMIDB has ~2x more rest than imagery epochs; undersample rest."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    neg = rng.choice(neg, size=min(len(neg), len(pos)), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    return X[idx], y[idx]

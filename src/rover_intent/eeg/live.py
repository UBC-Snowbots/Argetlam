"""Live EEG -> p_grasp -> UDP. BrainFlow, so the same code runs on the synthetic board today and an
OpenBCI Cyton later (--board cyton --serial /dev/ttyUSB0 or COM3).

    python -m rover_intent.eeg.live --model models/mi_riemann.joblib --host <vm> [--board synthetic]

On the synthetic board the output is meaningless (random signal); it only proves the pipeline runs.
Model files: a single motor-imagery model (train_mi.py: {"model": ...}) or the hybrid grasp decoder
(hybrid_eval.py: {"decoder": HybridDecoder, ...}); the hybrid is streamed with causal filters (OnlineHybrid).
Hybrid expects 8 channels in MOTOR_8 order (FC3 FC4 C3 Cz C4 CP3 CP4 Pz). It was trained at 256 Hz; the Cyton
runs at 250 Hz (~2 % time-scale mismatch; recalibrate on the headset for real use).
Channel order must match eeg.data.MOTOR_8: wire the Cyton N1P..N8P to FC3 FC4 C3 Cz C4 CP3 CP4 Pz.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..transport.udp import Sender
from ..types import EEGState
from .data import BAND, EPOCH

WINDOW_S = EPOCH[1] - EPOCH[0]
TARGET_SFREQ = 160.0


def main():
    import joblib
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    from scipy.signal import butter, resample_poly, sosfiltfilt

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--board", choices=["synthetic", "cyton"], default="synthetic")
    ap.add_argument("--serial", default="")
    ap.add_argument("--hz", type=float, default=8.0)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    board_id = BoardIds.SYNTHETIC_BOARD if args.board == "synthetic" else BoardIds.CYTON_BOARD
    params = BrainFlowInputParams()
    params.serial_port = args.serial
    board = BoardShim(board_id, params)
    fs = BoardShim.get_sampling_rate(board_id)
    chans = BoardShim.get_eeg_channels(board_id)[:8]
    sos = butter(4, BAND, btype="band", fs=fs, output="sos")
    n = int(WINDOW_S * fs) + int(fs)  # 1 s extra for filter edge effects
    tx = Sender(args.host, args.port)
    board.prepare_session()
    board.start_stream()
    if "decoder" in bundle:  # hybrid: stream every new sample through causal filters
        from .hybrid import OnlineHybrid
        online = OnlineHybrid(bundle["decoder"], n_ch=len(chans), step=max(1, int(fs / args.hz)))
        try:
            while True:
                data = board.get_board_data()  # everything since the last call
                if data.shape[1]:
                    for _pc, _pl, pg in online.push(data[chans] * 1e-6):
                        tx.send(EEGState(t=time.time(), p_grasp=pg))
                time.sleep(0.02)
        finally:
            board.stop_stream()
            board.release_session()
        return
    clf = bundle["model"]
    try:
        time.sleep(n / fs)
        while True:
            x = board.get_current_board_data(n)[chans] * 1e-6  # uV -> V, as MNE uses
            x = sosfiltfilt(sos, x, axis=1)[:, -int(WINDOW_S * fs):]
            x = resample_poly(x, int(TARGET_SFREQ), int(fs), axis=1)
            p = float(clf.predict_proba(x[None].astype(np.float32))[0, 1])
            tx.send(EEGState(t=time.time(), p_grasp=p))
            time.sleep(1 / args.hz)
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()

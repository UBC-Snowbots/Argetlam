"""--board petal: the LSL reader (eeg/lsl_board.py) that stands in for BrainFlow when Petal Metrics streams the Muse."""
import os
import time

import numpy as np
import pytest

from rover_intent.eeg.lsl_board import MUSE_LABELS, pick_channels


def test_pick_channels_by_label_reorders_to_muse_order():
    assert pick_channels(["AF7", "TP9", "TP10", "AF8", "AUX"], 5) == [1, 0, 3, 2]
    assert pick_channels(["tp9", "af7", "af8", "tp10"], 4) == [0, 1, 2, 3]


def test_pick_channels_falls_back_to_first_four():
    assert pick_channels([], 5) == [0, 1, 2, 3]
    assert pick_channels(["ch1", "ch2", "ch3", "ch4", "ch5"], 5) == [0, 1, 2, 3]
    with pytest.raises(RuntimeError):
        pick_channels([], 3)


def test_open_board_petal_reads_a_petal_style_stream():
    pylsl = pytest.importorskip("pylsl")
    from rover_intent.eeg.clench import _read, open_board  # noqa: F401  (open_board is the real entry point)
    from rover_intent.eeg.lsl_board import open_lsl

    stype = f"EEGTEST{os.getpid()}"   # unique type so a real stream on the LAN can't be picked up by the test
    order = ["AF7", "TP9", "TP10", "AF8", "AUX"]   # shuffled on purpose: the reader must reorder by label
    info = pylsl.StreamInfo("PetalStream_eeg", stype, len(order), 256, "float32", f"petal-test-{os.getpid()}")
    chans = info.desc().append_child("channels")
    for lab in order:
        chans.append_child("channel").append_child_value("label", lab)
    outlet = pylsl.StreamOutlet(info)
    value = {"TP9": 1.0, "AF7": 2.0, "AF8": 3.0, "TP10": 4.0, "AUX": 9.0}
    sample = [value[lab] for lab in order]

    board, fs, ch = open_lsl(stream_type=stype, timeout=5.0)
    try:
        assert fs == 256 and ch == [0, 1, 2, 3]
        for _ in range(200):       # wait for the inlet to connect, then push a burst
            if outlet.have_consumers():
                break
            time.sleep(0.02)
        for _ in range(64):
            outlet.push_sample(sample)
        got = np.zeros((4, 0))
        t_end = time.time() + 3.0
        while got.shape[1] < 64 and time.time() < t_end:
            time.sleep(0.05)
            got = np.concatenate([got, _read(board, ch)], axis=1)
        assert got.shape[0] == 4 and got.shape[1] >= 64
        # rows come back as TP9 AF7 AF8 TP10 (AUX dropped), same order as BrainFlow's MUSE_2_BOARD
        np.testing.assert_allclose(got[:, -1], [value[m] for m in MUSE_LABELS])
        assert _read(board, ch).shape == (4, 0)   # get_board_data drains the buffer, like BrainFlow
    finally:
        board.stop_stream()
        board.release_session()

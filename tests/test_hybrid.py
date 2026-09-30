import numpy as np
import pytest

pytest.importorskip("pyriemann")
from scipy.signal import sosfilt

from rover_intent.eeg.hybrid import (MRCP, MUBETA, HybridDecoder, OnlineHybrid, StreamFilter, car, sos_for,
                                     windows_at)

FS = 256


@pytest.fixture(scope="module")
def dec():
    rng = np.random.default_rng(0)
    w = lambda n, s: (rng.standard_normal((n, 8, FS)) * s).astype(np.float32)
    return HybridDecoder(fs=FS).fit(w(40, 1.0), w(40, 2.0), w(40, 1.0) + 0.5, w(40, 1.0))


def test_stream_filter_equals_batch():
    x = np.random.default_rng(1).standard_normal((8, 3000)).astype(np.float32)
    f = StreamFilter(MUBETA, FS, 8)
    chunked = np.concatenate([f(x[:, i:i + 37]) for i in range(0, 3000, 37)], axis=1)
    assert np.allclose(chunked, sosfilt(sos_for(MUBETA, FS), x, axis=1), atol=1e-4)


def test_online_is_chunking_invariant_and_matches_batch(dec):
    x = np.random.default_rng(2).standard_normal((8, 4 * FS)).astype(np.float32)
    a = OnlineHybrid(dec)
    out_a = [p for i in range(0, x.shape[1], 7) for p in a.push(x[:, i:i + 7])]
    b = OnlineHybrid(dec)
    out_b = b.push(x)
    assert len(out_a) == len(out_b) > 0 and np.allclose(out_a, out_b, atol=1e-5)
    # batch path used by the evaluation: filter once, cut windows at the same ends, fuse
    mb = sosfilt(sos_for(MUBETA, FS), x, axis=1).astype(np.float32)
    mr = sosfilt(sos_for(MRCP, FS), car(x), axis=1).astype(np.float32)
    ends = np.arange(FS, x.shape[1] + 1, 32)
    pc, pl = dec.p_coming(windows_at(mb, ends, FS)), dec.p_close(windows_at(mr, ends, FS))
    fused = dec.fuse(pc, pl, 32 / FS)
    assert np.allclose(np.array(out_b)[:, 2], fused, atol=1e-4)
    assert np.all(np.array(out_b)[:, 2] <= np.array(out_b)[:, 1] + 1e-9)  # p_grasp never exceeds p_close

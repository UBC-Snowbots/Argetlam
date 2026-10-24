"""The numpy-only scipy.signal stand-ins (used on the laptop when Smart App Control blocks SciPy) match SciPy."""
import numpy as np
import pytest

from rover_intent.eeg import np_signal as nps

sig = pytest.importorskip("scipy.signal")
FS = 256.0


def _resp(sos, x, zi=None):
    return sig.sosfilt(sos, x, axis=-1) if zi is None else sig.sosfilt(sos, x, axis=-1, zi=zi)[0]


@pytest.mark.parametrize("btype,wn", [("band", [20.0, 115.0]), ("band", [8.0, 30.0]), ("low", 40.0),
                                      ("high", 1.0)])
@pytest.mark.parametrize("order", [2, 4])
def test_butter_matches_scipy_response(btype, wn, order):
    """Section pairing may differ, so compare what matters: the filter's output on a random signal."""
    x = np.random.default_rng(0).normal(size=(4, 2048))
    ours = nps.butter(order, wn, btype=btype, fs=FS)
    ref = sig.butter(order, wn, btype=btype, fs=FS, output="sos")
    assert ours.shape == ref.shape
    np.testing.assert_allclose(nps.sosfilt(ours, x), _resp(ref, x), rtol=1e-7, atol=1e-9)


def test_notch_and_tf2sos_match_scipy():
    b, a = nps.iirnotch(60.0, 30.0, fs=FS)
    rb, ra = sig.iirnotch(60.0, 30.0, fs=FS)
    np.testing.assert_allclose(b, rb, rtol=1e-12)
    np.testing.assert_allclose(a, ra, rtol=1e-12)
    np.testing.assert_allclose(nps.tf2sos(b, a), sig.tf2sos(rb, ra), rtol=1e-12, atol=1e-15)


def test_sosfilt_state_and_zi_match_scipy_for_the_clench_chain():
    """The exact chain clench.EmgFrontEnd builds (notch + 4th-order band), streamed in uneven chunks with state."""
    sos = np.vstack([nps.tf2sos(*nps.iirnotch(60.0, 30.0, fs=FS)), nps.butter(4, [20.0, 115.2], "band", fs=FS)])
    ref_sos = np.vstack([sig.tf2sos(*sig.iirnotch(60.0, 30.0, fs=FS)),
                         sig.butter(4, [20.0, 115.2], btype="band", fs=FS, output="sos")])
    x = np.random.default_rng(1).normal(size=(4, 3000))
    ys, z = [], np.zeros((sos.shape[0], 4, 2))
    for a, b in [(0, 7), (7, 300), (300, 301), (301, 3000)]:
        y, z = nps.sosfilt(sos, x[:, a:b], axis=1, zi=z)
        ys.append(y)
    np.testing.assert_allclose(np.hstack(ys), sig.sosfilt(ref_sos, x, axis=1), rtol=1e-7, atol=1e-9)
    # steady-state initial conditions: a constant input passes with no start-up transient, same as SciPy
    step = np.full((1, 400), 3.0)
    ours = nps.sosfilt(sos, step, axis=1, zi=nps.sosfilt_zi(sos)[:, None, :] * 3.0)[0]
    ref = sig.sosfilt(ref_sos, step, axis=1, zi=sig.sosfilt_zi(ref_sos)[:, None, :] * 3.0)[0]
    np.testing.assert_allclose(ours, ref, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("n,nperseg", [(1024, 256), (300, 256), (777, 64), (255, 255)])
def test_welch_matches_scipy(n, nperseg):
    x = np.random.default_rng(2).normal(size=(4, n))
    f, p = nps.welch(x, fs=FS, nperseg=nperseg, noverlap=min(nperseg, n) // 2, axis=1)
    rf, rp = sig.welch(x, fs=FS, nperseg=min(nperseg, n), noverlap=min(nperseg, n) // 2, axis=1)
    np.testing.assert_allclose(f, rf)
    np.testing.assert_allclose(p, rp, rtol=1e-10, atol=1e-14)


def test_session_modules_fall_back_when_scipy_is_blocked(monkeypatch):
    """Simulate the laptop: importing scipy.signal raises ImportError -> alpha and clench load numpy versions."""
    import builtins
    import importlib
    import sys
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("DLL load failed while importing cython_blas: An Application Control policy "
                              "has blocked this file.")
        return real_import(name, *a, **k)

    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m.startswith("rover_intent.eeg.alpha")
             or m.startswith("rover_intent.eeg.clench")}
    monkeypatch.setattr(builtins, "__import__", blocked)
    try:
        alpha = importlib.import_module("rover_intent.eeg.alpha")
        clench = importlib.import_module("rover_intent.eeg.clench")
        assert alpha.welch is nps.welch and clench.sosfilt is nps.sosfilt
        x = np.random.default_rng(3).normal(size=(4, 512))
        assert alpha.band_power(x, FS).shape == (4,)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        for m in [m for m in sys.modules if m.startswith("rover_intent.eeg.alpha")
                  or m.startswith("rover_intent.eeg.clench")]:
            sys.modules.pop(m)
        sys.modules.update(saved)

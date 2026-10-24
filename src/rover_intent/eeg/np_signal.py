"""Numpy-only stand-ins for the few scipy.signal functions the Muse laptop path uses (welch, butter, iirnotch,
tf2sos, sosfilt, sosfilt_zi). Used only when SciPy can't be imported: on the Windows laptop, Windows Smart App
Control blocks SciPy's compiled DLLs (2026-10-23: "DLL load failed while importing cython_blas"), while numpy and
BrainFlow load fine. The math follows scipy.signal (same defaults for the arguments we use) and
tests/test_np_signal.py checks the outputs against SciPy on the VM.

Scope is deliberately narrow: Butterworth low/high/band design as second-order sections, a notch, Direct Form II
transposed SOS filtering with state along the last or a given axis, and a Hann-window Welch PSD (constant detrend,
density scaling, one-sided, mean average).
"""
from __future__ import annotations

import numpy as np


# --- filter design -----------------------------------------------------------------------------------------------
def _bilinear_zpk(z, p, k, fs):
    fs2 = 2.0 * fs
    degree = len(p) - len(z)
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)
    z_z = np.append(z_z, -np.ones(degree))          # zeros at infinity map to Nyquist
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))
    return z_z, p_z, k_z


def _zpk2sos(z, p, k):
    """Pair conjugate poles into sections (any valid pairing gives the same overall filter). Real poles and the
    zeros are paired in order; the gain goes on the first section."""
    p = np.asarray(p, complex)
    z = np.asarray(z, complex)
    n_sec = (max(len(p), len(z)) + 1) // 2
    z = np.append(z, np.zeros(2 * n_sec - len(z)))
    p = np.append(p, np.zeros(2 * n_sec - len(p)))

    def pairs(r):
        cplx = sorted([x for x in r if x.imag > 1e-12], key=lambda x: -abs(x))
        reals = sorted([x.real for x in r if abs(x.imag) <= 1e-12], key=lambda x: -abs(x))
        out = [(x, np.conj(x)) for x in cplx]
        out += [(reals[i], reals[i + 1] if i + 1 < len(reals) else 0.0) for i in range(0, len(reals), 2)]
        return out

    pp, zz = pairs(p), pairs(z)
    sos = np.zeros((n_sec, 6))
    for i in range(n_sec):
        z1, z2 = zz[i]
        p1, p2 = pp[i]
        sos[i, :3] = np.real(np.poly([z1, z2]))
        sos[i, 3:] = np.real(np.poly([p1, p2]))
    sos[0, :3] *= k
    return sos


def butter(N, Wn, btype="low", fs=None, output="sos"):
    """Butterworth design, like scipy.signal.butter(..., output="sos"). Wn in Hz when fs is given."""
    if output != "sos":
        raise NotImplementedError("np_signal.butter only returns second-order sections")
    Wn = np.atleast_1d(np.asarray(Wn, float))
    if fs is not None:
        Wn = 2.0 * Wn / fs                               # normalise to Nyquist = 1, like scipy
    fs_i = 2.0
    warped = 2.0 * fs_i * np.tan(np.pi * Wn / fs_i)
    p = np.exp(1j * np.pi * (2 * np.arange(N) + N + 1) / (2 * N))   # analog lowpass prototype, -1 rad/s poles
    z = np.array([], complex)
    k = 1.0
    bt = {"band": "bandpass", "bandpass": "bandpass", "low": "lowpass", "lowpass": "lowpass",
          "high": "highpass", "highpass": "highpass"}[btype]
    if bt == "lowpass":
        z, p, k = z, p * warped[0], k * warped[0] ** len(p)
    elif bt == "highpass":
        degree = len(p)
        k = k * np.real(1.0 / np.prod(-p))
        p = warped[0] / p
        z = np.zeros(degree)
    else:
        wo = np.sqrt(warped[0] * warped[1])
        bw = warped[1] - warped[0]
        degree = len(p)
        p_lp = p * bw / 2.0
        p = np.concatenate([p_lp + np.sqrt(p_lp ** 2 - wo ** 2), p_lp - np.sqrt(p_lp ** 2 - wo ** 2)])
        z = np.zeros(degree)
        k = k * bw ** degree
    z, p, k = _bilinear_zpk(z, p, k, fs_i)
    return _zpk2sos(z, p, k)


def iirnotch(w0, Q, fs=2.0):
    """Second-order notch, like scipy.signal.iirnotch. Returns (b, a)."""
    w0 = 2.0 * w0 / fs
    bw = w0 / Q * np.pi
    w0 = w0 * np.pi
    gb = 1.0 / np.sqrt(2.0)
    beta = (np.sqrt(1.0 - gb ** 2) / gb) * np.tan(bw / 2.0)
    gain = 1.0 / (1.0 + beta)
    b = gain * np.array([1.0, -2.0 * np.cos(w0), 1.0])
    a = np.array([1.0, -2.0 * gain * np.cos(w0), 2.0 * gain - 1.0])
    return b, a


def tf2sos(b, a):
    """Only the second-order case we need: one section, normalised by a[0]."""
    b, a = np.asarray(b, float), np.asarray(a, float)
    if len(b) > 3 or len(a) > 3:
        raise NotImplementedError("np_signal.tf2sos handles second-order filters only")
    b = np.pad(b, (0, 3 - len(b)))
    a = np.pad(a, (0, 3 - len(a)))
    return np.concatenate([b, a])[None, :] / a[0]


# --- filtering ---------------------------------------------------------------------------------------------------
def sosfilt_zi(sos):
    """Initial state for a step response at steady state, like scipy.signal.sosfilt_zi. Shape (n_sections, 2)."""
    sos = np.asarray(sos, float)
    zi = np.empty((sos.shape[0], 2))
    scale = 1.0
    for s, sec in enumerate(sos):
        b, a = sec[:3] / sec[3], sec[3:] / sec[3]
        i_minus_a = np.array([[1.0 + a[1], -1.0], [a[2], 1.0]])
        zi[s] = scale * np.linalg.solve(i_minus_a, b[1:] - a[1:] * b[0])
        scale *= b.sum() / a.sum()
    return zi


def sosfilt(sos, x, axis=-1, zi=None):
    """Cascade of Direct Form II transposed biquads, like scipy.signal.sosfilt. zi has shape
    (n_sections, ..., 2) with `axis` removed from x's shape; returns (y, zf) when zi is given, else y."""
    sos = np.asarray(sos, float)
    x = np.moveaxis(np.asarray(x, float), axis, -1)
    lead = x.shape[:-1]
    n = x.shape[-1]
    if zi is None:
        z = np.zeros((sos.shape[0], *lead, 2))
    else:
        z = np.array(zi, float, copy=True).reshape(sos.shape[0], *lead, 2)
    y = x.copy()
    for s, sec in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = sec
        b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
        z0, z1 = z[s, ..., 0].copy(), z[s, ..., 1].copy()
        out = np.empty_like(y)
        for i in range(n):
            xi = y[..., i]
            yi = b0 * xi + z0
            z0 = b1 * xi - a1 * yi + z1
            z1 = b2 * xi - a2 * yi
            out[..., i] = yi
        z[s, ..., 0], z[s, ..., 1] = z0, z1
        y = out
    y = np.moveaxis(y, -1, axis)
    return y if zi is None else (y, z)


# --- spectrum ----------------------------------------------------------------------------------------------------
def welch(x, fs=1.0, nperseg=256, noverlap=None, axis=-1):
    """Welch PSD with scipy's defaults: periodic Hann window, constant detrend, density scaling, one-sided,
    mean over segments. Returns (f, pxx) with pxx's frequency axis at `axis`."""
    x = np.moveaxis(np.asarray(x, float), axis, -1)
    n = x.shape[-1]
    nperseg = min(int(nperseg), n)
    noverlap = nperseg // 2 if noverlap is None else int(noverlap)
    step = nperseg - noverlap
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nperseg) / nperseg)
    scale = 1.0 / (fs * (win ** 2).sum())
    starts = np.arange(0, n - nperseg + 1, step)
    segs = np.stack([x[..., s:s + nperseg] for s in starts], axis=-2)        # (..., n_seg, nperseg)
    segs = segs - segs.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(segs * win, axis=-1)) ** 2 * scale
    if nperseg % 2 == 0:
        spec[..., 1:-1] *= 2.0
    else:
        spec[..., 1:] *= 2.0
    pxx = spec.mean(axis=-2)
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return f, np.moveaxis(pxx, -1, axis)

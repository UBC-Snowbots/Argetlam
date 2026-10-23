"""SSVEP decoding: training-free CCA / FBCCA and calibrated (ensemble) TRCA.

Steady-state visual evoked potentials: when you look at something flickering at f Hz, the occipital EEG grows a
component at f and its harmonics. If the screen shows several targets at different frequencies, the one you are
looking at can be read out. On a Muse 2 there is no occipital electrode (channels are TP9/TP10 behind the ears and
AF7/AF8 on the forehead), so this is a research pipeline we characterise offline first (see docs/muse_readiness.md).

Everything here is pure numpy/scipy on arrays shaped (n_channels, n_times), so the offline studies, the synthetic
tests and the live detector (eeg/muse_detect.py) all share one implementation.

- reference_signals(freqs, fs, n, harmonics) -> dict f -> (2*harmonics, n) sin/cos bank
- cca_scores(X, freqs, fs, harmonics) -> correlation per candidate frequency (training-free)
- fbcca_scores(X, freqs, fs, ...) -> filter-bank CCA score per candidate (training-free, usually stronger)
- TRCA(fs).fit(epochs, labels).scores(X) -> correlation per class (needs per-user calibration)
- itr_bits_per_min(accuracy, n_classes, seconds) -> Wolpaw information transfer rate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


def notch_mains(X: np.ndarray, fs: float, mains=(50.0, 60.0)) -> np.ndarray:
    """Remove mains hum (and its narrow band) before CCA/FBCCA. Live Muse EEG has strong 50/60 Hz line noise, and a
    5th harmonic reference of a 12 or 15 Hz target lands on 60 Hz, so an un-notched signal makes every candidate
    correlate with the mains. Tsinghua benchmark data is already clean, so the offline study does not need this."""
    y = np.asarray(X, float)
    pad = min(y.shape[-1] - 1, 48)
    for f0 in mains:
        if 0 < f0 < 0.48 * fs:
            b, a = iirnotch(f0, 30.0, fs=fs)
            y = filtfilt(b, a, y, axis=-1, padlen=pad)
    return y


def reference_signals(freqs, fs: float, n: int, harmonics: int = 3) -> dict:
    """Sin/cos reference bank per frequency: (2*harmonics, n). Used as the CCA target Y."""
    t = np.arange(n) / fs
    out = {}
    for f in freqs:
        rows = []
        for h in range(1, harmonics + 1):
            rows.append(np.sin(2 * np.pi * h * f * t))
            rows.append(np.cos(2 * np.pi * h * f * t))
        out[float(f)] = np.asarray(rows)
    return out


def _canon_corr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Canonical correlations between X (p, n) and Y (q, n) along time. Returns sorted singular values (desc)."""
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    # QR of the (n, p) and (n, q) data matrices; canonical correlations are the singular values of Qx^T Qy.
    Qx, _ = np.linalg.qr(X.T)
    Qy, _ = np.linalg.qr(Y.T)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def cca_scores(X: np.ndarray, freqs, fs: float, harmonics: int = 3, refs: dict | None = None) -> np.ndarray:
    """Max canonical correlation of X (n_ch, n_times) against each frequency's sin/cos bank. argmax = prediction."""
    X = np.asarray(X, float)
    refs = refs if refs is not None else reference_signals(freqs, fs, X.shape[1], harmonics)
    return np.array([_canon_corr(X, refs[float(f)])[0] for f in freqs])


def _subband_sos(k: int, fs: float, high: float = 88.0):
    """FBCCA sub-band k (0-indexed): pass [8*(k+1), high] Hz (Chen et al. 2015)."""
    lo = 8.0 * (k + 1)
    high = min(high, 0.48 * fs)
    if lo >= high:
        return None
    return butter(4, [lo, high], btype="band", fs=fs, output="sos")


def fbcca_scores(X: np.ndarray, freqs, fs: float, harmonics: int = 3, n_bands: int = 5,
                 a: float = 1.25, b: float = 0.25, refs: dict | None = None) -> np.ndarray:
    """Filter-bank CCA: sum over sub-bands of w(k) * rho_k^2, w(k) = (k+1)^-a + b. argmax = prediction."""
    X = np.asarray(X, float)
    refs = refs if refs is not None else reference_signals(freqs, fs, X.shape[1], harmonics)
    total = np.zeros(len(freqs))
    pad = min(X.shape[1] - 1, int(0.2 * fs) + 6)
    for k in range(n_bands):
        sos = _subband_sos(k, fs)
        if sos is None:
            continue
        xb = sosfiltfilt(sos, X, axis=1, padlen=pad)
        w = (k + 1) ** (-a) + b
        rho = np.array([_canon_corr(xb, refs[float(f)])[0] for f in freqs])
        total += w * rho ** 2
    return total


@dataclass
class TRCA:
    """Task-related component analysis (Nakanishi et al. 2018), ensemble by default. Needs per-user calibration.

    fit(epochs, labels): epochs (n_trials, n_ch, n_times), labels ints. scores(X): correlation per class.
    """
    fs: float
    ensemble: bool = True
    bandpass: tuple[float, float] | None = (7.0, 70.0)

    def _filt(self, X: np.ndarray) -> np.ndarray:
        if not self.bandpass:
            return X
        hi = min(self.bandpass[1], 0.48 * self.fs)
        b, a = butter(4, [self.bandpass[0], hi], btype="band", fs=self.fs)
        pad = min(X.shape[-1] - 1, int(0.2 * self.fs) + 6)
        return filtfilt(b, a, X, axis=-1, padlen=pad)

    @staticmethod
    def _trca_filter(trials: np.ndarray) -> np.ndarray:
        """Top TRCA spatial filter for one class. trials: (n_trials, n_ch, n_times)."""
        nt, nc, ns = trials.shape
        # S = covariance of summed trials minus per-trial autocovariance => between-trial covariance.
        summ = trials.sum(axis=0)  # (nc, ns)
        S = summ @ summ.T
        for x in trials:
            S -= x @ x.T
        cat = trials.transpose(1, 0, 2).reshape(nc, nt * ns)  # (nc, nt*ns)
        cat = cat - cat.mean(axis=1, keepdims=True)
        Q = cat @ cat.T
        Q += 1e-6 * np.trace(Q) / nc * np.eye(nc)  # ridge: Muse has only 2-4 channels
        w, v = np.linalg.eig(np.linalg.solve(Q, S))
        return np.real(v[:, np.argmax(np.real(w))])

    def fit(self, epochs: np.ndarray, labels) -> "TRCA":
        epochs = self._filt(np.asarray(epochs, float))
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        self.filters_, self.templates_ = [], []
        for c in self.classes_:
            trials = epochs[labels == c]
            w = self._trca_filter(trials)
            self.filters_.append(w)
            self.templates_.append(trials.mean(axis=0))  # (n_ch, n_times)
        self.filters_ = np.array(self.filters_)  # (n_classes, n_ch)
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Correlation of X (n_ch, n_times) with each class template. argmax = prediction."""
        X = self._filt(np.asarray(X, float))
        W = self.filters_.T if self.ensemble else None  # (n_ch, n_classes)
        out = []
        for i in range(len(self.classes_)):
            if self.ensemble:
                a = (W.T @ X)           # (n_classes, n_times)
                b = (W.T @ self.templates_[i])
            else:
                w = self.filters_[i]
                a = (w @ X)[None, :]
                b = (w @ self.templates_[i])[None, :]
            out.append(_pearson(a.ravel(), b.ravel()))
        return np.array(out)

    def predict(self, X: np.ndarray):
        return self.classes_[int(np.argmax(self.scores(X)))]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def itr_bits_per_min(accuracy: float, n_classes: int, seconds: float) -> float:
    """Wolpaw ITR in bits/min. seconds = decision window (plus any gaze-shift gap you add outside)."""
    p = min(max(accuracy, 1e-6), 1 - 1e-9)
    n = n_classes
    if n < 2 or seconds <= 0:
        return 0.0
    bits = np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1)) if n > 1 else 0.0
    bits = max(bits, 0.0)
    return float(bits * 60.0 / seconds)

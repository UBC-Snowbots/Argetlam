"""Alpha (eyes-closed) detection: 8-12 Hz power goes up when the eyes close (Berger effect).

This is the robust, electrode-light fallback control for the Muse: the alpha rhythm is strongest occipitally but is
still picked up at the temporal/forehead dry electrodes the Muse has. "Close your eyes to confirm" is a slow but
dependable switch. Pure numpy/scipy on (n_channels, n_times) arrays, shared by the offline study, the synthetic
tests and the live detector.

- band_power(X, fs, band) -> per-channel power (Welch)
- rel_alpha(X, fs, ...) -> per-channel alpha power / broadband power (a montage-robust feature)
- alpha_log_features(X, fs, ...) -> log band powers, the feature vector for the LDA classifier
- AlphaDetector: calibrate on eyes-open vs eyes-closed windows, then push() a stream for an online decision
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
try:
    from scipy.signal import welch
except ImportError:  # e.g. Windows Smart App Control blocks SciPy's DLLs on the laptop: numpy-only stand-in
    from rover_intent.eeg.np_signal import welch

ALPHA_BAND = (8.0, 12.0)
BROAD_BAND = (4.0, 30.0)


def band_power(X: np.ndarray, fs: float, band=ALPHA_BAND, nperseg: int | None = None) -> np.ndarray:
    """Average PSD in `band` per channel, via Welch. X: (n_ch, n_times) -> (n_ch,)."""
    X = np.asarray(X, float)
    n = X.shape[1]
    nperseg = min(n, nperseg or int(fs))  # ~1 s segments; capped at the window length
    nperseg = max(nperseg, 16)
    f, pxx = welch(X, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=1)
    sel = (f >= band[0]) & (f <= band[1])
    if not sel.any():
        sel = np.argmin(np.abs(f - np.mean(band)))[None]
    return pxx[:, sel].mean(axis=1)


def rel_alpha(X: np.ndarray, fs: float, alpha=ALPHA_BAND, broad=BROAD_BAND, nperseg: int | None = None) -> np.ndarray:
    """Per-channel alpha power divided by broadband power. Robust to per-electrode gain / contact differences."""
    a = band_power(X, fs, alpha, nperseg)
    b = band_power(X, fs, broad, nperseg)
    return a / np.maximum(b, 1e-12)


def alpha_log_features(X: np.ndarray, fs: float, alpha=ALPHA_BAND, broad=BROAD_BAND,
                       nperseg: int | None = None) -> np.ndarray:
    """Feature vector for a classifier: per-channel log alpha power and log relative alpha, concatenated."""
    a = band_power(X, fs, alpha, nperseg)
    r = rel_alpha(X, fs, alpha, broad, nperseg)
    return np.concatenate([np.log(np.maximum(a, 1e-12)), np.log(np.maximum(r, 1e-12))])


@dataclass
class AlphaConfig:
    alpha: tuple[float, float] = ALPHA_BAND
    broad: tuple[float, float] = BROAD_BAND
    enter_s: float = 1.0   # closed decision must persist this long before firing
    refractory_s: float = 2.0


@dataclass
class AlphaDetector:
    """Online eyes-closed detector. Calibrate a threshold on the relative-alpha feature from labelled windows,
    then feed windows with push() -> fires once per sustained eyes-closed episode (a confirm/grab event)."""
    fs: float
    threshold: float            # on mean-over-channels relative alpha
    cfg: AlphaConfig = field(default_factory=AlphaConfig)
    t0: float = 0.0

    def __post_init__(self):
        self._closed_since = None
        self._refractory_until = -np.inf
        self.metric = 0.0

    @classmethod
    def calibrate(cls, open_windows, closed_windows, fs: float, cfg: AlphaConfig | None = None):
        """open_windows / closed_windows: lists of (n_ch, n_times) arrays. Threshold = midpoint that best splits
        the mean relative-alpha metric. Returns (detector, stats)."""
        cfg = cfg or AlphaConfig()
        mo = np.array([rel_alpha(w, fs, cfg.alpha, cfg.broad).mean() for w in open_windows])
        mc = np.array([rel_alpha(w, fs, cfg.alpha, cfg.broad).mean() for w in closed_windows])
        # threshold: value maximising balanced accuracy on the calibration windows
        cand = np.sort(np.concatenate([mo, mc]))
        mids = (cand[:-1] + cand[1:]) / 2 if len(cand) > 1 else cand
        best_t, best_acc = float(np.median(np.concatenate([mo, mc]))), -1.0
        for t in mids:
            acc = 0.5 * ((mo < t).mean() + (mc >= t).mean())
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        stats = {"threshold": best_t, "open_mean": float(mo.mean()), "closed_mean": float(mc.mean()),
                 "separation": float((mc.mean() - mo.mean()) / (mo.std() + mc.std() + 1e-9)),
                 "train_bal_acc": float(best_acc)}
        return cls(fs, best_t, cfg), stats

    def push_window(self, X: np.ndarray, t: float) -> bool:
        """Feed one analysis window ending at time t. Returns True exactly when a confirm fires."""
        self.metric = float(rel_alpha(X, self.fs, self.cfg.alpha, self.cfg.broad).mean())
        closed = self.metric >= self.threshold
        fired = False
        if closed:
            self._closed_since = t if self._closed_since is None else self._closed_since
            if t - self._closed_since >= self.cfg.enter_s and t >= self._refractory_until:
                fired = True
                self._refractory_until = t + self.cfg.refractory_s
                self._closed_since = None
        else:
            self._closed_since = None
        return fired

"""Hybrid grasp decoder: stage 1 "grasp coming" (mu/beta, Riemann) x stage 2 "close now" (MRCP, shrinkage LDA).

Evidence (docs/LOG.md, BNCI 001-2020, 8 motor channels): mu/beta separates pre-grasp from idle (86 % calibrated) but
cannot tell closing from reach start (67 %); MRCP can (81 %). So:
    p_coming(t) = stage 1 on the last 1 s (8-30 Hz)
    p_close(t)  = stage 2 on the last 1 s (0.3-3 Hz, 8-channel common average, 16 samples/ch)
    p_grasp(t)  = min( max(p_coming over the last `arm_s`), p_close(t) )
p_grasp feeds the Arbiter (hysteresis 0.7/0.3 + 0.5 s dwell) exactly like the old single-model p_grasp.
Everything is causal and streaming: `OnlineHybrid.push(chunk)` keeps filter state between chunks, so the same code
runs offline (pseudo-online evaluation) and live (BrainFlow).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

MOTOR_8 = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Pz"]
MUBETA, MRCP = (8.0, 30.0), (0.3, 3.0)
MRCP_POINTS = 16  # samples per channel per 1 s window


def sos_for(band, fs):
    from scipy.signal import butter
    return butter(4, band, btype="band", fs=fs, output="sos")


class StreamFilter:
    """Causal band-pass with persistent state: filtering chunk by chunk == filtering the whole signal at once."""

    def __init__(self, band, fs, n_ch):
        from scipy.signal import sosfilt_zi
        self.sos = sos_for(band, fs)
        self.zi = np.repeat(sosfilt_zi(self.sos)[:, None, :], n_ch, axis=1) * 0.0

    def __call__(self, x):
        from scipy.signal import sosfilt
        y, self.zi = sosfilt(self.sos, x, axis=1, zi=self.zi)
        return y.astype(np.float32)


def car(x):
    return x - x.mean(axis=0, keepdims=True)


def mrcp_feat(w):
    """(n, ch, fs) window of MRCP-band signal -> (n, ch*16) amplitudes at 16 evenly spaced points."""
    idx = np.linspace(0, w.shape[-1] - 1, MRCP_POINTS).round().astype(int)
    return w[..., idx].reshape(w.shape[0], -1)


def windows_at(x, ends, fs):
    """1 s windows ending at sample indices `ends` (exclusive) -> (n, ch, fs)."""
    return np.stack([x[:, e - fs:e] for e in ends]) if len(ends) else np.zeros((0, x.shape[0], fs), np.float32)


@dataclass
class HybridDecoder:
    fs: int = 256
    arm_s: float = 1.5
    stage1: object = None
    stage2: object = None

    def fit(self, mb_pos, mb_neg, mr_pos, mr_neg):
        """mb_*: (n, ch, fs) mu/beta-filtered windows; mr_*: (n, ch, fs) MRCP-filtered CAR windows."""
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X1 = np.concatenate([mb_pos, mb_neg]); y1 = np.r_[np.ones(len(mb_pos)), np.zeros(len(mb_neg))]
        w1 = np.where(y1 == 1, len(y1) / (2 * len(mb_pos)), len(y1) / (2 * len(mb_neg)))  # balance classes
        self.stage1 = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=1000))
        self.stage1.fit(X1, y1, logisticregression__sample_weight=w1)
        X2 = mrcp_feat(np.concatenate([mr_pos, mr_neg])); y2 = np.r_[np.ones(len(mr_pos)), np.zeros(len(mr_neg))]
        self.stage2 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                                                               priors=[0.5, 0.5]))
        self.stage2.fit(X2, y2)
        return self

    def p_coming(self, mb_windows):
        return self.stage1.predict_proba(mb_windows)[:, 1]

    def p_close(self, mr_windows):
        return self.stage2.predict_proba(mrcp_feat(mr_windows))[:, 1]

    def fuse(self, p_coming, p_close, step_s):
        """Vectorised fusion over a time series sampled every step_s."""
        k = max(1, int(round(self.arm_s / step_s)))
        armed = np.array([p_coming[max(0, i - k + 1):i + 1].max() for i in range(len(p_coming))])
        return np.minimum(armed, p_close)


@dataclass
class OnlineHybrid:
    """Streaming wrapper: push raw 8-channel EEG chunks (volts, MOTOR_8 order), get p values every `step` samples."""
    decoder: HybridDecoder
    n_ch: int = 8
    step: int = 32
    _buf: np.ndarray = field(default=None, repr=False)
    _hist: deque = field(default=None, repr=False)

    def __post_init__(self):
        fs = self.decoder.fs
        self.f_mb = StreamFilter(MUBETA, fs, self.n_ch)
        self.f_mr = StreamFilter(MRCP, fs, self.n_ch)
        self._mb = np.zeros((self.n_ch, 0), np.float32)
        self._mr = np.zeros((self.n_ch, 0), np.float32)
        self._since = 0
        self._hist = deque(maxlen=max(1, int(round(self.decoder.arm_s * fs / self.step))))

    def push(self, chunk):
        """chunk: (8, n) raw samples, any n. Returns (p_coming, p_close, p_grasp) for every completed step; the
        result does not depend on how the stream is chunked."""
        fs, out, i, n = self.decoder.fs, [], 0, chunk.shape[1]
        while i < n:
            take = min(self.step - self._since, n - i)
            piece = chunk[:, i:i + take]
            i += take
            self._mb = np.concatenate([self._mb, self.f_mb(piece)], axis=1)[:, -fs:]
            self._mr = np.concatenate([self._mr, self.f_mr(car(piece))], axis=1)[:, -fs:]
            self._since += take
            if self._since == self.step:
                self._since = 0
                if self._mb.shape[1] >= fs:
                    pc = float(self.decoder.p_coming(self._mb[None])[0])
                    pl = float(self.decoder.p_close(self._mr[None])[0])
                    self._hist.append(pc)
                    out.append((pc, pl, min(max(self._hist), pl)))
        return out

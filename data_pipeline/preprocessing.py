"""
LogiEdge - Task C2 / C3: preprocessing pipeline.

Strict order, identical in training and at the edge:

  1. FILTER        5-sample moving average on temperature and on vibration
  2. WINDOW        30 s window, 10 s step  (66% overlap)
  3. FEATURES      6 values, feature-level fusion of the two streams:
                     f0 temp_mean       degC
                     f1 temp_std        degC
                     f2 temp_roc        degC/min  (least-squares slope x 60)
                     f3 vib_rms         g
                     f4 vib_peak        g
                     f5 vib_kurtosis    dimensionless (Fisher, excess)
  4. NORMALISE     z-score with mean/std FROZEN in training_stats.npy.
                   Never recomputed from live data - see the note below.

Why the stats must be frozen (Task C2, mandatory experiment):
a truck sitting in a compressor-failure state for an hour would, under adaptive
normalisation, have its own fault re-centred to "mean" and the model would fall
silent exactly when it matters. The 3-sigma-shift experiment in
experiments/normalisation_experiment.py quantifies that failure.

Data fusion level (Task C3): FEATURE-LEVEL. Justified in reports/final_report.md
Section 2 - the two streams have incommensurate rates (1 Hz vs 0.5 Hz) and units,
so data-level fusion would require resampling and destroys the kurtosis signal;
decision-level fusion cannot express the CONJUNCTION (drift AND bearing wear) that
separates Warning from Critical.
"""

from collections import deque
from pathlib import Path

import numpy as np

WINDOW_S = 30.0
STEP_S = 10.0
MA_TAPS = 5
FEATURE_NAMES = ["temp_mean", "temp_std", "temp_roc_c_per_min",
                 "vib_rms", "vib_peak", "vib_kurtosis"]
N_FEATURES = len(FEATURE_NAMES)
CLASS_NAMES = ["Normal", "Warning", "Critical"]
DEFAULT_STATS = Path(__file__).with_name("training_stats.npy")


# ---------------------------------------------------------------- 1. FILTER
class MovingAverage:
    """Causal 5-sample moving average. Streaming - no lookahead."""

    def __init__(self, taps=MA_TAPS):
        self.buf = deque(maxlen=taps)

    def push(self, x):
        self.buf.append(float(x))
        return sum(self.buf) / len(self.buf)


# ------------------------------------------------- 2+3. WINDOW AND FEATURES
def _kurtosis(x):
    """Fisher (excess) kurtosis; 0.0 for a Gaussian. Guarded for flat windows."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4:
        return 0.0
    m = x.mean()
    s = x.std()
    if s < 1e-9:
        return 0.0
    return float(((x - m) ** 4).mean() / s ** 4 - 3.0)


def _roc_c_per_min(t, v):
    """Least-squares slope in degC/min. Robust to the 1 Hz noise floor; a two-point
    difference would be dominated by the 0.3 degC sensor sigma."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if t.size < 2 or np.ptp(t) < 1e-6:
        return 0.0
    slope = np.polyfit(t, v, 1)[0]      # degC per second
    return float(slope * 60.0)


def extract_features(t_temp, temp, t_vib, vib):
    """The 6-value fused feature vector for one window."""
    temp = np.asarray(temp, dtype=np.float64)
    vib = np.asarray(vib, dtype=np.float64)
    if temp.size == 0 or vib.size == 0:
        return None
    return np.array([
        temp.mean(),
        temp.std(),
        _roc_c_per_min(t_temp, temp),
        float(np.sqrt((vib ** 2).mean())),
        float(np.max(np.abs(vib))),
        _kurtosis(vib),
    ], dtype=np.float32)


class WindowFeaturiser:
    """Streaming 30 s / 10 s sliding window over the two filtered streams.

    Feed it (sim_time, kind, payload) tuples in time order - exactly what
    simulator.stream() yields and exactly what the MQTT callback delivers.
    Emits a 6-vector every 10 s once the first full 30 s window is buffered.
    """

    def __init__(self, window_s=WINDOW_S, step_s=STEP_S):
        self.window_s = window_s
        self.step_s = step_s
        self.ma_t = MovingAverage()
        self.ma_v = MovingAverage()
        self.temp = deque()   # (t, filtered_value)
        self.vib = deque()
        self.door_open = False
        self.next_emit = window_s
        self.now = 0.0

    def push(self, t, kind, payload):
        """Returns a feature vector when a window closes, else None."""
        self.now = t
        if kind == "temperature":
            self.temp.append((t, self.ma_t.push(payload["value_c"])))
        elif kind == "vibration":
            self.vib.append((t, self.ma_v.push(payload["rms_g"])))
        elif kind == "door":
            self.door_open = (payload["event"] == "OPEN")

        cutoff = t - self.window_s
        while self.temp and self.temp[0][0] < cutoff:
            self.temp.popleft()
        while self.vib and self.vib[0][0] < cutoff:
            self.vib.popleft()

        if t >= self.next_emit and self.temp and self.vib:
            self.next_emit += self.step_s
            tt, tv = zip(*self.temp)
            vt, vv = zip(*self.vib)
            return extract_features(tt, tv, vt, vv)
        return None


def windows_from_stream(stream):
    """Batch helper: full simulator stream -> (N, 6) raw feature matrix."""
    return windows_with_times(stream)[0]


def windows_with_times(stream):
    """As above, plus the closing timestamp of each window.

    The timestamps are what let generate_dataset.py label a window by its true
    CONDITION rather than by the run it came from - see the note there."""
    wf = WindowFeaturiser()
    out, ts = [], []
    for t, kind, payload in stream:
        f = wf.push(t, kind, payload)
        if f is not None:
            out.append(f)
            ts.append(t)
    return np.asarray(out, dtype=np.float32), np.asarray(ts, dtype=np.float64)


# ----------------------------------------------------------- 4. NORMALISE
class Normaliser:
    """z-score with frozen statistics. fit() is called ONCE, offline, on clean
    Normal-class data; at the edge only load()/transform() are ever used."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1e-6          # never divide by zero
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def save(self, path=DEFAULT_STATS):
        np.save(path, np.stack([self.mean, self.std]).astype(np.float32))
        return Path(path)

    @classmethod
    def load(cls, path=DEFAULT_STATS):
        a = np.load(path)
        return cls(a[0].astype(np.float32), a[1].astype(np.float32))

    def transform(self, x):
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std
                ).astype(np.float32)

    def shifted(self, sigmas=3.0):
        """Corrupted copy for the mandatory 3-sigma experiment: pretend the stats
        were recomputed on data whose mean sat 3 sigma high."""
        return Normaliser(self.mean + sigmas * self.std, self.std)

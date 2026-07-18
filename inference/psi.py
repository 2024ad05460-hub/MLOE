"""Reusable PSI monitor used by both the CLI demo and the inference service."""
from __future__ import annotations
import json
import time
from collections import deque
from pathlib import Path
import numpy as np

BINS = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0001], dtype=np.float64)
EPS = 0.005


def distribution(values) -> np.ndarray:
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=BINS)
    proportions = counts / max(1, counts.sum())
    proportions = np.maximum(proportions, EPS)
    return proportions / proportions.sum()


def psi(expected, actual) -> float:
    expected = np.maximum(np.asarray(expected, dtype=np.float64), EPS)
    actual = np.maximum(np.asarray(actual, dtype=np.float64), EPS)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


class RollingPSI:
    def __init__(self, reference_path, rolling_n=100, interval_s=60.0, alert=0.25, clear=0.10):
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        self.expected = np.asarray(reference["distribution"], dtype=np.float64)
        self.rolling_n = int(rolling_n)
        self.interval_s = float(interval_s)
        self.alert_threshold = float(alert)
        self.clear_threshold = float(clear)
        self.scores = deque(maxlen=self.rolling_n)
        self.classes = deque(maxlen=self.rolling_n)
        self.last_check = 0.0
        self.alert_active = False

    def update(self, probabilities, predicted_class: int, now=None):
        now = time.time() if now is None else float(now)
        self.scores.append(float(probabilities[0]))
        self.classes.append(int(predicted_class))
        if len(self.scores) < self.rolling_n:
            return None
        if self.last_check and now - self.last_check < self.interval_s:
            return None
        self.last_check = now
        current = distribution(self.scores)
        value = psi(self.expected, current)
        critical_share = float(np.mean(np.asarray(self.classes) == 2))
        state_change = None
        if value > self.alert_threshold and not self.alert_active:
            self.alert_active = True
            state_change = "alert"
        elif value < self.clear_threshold and self.alert_active:
            self.alert_active = False
            state_change = "clear"
        return {
            "psi": value,
            "critical_share": critical_share,
            "distribution": current.tolist(),
            "state_change": state_change,
            "alert_active": self.alert_active,
            "window_size": len(self.scores),
        }

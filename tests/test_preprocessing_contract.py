"""Contract tests for the LogiEdge preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data_pipeline.preprocessing as preprocessing  # noqa: E402
from data_pipeline.simulator import ColdChainSimulator  # noqa: E402


FEATURE_NAMES = preprocessing.FEATURE_NAMES
Normaliser = preprocessing.Normaliser
WindowFeaturiser = preprocessing.WindowFeaturiser
extract_features = preprocessing.extract_features

EXPECTED_FEATURE_NAMES = [
    "temp_mean",
    "temp_std",
    "temp_roc_c_per_min",
    "vib_rms",
    "vib_peak",
    "vib_kurtosis",
]

EXPECTED_FEATURE_COUNT = 6

TRAINING_STATS_PATH = (
    ROOT
    / "data_pipeline"
    / "training_stats.npy"
)


def _normaliser_array(
    normaliser: Any,
    public_name: str,
    fitted_name: str,
) -> np.ndarray:
    """Return a normaliser array under either supported name."""

    if hasattr(normaliser, public_name):
        value = getattr(normaliser, public_name)
    elif hasattr(normaliser, fitted_name):
        value = getattr(normaliser, fitted_name)
    else:
        raise AssertionError(
            f"Normaliser has neither {public_name!r} "
            f"nor {fitted_name!r}."
        )

    return np.asarray(
        value,
        dtype=np.float32,
    ).reshape(-1)


def _load_normaliser() -> Normaliser:
    """Load the frozen training statistics."""

    assert TRAINING_STATS_PATH.is_file(), (
        "Frozen training statistics are missing: "
        f"{TRAINING_STATS_PATH}"
    )

    return Normaliser.load(
        TRAINING_STATS_PATH
    )


def _extract_constant_features() -> np.ndarray:
    """Extract features from constant timestamped sensor signals."""

    temperature = np.full(
        30,
        4.0,
        dtype=np.float32,
    )

    vibration = np.full(
        15,
        0.45,
        dtype=np.float32,
    )

    temperature_timestamps = np.arange(
        temperature.size,
        dtype=np.float32,
    )

    vibration_timestamps = (
        np.arange(
            vibration.size,
            dtype=np.float32,
        )
        * 2.0
    )

    return np.asarray(
        extract_features(
            temperature_timestamps,
            temperature,
            vibration_timestamps,
            vibration,
        ),
        dtype=np.float32,
    )


def test_preprocessing_constants() -> None:
    """Timing constants must match the assignment contract."""

    assert hasattr(preprocessing, "MA_TAPS"), (
        "preprocessing.MA_TAPS is missing."
    )

    assert hasattr(preprocessing, "WINDOW_S"), (
        "preprocessing.WINDOW_S is missing."
    )

    assert hasattr(preprocessing, "STEP_S"), (
        "preprocessing.STEP_S is missing."
    )

    assert preprocessing.MA_TAPS == 5, (
        "Moving-average filter must use 5 samples, "
        f"but found {preprocessing.MA_TAPS!r}."
    )

    assert preprocessing.WINDOW_S == 30, (
        "Feature window must be 30 seconds, "
        f"but found {preprocessing.WINDOW_S!r}."
    )

    assert preprocessing.STEP_S == 10, (
        "Feature step must be 10 seconds, "
        f"but found {preprocessing.STEP_S!r}."
    )


def test_feature_order() -> None:
    """Training and inference must share one fixed feature order."""

    assert list(FEATURE_NAMES) == EXPECTED_FEATURE_NAMES, (
        "Feature order does not match the preprocessing contract.\n"
        f"Expected: {EXPECTED_FEATURE_NAMES}\n"
        f"Actual:   {list(FEATURE_NAMES)}"
    )

    assert len(FEATURE_NAMES) == EXPECTED_FEATURE_COUNT


def test_feature_vector_has_six_values() -> None:
    """Feature extraction must produce six finite values."""

    features = _extract_constant_features()

    assert features.shape == (EXPECTED_FEATURE_COUNT,)
    assert np.isfinite(features).all()


def test_constant_input_feature_values() -> None:
    """Constant inputs must produce stable expected features."""

    features = _extract_constant_features()

    expected = np.asarray(
        [
            4.0,
            0.0,
            0.0,
            0.45,
            0.45,
            0.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        features,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_pipeline_emits_valid_features() -> None:
    """Simulator output must produce valid feature windows."""

    featuriser = WindowFeaturiser()
    emitted: list[np.ndarray] = []

    simulator = ColdChainSimulator(
        "none",
        "TEST",
        seed=123,
    )

    for timestamp, event_type, payload in simulator.stream(90):
        feature_vector = featuriser.push(
            timestamp,
            event_type,
            payload,
        )

        if feature_vector is not None:
            emitted.append(
                np.asarray(
                    feature_vector,
                    dtype=np.float32,
                )
            )

    assert emitted, (
        "WindowFeaturiser did not emit any feature vectors."
    )

    feature_matrix = np.asarray(
        emitted,
        dtype=np.float32,
    )

    assert feature_matrix.ndim == 2
    assert feature_matrix.shape[1] == EXPECTED_FEATURE_COUNT
    assert np.isfinite(feature_matrix).all()


def test_training_stats_file_exists() -> None:
    """Frozen training statistics must exist."""

    assert TRAINING_STATS_PATH.is_file(), (
        "Training statistics file is missing: "
        f"{TRAINING_STATS_PATH}"
    )


def test_training_stats_have_six_values() -> None:
    """Frozen statistics must match the six model features."""

    normaliser = _load_normaliser()

    mean = _normaliser_array(
        normaliser,
        "mean",
        "mean_",
    )

    std = _normaliser_array(
        normaliser,
        "std",
        "std_",
    )

    assert mean.shape == (EXPECTED_FEATURE_COUNT,)
    assert std.shape == (EXPECTED_FEATURE_COUNT,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    assert np.all(std > 0.0)


def test_normalisation_is_deterministic() -> None:
    """Repeated normalisation must return identical results."""

    normaliser = _load_normaliser()

    sample = np.asarray(
        [
            4.0,
            0.2,
            0.0,
            0.45,
            0.55,
            0.0,
        ],
        dtype=np.float32,
    )

    first_result = np.asarray(
        normaliser.transform(sample),
        dtype=np.float32,
    )

    second_result = np.asarray(
        normaliser.transform(sample),
        dtype=np.float32,
    )

    assert first_result.shape == (EXPECTED_FEATURE_COUNT,)
    assert np.isfinite(first_result).all()

    np.testing.assert_array_equal(
        first_result,
        second_result,
    )


def test_normalisation_matches_formula() -> None:
    """Normalisation must implement the frozen z-score formula."""

    normaliser = _load_normaliser()

    sample = np.asarray(
        [
            4.0,
            0.2,
            0.0,
            0.45,
            0.55,
            0.0,
        ],
        dtype=np.float32,
    )

    transformed = np.asarray(
        normaliser.transform(sample),
        dtype=np.float32,
    )

    mean = _normaliser_array(
        normaliser,
        "mean",
        "mean_",
    )

    std = _normaliser_array(
        normaliser,
        "std",
        "std_",
    )

    expected = (
        sample - mean
    ) / std

    np.testing.assert_allclose(
        transformed,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_normaliser_does_not_modify_input() -> None:
    """Normalisation must not modify the input array."""

    normaliser = _load_normaliser()

    sample = np.asarray(
        [
            4.0,
            0.2,
            0.0,
            0.45,
            0.55,
            0.0,
        ],
        dtype=np.float32,
    )

    original = sample.copy()

    normaliser.transform(sample)

    np.testing.assert_array_equal(
        sample,
        original,
    )
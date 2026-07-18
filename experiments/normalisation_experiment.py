"""
Normalisation sensitivity experiment for LogiEdge.

The experiment:

1. Loads the frozen held-out validation split.
2. Loads the frozen training mean and standard deviation.
3. Evaluates the M3 TFLite model using the correct statistics.
4. Shifts the mean by +3 standard deviations.
5. Evaluates the same validation samples again.
6. Saves accuracy and Critical-class recall for both conditions.

Output:
    experiments/normalisation_experiment.csv
"""

"""
Normalisation sensitivity experiment for LogiEdge.
"""


import os
import warnings

# These settings must be applied before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings(
    "ignore",
    message=r".*tf\.lite\.Interpreter is deprecated.*",
    category=UserWarning,
)

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET_CANDIDATES = [
    ROOT / "training" / "dataset.npz",
    ROOT / "training" / "assignment_dataset.npz",
    ROOT / "data_pipeline" / "dataset.npz",
]

STATS_PATH = ROOT / "data_pipeline" / "training_stats.npy"

MODEL_CANDIDATES = [
    ROOT / "training" / "models" / "m3_pruned_int8.tflite",
    ROOT / "training" / "models" / "m2_ptq_int8.tflite",
    ROOT / "training" / "models" / "m1_fp32.tflite",
]

OUTPUT_PATH = ROOT / "experiments" / "normalisation_experiment.csv"

CONFUSION_OUTPUT_PATH = (
    ROOT / "experiments" / "normalisation_confusion_matrices.npz"
)

EXPECTED_FEATURES = 6
CRITICAL_CLASS = 2


# ---------------------------------------------------------------------------
# General validation helpers
# ---------------------------------------------------------------------------

def require_file(path: Path, description: str) -> Path:
    """Return a path when it exists or raise a clear error."""

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )

    return path


def first_existing_path(
    candidates: list[Path],
    description: str,
) -> Path:
    """Return the first existing file from a list of candidates."""

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    formatted = "\n".join(f"  - {path}" for path in candidates)

    raise FileNotFoundError(
        f"{description} was not found. Checked:\n{formatted}"
    )


def ensure_feature_matrix(
    values: np.ndarray,
    description: str,
) -> np.ndarray:
    """
    Convert a feature array to a two-dimensional float32 matrix.

    Expected final shape:
        (number_of_samples, number_of_features)
    """

    array = np.asarray(values)

    if array.ndim == 1:
        if array.size % EXPECTED_FEATURES != 0:
            raise ValueError(
                f"{description} is one-dimensional with {array.size} "
                f"values and cannot be reshaped into "
                f"{EXPECTED_FEATURES} features."
            )

        array = array.reshape(-1, EXPECTED_FEATURES)

    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)

    if array.ndim != 2:
        raise ValueError(
            f"{description} must be two-dimensional after conversion. "
            f"Received shape: {array.shape}"
        )

    if array.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{description} must contain exactly "
            f"{EXPECTED_FEATURES} features per sample. "
            f"Received shape: {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{description} contains NaN or infinite values."
        )

    return array.astype(np.float32, copy=False)


def ensure_labels(
    values: np.ndarray,
    expected_samples: int,
    description: str,
) -> np.ndarray:
    """Return a one-dimensional integer class-label vector."""

    labels = np.asarray(values)

    if labels.ndim == 2 and labels.shape[1] > 1:
        # Support one-hot encoded labels.
        labels = np.argmax(labels, axis=1)

    labels = labels.reshape(-1)

    if labels.shape[0] != expected_samples:
        raise ValueError(
            f"{description} contains {labels.shape[0]} labels, but "
            f"{expected_samples} feature rows were loaded."
        )

    if not np.all(np.isfinite(labels)):
        raise ValueError(
            f"{description} contains NaN or infinite values."
        )

    return labels.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def pick_key(
    archive: np.lib.npyio.NpzFile,
    candidate_keys: list[str],
) -> str | None:
    """Return the first matching key from an NPZ archive."""

    available = set(archive.files)

    for key in candidate_keys:
        if key in available:
            return key

    return None


def load_validation_data(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the raw held-out validation feature matrix and labels.

    Supported layouts include:

    1. Explicit validation arrays:
       x_val_raw / y_val
       X_val_raw / y_val
       x_validation / y_validation
       X_val / y_val

    2. Full arrays with validation indices:
       X_raw / y / val_idx
       features / labels / validation_indices

    The experiment intentionally requires raw, unnormalised validation
    features. It must not normalise an already normalised validation set.
    """

    print(f"[NORM] Loading dataset: {dataset_path}")

    with np.load(
        dataset_path,
        allow_pickle=True,
    ) as archive:
        print(
            "[NORM] Dataset keys: "
            + ", ".join(sorted(archive.files))
        )

        explicit_feature_key = pick_key(
            archive,
            [
                "x_val_raw",
                "X_val_raw",
                "x_validation_raw",
                "X_validation_raw",
                "val_features_raw",
                "validation_features_raw",
                "x_val",
                "X_val",
                "x_validation",
                "X_validation",
            ],
        )

        explicit_label_key = pick_key(
            archive,
            [
                "y_val",
                "Y_val",
                "y_validation",
                "Y_validation",
                "val_labels",
                "validation_labels",
            ],
        )

        if (
            explicit_feature_key is not None
            and explicit_label_key is not None
        ):
            features = ensure_feature_matrix(
                archive[explicit_feature_key],
                f"Validation features '{explicit_feature_key}'",
            )

            labels = ensure_labels(
                archive[explicit_label_key],
                expected_samples=features.shape[0],
                description=(
                    f"Validation labels '{explicit_label_key}'"
                ),
            )

            print(
                "[NORM] Using explicit validation arrays: "
                f"{explicit_feature_key}, {explicit_label_key}"
            )

            return features, labels

        full_feature_key = pick_key(
            archive,
            [
                "x_raw",
                "X_raw",
                "features_raw",
                "raw_features",
                "X",
                "x",
                "features",
            ],
        )

        full_label_key = pick_key(
            archive,
            [
                "y",
                "Y",
                "labels",
                "targets",
                "target",
            ],
        )

        validation_index_key = pick_key(
            archive,
            [
                "val_idx",
                "validation_idx",
                "validation_indices",
                "val_indices",
                "test_idx",
                "test_indices",
            ],
        )

        if (
            full_feature_key is None
            or full_label_key is None
            or validation_index_key is None
        ):
            raise KeyError(
                "Unable to locate the held-out validation data.\n"
                "The dataset must contain either:\n"
                "  x_val_raw and y_val\n"
                "or:\n"
                "  raw feature array, label array and validation "
                "indices.\n"
                f"Available keys: {archive.files}"
            )

        all_features = ensure_feature_matrix(
            archive[full_feature_key],
            f"Full features '{full_feature_key}'",
        )

        all_labels = ensure_labels(
            archive[full_label_key],
            expected_samples=all_features.shape[0],
            description=f"Full labels '{full_label_key}'",
        )

        validation_indices = np.asarray(
            archive[validation_index_key],
        ).reshape(-1)

        if validation_indices.size == 0:
            raise ValueError(
                "The validation index array is empty."
            )

        if not np.issubdtype(
            validation_indices.dtype,
            np.integer,
        ):
            validation_indices = validation_indices.astype(np.int64)

        if np.any(validation_indices < 0):
            raise ValueError(
                "Validation indices contain negative values."
            )

        if np.any(validation_indices >= all_features.shape[0]):
            raise IndexError(
                "Validation indices exceed the dataset size."
            )

        print(
            "[NORM] Using grouped validation split: "
            f"{full_feature_key}, {full_label_key}, "
            f"{validation_index_key}"
        )

        return (
            all_features[validation_indices],
            all_labels[validation_indices],
        )


# ---------------------------------------------------------------------------
# Frozen statistics loading
# ---------------------------------------------------------------------------

def unpack_object_mapping(
    loaded: np.ndarray,
) -> dict[str, Any] | None:
    """Extract a mapping stored inside a NumPy object array."""

    if loaded.dtype != object:
        return None

    try:
        value = loaded.item()
    except (ValueError, AttributeError):
        return None

    if isinstance(value, dict):
        return value

    return None


def load_training_stats(
    stats_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load frozen mean and standard deviation.

    Supported training_stats.npy layouts:

    1. Dictionary:
       {"mean": ..., "std": ...}

    2. Dictionary:
       {"feature_mean": ..., "feature_std": ...}

    3. Numeric matrix:
       shape (2, 6), where row 0 is mean and row 1 is std

    4. Numeric matrix:
       shape (6, 2), where column 0 is mean and column 1 is std
    """

    print(f"[NORM] Loading frozen statistics: {stats_path}")

    loaded = np.load(
        stats_path,
        allow_pickle=True,
    )

    mapping = unpack_object_mapping(loaded)

    if mapping is not None:
        mean_key = next(
            (
                key
                for key in (
                    "mean",
                    "feature_mean",
                    "training_mean",
                    "mu",
                )
                if key in mapping
            ),
            None,
        )

        std_key = next(
            (
                key
                for key in (
                    "std",
                    "feature_std",
                    "training_std",
                    "sigma",
                    "scale",
                )
                if key in mapping
            ),
            None,
        )

        if mean_key is None or std_key is None:
            raise KeyError(
                "training_stats.npy contains a dictionary but does "
                "not contain recognised mean and standard-deviation "
                f"keys. Keys found: {list(mapping)}"
            )

        feature_mean = np.asarray(
            mapping[mean_key],
            dtype=np.float32,
        ).reshape(-1)

        feature_std = np.asarray(
            mapping[std_key],
            dtype=np.float32,
        ).reshape(-1)

    else:
        numeric = np.asarray(
            loaded,
            dtype=np.float32,
        )

        if numeric.shape == (2, EXPECTED_FEATURES):
            feature_mean = numeric[0]
            feature_std = numeric[1]

        elif numeric.shape == (EXPECTED_FEATURES, 2):
            feature_mean = numeric[:, 0]
            feature_std = numeric[:, 1]

        elif numeric.ndim == 1 and numeric.size == (
            EXPECTED_FEATURES * 2
        ):
            numeric = numeric.reshape(2, EXPECTED_FEATURES)
            feature_mean = numeric[0]
            feature_std = numeric[1]

        else:
            raise ValueError(
                "Unsupported training_stats.npy structure. "
                "Expected a mean/std dictionary or an array with "
                f"shape (2, {EXPECTED_FEATURES}) or "
                f"({EXPECTED_FEATURES}, 2). "
                f"Received shape: {numeric.shape}"
            )

    if feature_mean.shape != (EXPECTED_FEATURES,):
        raise ValueError(
            "Frozen mean must contain exactly "
            f"{EXPECTED_FEATURES} values. "
            f"Received shape: {feature_mean.shape}"
        )

    if feature_std.shape != (EXPECTED_FEATURES,):
        raise ValueError(
            "Frozen standard deviation must contain exactly "
            f"{EXPECTED_FEATURES} values. "
            f"Received shape: {feature_std.shape}"
        )

    if not np.all(np.isfinite(feature_mean)):
        raise ValueError(
            "Frozen mean contains NaN or infinite values."
        )

    if not np.all(np.isfinite(feature_std)):
        raise ValueError(
            "Frozen standard deviation contains NaN or infinite "
            "values."
        )

    if np.any(feature_std < 0):
        raise ValueError(
            "Frozen standard deviation contains negative values."
        )

    print(
        "[NORM] Frozen mean: "
        + np.array2string(
            feature_mean,
            precision=6,
            separator=", ",
        )
    )

    print(
        "[NORM] Frozen std : "
        + np.array2string(
            feature_std,
            precision=6,
            separator=", ",
        )
    )

    return feature_mean, feature_std


# ---------------------------------------------------------------------------
# Normalisation and metrics
# ---------------------------------------------------------------------------

def normalise(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalise raw features using fixed training statistics."""

    safe_std = np.where(std == 0, 1.0, std)

    normalised = (features - mean) / safe_std

    if not np.all(np.isfinite(normalised)):
        raise ValueError(
            "Normalisation produced NaN or infinite values."
        )

    return normalised.astype(np.float32, copy=False)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Calculate accuracy and Critical-class recall."""

    accuracy = accuracy_score(
        y_true,
        y_pred,
    ) * 100.0

    critical_recall = recall_score(
        y_true,
        y_pred,
        labels=[CRITICAL_CLASS],
        average=None,
        zero_division=0,
    )[0] * 100.0

    return float(accuracy), float(critical_recall)


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def quantise_input(
    values: np.ndarray,
    dtype: np.dtype,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Convert float32 model inputs to a quantised TFLite dtype."""

    if dtype not in (np.int8, np.uint8):
        return values.astype(dtype, copy=False)

    if scale <= 0:
        raise ValueError(
            "The TFLite input tensor is quantised but its scale is "
            f"invalid: {scale}"
        )

    quantised = np.round(values / scale + zero_point)

    limits = np.iinfo(dtype)

    quantised = np.clip(
        quantised,
        limits.min,
        limits.max,
    )

    return quantised.astype(dtype)


def dequantise_output(
    values: np.ndarray,
    dtype: np.dtype,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Convert a quantised TFLite output tensor to float32."""

    if dtype not in (np.int8, np.uint8):
        return values.astype(np.float32, copy=False)

    if scale <= 0:
        raise ValueError(
            "The TFLite output tensor is quantised but its scale is "
            f"invalid: {scale}"
        )

    return (
        values.astype(np.float32) - float(zero_point)
    ) * float(scale)


class TFLiteClassifier:
    """Small wrapper around a single-input TFLite classifier."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
        )

        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        if len(input_details) != 1:
            raise ValueError(
                "Expected exactly one model input tensor. "
                f"Found {len(input_details)}."
            )

        if len(output_details) != 1:
            raise ValueError(
                "Expected exactly one model output tensor. "
                f"Found {len(output_details)}."
            )

        self.input_detail = input_details[0]
        self.output_detail = output_details[0]

        self.input_index = int(
            self.input_detail["index"]
        )

        self.output_index = int(
            self.output_detail["index"]
        )

        self.input_dtype = np.dtype(
            self.input_detail["dtype"]
        )

        self.output_dtype = np.dtype(
            self.output_detail["dtype"]
        )

        self.input_scale, self.input_zero_point = (
            self.input_detail.get(
                "quantization",
                (0.0, 0),
            )
        )

        self.output_scale, self.output_zero_point = (
            self.output_detail.get(
                "quantization",
                (0.0, 0),
            )
        )

        input_shape = tuple(
            int(value)
            for value in self.input_detail["shape"]
        )

        if len(input_shape) != 2:
            raise ValueError(
                "Expected model input shape [batch, features]. "
                f"Received: {input_shape}"
            )

        model_features = input_shape[-1]

        if model_features != EXPECTED_FEATURES:
            raise ValueError(
                f"The TFLite model expects {model_features} "
                f"features, but the experiment requires "
                f"{EXPECTED_FEATURES}."
            )

        print(f"[NORM] Model: {model_path}")
        print(
            "[NORM] Input tensor: "
            f"shape={input_shape}, dtype={self.input_dtype}, "
            f"scale={self.input_scale}, "
            f"zero_point={self.input_zero_point}"
        )

        print(
            "[NORM] Output tensor: "
            f"dtype={self.output_dtype}, "
            f"scale={self.output_scale}, "
            f"zero_point={self.output_zero_point}"
        )

    def predict(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return predicted classes and dequantised model outputs.

        The function invokes the interpreter once per validation sample.
        """

        feature_matrix = ensure_feature_matrix(
            features,
            "Model input features",
        )

        predictions: list[int] = []
        outputs: list[np.ndarray] = []

        for row in feature_matrix:
            model_input = row.reshape(1, -1).astype(
                np.float32,
            )

            model_input = quantise_input(
                model_input,
                dtype=self.input_dtype,
                scale=float(self.input_scale),
                zero_point=int(self.input_zero_point),
            )

            self.interpreter.set_tensor(
                self.input_index,
                model_input,
            )

            self.interpreter.invoke()

            raw_output = self.interpreter.get_tensor(
                self.output_index,
            )

            output = dequantise_output(
                raw_output,
                dtype=self.output_dtype,
                scale=float(self.output_scale),
                zero_point=int(self.output_zero_point),
            )

            flattened = np.asarray(
                output,
                dtype=np.float32,
            ).reshape(-1)

            if flattened.size == 1:
                # Defensive support for a scalar binary classifier.
                predicted_class = int(
                    flattened[0] >= 0.5
                )
            else:
                predicted_class = int(
                    np.argmax(flattened)
                )

            predictions.append(predicted_class)
            outputs.append(flattened)

        try:
            output_matrix = np.vstack(outputs)
        except ValueError:
            output_matrix = np.asarray(
                outputs,
                dtype=object,
            )

        return (
            np.asarray(predictions, dtype=np.int64),
            output_matrix,
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(
    rows: list[dict[str, str | float | int]],
) -> None:
    """Write the experiment results to CSV."""

    if not rows:
        raise ValueError(
            "No normalisation experiment results were generated."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "condition",
        "mean_shift_sigma",
        "validation_samples",
        "accuracy_pct",
        "recall_critical_pct",
        "accuracy_change_points",
        "critical_recall_change_points",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Execute the normalisation sensitivity experiment."""

    dataset_path = first_existing_path(
        DATASET_CANDIDATES,
        "Validation dataset",
    )

    model_path = first_existing_path(
        MODEL_CANDIDATES,
        "TFLite model",
    )

    require_file(
        STATS_PATH,
        "Frozen training statistics",
    )

    x_validation_raw, y_validation = (
        load_validation_data(dataset_path)
    )

    feature_mean, feature_std = (
        load_training_stats(STATS_PATH)
    )

    print(
        "[NORM] Held-out validation shape: "
        f"X={x_validation_raw.shape}, "
        f"y={y_validation.shape}"
    )

    class_values, class_counts = np.unique(
        y_validation,
        return_counts=True,
    )

    print(
        "[NORM] Validation class distribution: "
        + ", ".join(
            f"{int(label)}={int(count)}"
            for label, count in zip(
                class_values,
                class_counts,
                strict=True,
            )
        )
    )

    if CRITICAL_CLASS not in class_values:
        raise ValueError(
            "The held-out validation split contains no Critical "
            f"class ({CRITICAL_CLASS}) samples. Critical recall "
            "cannot be evaluated."
        )

    classifier = TFLiteClassifier(model_path)

    # ---------------------------------------------------------------------
    # Condition 1: correct frozen training statistics
    # ---------------------------------------------------------------------

    x_correct = normalise(
        x_validation_raw,
        feature_mean,
        feature_std,
    )

    correct_predictions, _ = classifier.predict(
        x_correct,
    )

    correct_accuracy, correct_recall = evaluate(
        y_validation,
        correct_predictions,
    )

    correct_confusion = confusion_matrix(
        y_validation,
        correct_predictions,
        labels=[0, 1, 2],
    )

    # ---------------------------------------------------------------------
    # Condition 2: mean shifted by +3 standard deviations
    # ---------------------------------------------------------------------

    shifted_mean = feature_mean + (
        3.0 * feature_std
    )

    x_shifted = normalise(
        x_validation_raw,
        shifted_mean,
        feature_std,
    )

    shifted_predictions, _ = classifier.predict(
        x_shifted,
    )

    shifted_accuracy, shifted_recall = evaluate(
        y_validation,
        shifted_predictions,
    )

    shifted_confusion = confusion_matrix(
        y_validation,
        shifted_predictions,
        labels=[0, 1, 2],
    )

    accuracy_change = (
        shifted_accuracy - correct_accuracy
    )

    recall_change = (
        shifted_recall - correct_recall
    )

    rows: list[dict[str, str | float | int]] = [
        {
            "condition": "correct_stats",
            "mean_shift_sigma": 0.0,
            "validation_samples": int(
                y_validation.shape[0]
            ),
            "accuracy_pct": round(
                correct_accuracy,
                4,
            ),
            "recall_critical_pct": round(
                correct_recall,
                4,
            ),
            "accuracy_change_points": 0.0,
            "critical_recall_change_points": 0.0,
        },
        {
            "condition": "shifted_3sigma",
            "mean_shift_sigma": 3.0,
            "validation_samples": int(
                y_validation.shape[0]
            ),
            "accuracy_pct": round(
                shifted_accuracy,
                4,
            ),
            "recall_critical_pct": round(
                shifted_recall,
                4,
            ),
            "accuracy_change_points": round(
                accuracy_change,
                4,
            ),
            "critical_recall_change_points": round(
                recall_change,
                4,
            ),
        },
    ]

    save_results(rows)

    np.savez(
        CONFUSION_OUTPUT_PATH,
        labels=np.asarray([0, 1, 2], dtype=np.int64),
        correct_stats=correct_confusion,
        shifted_3sigma=shifted_confusion,
    )

    print()
    print("=" * 72)
    print("NORMALISATION SENSITIVITY EXPERIMENT")
    print("=" * 72)

    print(
        "Correct statistics:"
        f" accuracy={correct_accuracy:.2f}%,"
        f" Critical recall={correct_recall:.2f}%"
    )

    print(
        "Shifted mean (+3 sigma):"
        f" accuracy={shifted_accuracy:.2f}%,"
        f" Critical recall={shifted_recall:.2f}%"
    )

    print(
        "Accuracy change:"
        f" {accuracy_change:+.2f} percentage points"
    )

    print(
        "Critical recall change:"
        f" {recall_change:+.2f} percentage points"
    )

    print()
    print("Correct-statistics confusion matrix:")
    print(correct_confusion)

    print()
    print("Shifted-3sigma confusion matrix:")
    print(shifted_confusion)

    print()
    print(f"CSV saved to: {OUTPUT_PATH}")
    print(
        "Confusion matrices saved to: "
        f"{CONFUSION_OUTPUT_PATH}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"[NORM][ERROR] {exc}",
            file=sys.stderr,
        )
        raise

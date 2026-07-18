"""Train M1 with a leakage-safe truck-grouped validation holdout."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TensorFlow logging configuration
# These variables must be configured before importing TensorFlow.
# ---------------------------------------------------------------------------
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import contextlib
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

# Suppress TensorFlow Python-level logs.
tf.get_logger().setLevel("ERROR")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))

from preprocessing import CLASS_NAMES, N_FEATURES, Normaliser  # noqa: E402

MODELS = ROOT / "training" / "models"
DATASET_PATH = ROOT / "training" / "dataset.npz"
STATS_PATH = ROOT / "data_pipeline" / "training_stats.npy"

ACC_GATE = 0.88
SEED = 42
EPOCHS = 160
BATCH_SIZE = 32


@contextlib.contextmanager
def suppress_native_stderr() -> Iterator[None]:
    """
    Temporarily suppress native C/C++ messages written directly to stderr.

    TensorFlow Lite conversion can print messages such as:
        Ignored output_format.
        Ignored drop_control_dependency.
        MLIR V1 optimization pass is not enabled.

    These messages bypass Python's normal logging system, so stderr must be
    redirected at the operating-system file-descriptor level.
    """
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


def load_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    list[int],
]:
    """Load and normalize data while checking for truck-group leakage."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run: python .\\training\\generate_dataset.py"
        )

    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Training statistics not found: {STATS_PATH}\n"
            "Run: python .\\training\\generate_dataset.py"
        )

    with np.load(DATASET_PATH) as data:
        required_keys = {
            "X_raw",
            "y",
            "group_id",
            "train_idx",
            "val_idx",
        }

        missing_keys = required_keys.difference(data.files)

        if missing_keys:
            missing_text = ", ".join(sorted(missing_keys))
            raise RuntimeError(
                f"Dataset is missing required fields: {missing_text}. "
                "Re-run generate_dataset.py."
            )

        x_raw = data["X_raw"]
        y = data["y"]
        group_id = data["group_id"]
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]

    train_groups = set(group_id[train_idx].tolist())
    val_groups = set(group_id[val_idx].tolist())
    overlapping_groups = train_groups.intersection(val_groups)

    if overlapping_groups:
        raise RuntimeError(
            "Train/validation truck leakage detected. "
            f"Overlapping groups: {sorted(overlapping_groups)}"
        )

    normaliser = Normaliser.load(STATS_PATH)
    x_normalized = normaliser.transform(x_raw)

    x_train = x_normalized[train_idx].astype(np.float32)
    y_train = y[train_idx].astype(np.int32)
    x_validation = x_normalized[val_idx].astype(np.float32)
    y_validation = y[val_idx].astype(np.int32)

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        sorted(train_groups),
        sorted(val_groups),
    )


def build_model() -> tf.keras.Model:
    """Create the M1 fully connected neural network."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(N_FEATURES,),
                name="features",
            ),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                name="d1",
            ),
            tf.keras.layers.Dense(
                16,
                activation="relu",
                name="d2",
            ),
            tf.keras.layers.Dense(
                len(CLASS_NAMES),
                activation="softmax",
                name="out",
            ),
        ],
        name="logibridge_mlp",
    )

    return model


def class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], list[list[int]]]:
    """Calculate recall, precision, and confusion matrix by class."""
    recall: dict[str, float] = {}
    precision: dict[str, float] = {}

    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(
            ((y_true == class_index) & (y_pred == class_index)).sum()
        )
        false_negative = int(
            ((y_true == class_index) & (y_pred != class_index)).sum()
        )
        false_positive = int(
            ((y_true != class_index) & (y_pred == class_index)).sum()
        )

        recall[class_name] = true_positive / max(
            1,
            true_positive + false_negative,
        )

        precision[class_name] = true_positive / max(
            1,
            true_positive + false_positive,
        )

    confusion_matrix = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(CLASS_NAMES),
    ).numpy()

    return recall, precision, confusion_matrix.tolist()


def convert_to_tflite(model: tf.keras.Model) -> bytes:
    """Convert the trained Keras model to TensorFlow Lite silently."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Keep the model in standard FP32 format.
    converter.optimizations = []

    # TensorFlow emits several harmless native warnings here.
    with suppress_native_stderr():
        tflite_model = converter.convert()

    return tflite_model


def save_artifacts(
    model: tf.keras.Model,
    tflite_model: bytes,
    metrics: dict,
) -> None:
    """Save Keras, TensorFlow Lite, and metrics artifacts."""
    MODELS.mkdir(parents=True, exist_ok=True)

    keras_path = MODELS / "m1_fp32.keras"
    tflite_path = MODELS / "m1_fp32.tflite"
    metrics_path = MODELS / "m1_metrics.json"

    model.save(keras_path, overwrite=True)
    tflite_path.write_bytes(tflite_model)
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"[SAVE] Keras model: {keras_path}")
    print(f"[SAVE] TFLite model: {tflite_path}")
    print(f"[SAVE] Metrics: {metrics_path}")


def main() -> int:
    """Train, evaluate, convert, and save the M1 model."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_groups,
        validation_groups,
    ) = load_data()

    print(
        f"[TRAIN] train trucks={train_groups}; "
        f"validation trucks={validation_groups}; overlap=0"
    )

    print(
        f"[TRAIN] train windows={len(x_train)}; "
        f"validation windows={len(x_validation)}"
    )

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            mode="max",
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=12,
            factor=0.5,
            min_lr=1e-5,
            verbose=0,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=callbacks,
        shuffle=True,
    )

    validation_loss, validation_accuracy = model.evaluate(
        x_validation,
        y_validation,
        verbose=0,
    )

    probabilities = model.predict(
        x_validation,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    recall, precision, confusion = class_metrics(
        y_validation,
        predictions,
    )

    tflite_model = convert_to_tflite(model)

    accuracy_gate_passed = validation_accuracy > ACC_GATE

    metrics = {
        "variant": "M1_FP32",
        "validation_method": "grouped_by_simulated_truck",
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "group_overlap": [],
        "train_windows": int(len(x_train)),
        "validation_windows": int(len(x_validation)),
        "val_accuracy": float(validation_accuracy),
        "val_loss": float(validation_loss),
        "recall": {
            key: float(value)
            for key, value in recall.items()
        },
        "precision": {
            key: float(value)
            for key, value in precision.items()
        },
        "confusion": confusion,
        "params": int(model.count_params()),
        "tflite_bytes": int(len(tflite_model)),
        "epochs_requested": EPOCHS,
        "epochs_completed": int(len(history.history["loss"])),
        "batch_size": BATCH_SIZE,
        "random_seed": SEED,
        "accuracy_gate": ACC_GATE,
        "accuracy_gate_passed": bool(accuracy_gate_passed),
    }

    save_artifacts(
        model=model,
        tflite_model=tflite_model,
        metrics=metrics,
    )

    print(
        f"[TRAIN] grouped validation accuracy="
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        "[TRAIN] recall "
        + " ".join(
            f"{class_name}={value * 100:.1f}%"
            for class_name, value in recall.items()
        )
    )

    print(
        "[TRAIN] precision "
        + " ".join(
            f"{class_name}={value * 100:.1f}%"
            for class_name, value in precision.items()
        )
    )

    if not accuracy_gate_passed:
        print(
            f"[FAIL] accuracy {validation_accuracy:.4f} "
            f"did not exceed gate {ACC_GATE:.4f}"
        )
        return 1

    print("[PASS] grouped validation accuracy gate cleared")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted by user.")
        raise SystemExit(130)
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        raise SystemExit(1)
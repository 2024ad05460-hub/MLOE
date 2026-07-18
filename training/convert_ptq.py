"""Create M2 using full INT8 post-training quantisation.

The representative calibration dataset is selected only from the grouped
training split, preventing validation-data leakage.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Logging configuration
# These environment variables must be set before TensorFlow is imported.
# ---------------------------------------------------------------------------
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import contextlib
import json
import sys
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

# Suppress TensorFlow Python-level logging.
tf.get_logger().setLevel("ERROR")

# Suppress known harmless TensorFlow/LiteRT UserWarnings.
warnings.filterwarnings(
    "ignore",
    message=r".*Statistics for quantized inputs were expected.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.lite\.Interpreter is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*LiteRT interpreter.*",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PIPELINE_DIR = ROOT / "data_pipeline"
OPTIMISATION_DIR = ROOT / "optimisation"

sys.path.insert(0, str(DATA_PIPELINE_DIR))
sys.path.insert(0, str(OPTIMISATION_DIR))

from preprocessing import Normaliser  # noqa: E402

MODELS = ROOT / "training" / "models"
DATASET_PATH = ROOT / "training" / "dataset.npz"
STATS_PATH = DATA_PIPELINE_DIR / "training_stats.npy"
M1_MODEL_PATH = MODELS / "m1_fp32.keras"
M2_MODEL_PATH = MODELS / "m2_ptq_int8.tflite"
M2_METRICS_PATH = MODELS / "m2_metrics.json"

N_CALIB = 250
CALIBRATION_SEED = 0


@contextlib.contextmanager
def suppress_native_output() -> Iterator[None]:
    """Temporarily suppress Python and native C/C++ stdout and stderr.

    TensorFlow Lite conversion can emit harmless messages directly through
    native file descriptors, bypassing Python logging. This context manager
    suppresses those messages only during conversion or TFLite evaluation.
    """
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    saved_stdout_fd: int | None = None
    saved_stderr_fd: int | None = None

    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()

        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)

        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

    finally:
        if saved_stdout_fd is not None and stdout_fd is not None:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)

        if saved_stderr_fd is not None and stderr_fd is not None:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)


def validate_required_files() -> None:
    """Confirm that all required input artifacts are available."""
    required_files = [
        DATASET_PATH,
        STATS_PATH,
        M1_MODEL_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_text = "\n".join(f"  - {path}" for path in missing_files)

        raise FileNotFoundError(
            "Required files are missing:\n"
            f"{missing_text}\n\n"
            "Run these commands first:\n"
            "  python .\\training\\generate_dataset.py\n"
            "  python .\\training\\train_model.py"
        )


def load_split() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    list[int],
]:
    """Load normalized training and grouped validation datasets."""
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
        labels = data["y"]
        group_ids = data["group_id"]
        train_indices = data["train_idx"]
        validation_indices = data["val_idx"]

    train_groups = set(group_ids[train_indices].tolist())
    validation_groups = set(group_ids[validation_indices].tolist())
    group_overlap = train_groups.intersection(validation_groups)

    if group_overlap:
        raise RuntimeError(
            "Training/validation truck leakage detected. "
            f"Overlapping groups: {sorted(group_overlap)}"
        )

    normaliser = Normaliser.load(STATS_PATH)
    normalized_features = normaliser.transform(x_raw).astype(np.float32)

    x_train = normalized_features[train_indices]
    y_train = labels[train_indices].astype(np.int32)

    x_validation = normalized_features[validation_indices]
    y_validation = labels[validation_indices].astype(np.int32)

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        sorted(train_groups),
        sorted(validation_groups),
    )


def calibration_indices(
    labels: np.ndarray,
    sample_count: int,
    seed: int = CALIBRATION_SEED,
) -> np.ndarray:
    """Create a class-balanced calibration sample from training data only."""
    if labels.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional labels, received shape {labels.shape}."
        )

    if len(labels) == 0:
        raise ValueError("Training labels are empty.")

    sample_count = min(sample_count, len(labels))

    if sample_count <= 0:
        raise ValueError("Calibration sample count must be greater than zero.")

    random_generator = np.random.default_rng(seed)
    selected_indices: list[int] = []

    classes = np.unique(labels)
    quota_per_class = max(1, sample_count // len(classes))

    # Select a balanced sample from each class.
    for class_id in classes:
        class_pool = np.flatnonzero(labels == class_id)

        class_sample_count = min(
            quota_per_class,
            len(class_pool),
        )

        chosen = random_generator.choice(
            class_pool,
            size=class_sample_count,
            replace=False,
        )

        selected_indices.extend(chosen.tolist())

    # Fill any remaining positions from unused training samples.
    remaining_count = sample_count - len(selected_indices)

    if remaining_count > 0:
        all_indices = np.arange(len(labels), dtype=np.int64)

        remaining_pool = np.setdiff1d(
            all_indices,
            np.asarray(selected_indices, dtype=np.int64),
            assume_unique=False,
        )

        fill_count = min(
            remaining_count,
            len(remaining_pool),
        )

        additional_indices = random_generator.choice(
            remaining_pool,
            size=fill_count,
            replace=False,
        )

        selected_indices.extend(additional_indices.tolist())

    random_generator.shuffle(selected_indices)

    return np.asarray(
        selected_indices[:sample_count],
        dtype=np.int64,
    )


def quantise(
    keras_path: Path,
    output_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[bytes, np.ndarray]:
    """Convert the FP32 Keras model to a fully quantized INT8 model."""
    model = tf.keras.models.load_model(
        keras_path,
        compile=False,
    )

    calibration = calibration_indices(
        labels=y_train,
        sample_count=min(N_CALIB, len(x_train)),
    )

    def representative_dataset():
        """Yield class-balanced FP32 calibration samples."""
        for index in calibration:
            sample = x_train[index : index + 1].astype(
                np.float32,
                copy=False,
            )

            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable full post-training integer quantisation.
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT,
    ]

    converter.representative_dataset = representative_dataset

    # Require only INT8-supported TensorFlow Lite operations.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    # Set both model interface tensors to INT8.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Suppress only harmless conversion output.
    with suppress_native_output():
        model_bytes = converter.convert()

    if not model_bytes:
        raise RuntimeError("TensorFlow Lite conversion produced an empty model.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(model_bytes)

    return model_bytes, calibration


def evaluate_tflite(
    model_path: Path,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> dict:
    """Evaluate the INT8 TFLite model while suppressing LiteRT warnings."""
    # Import after path configuration and inside this function.
    from tflite_eval import evaluate  # noqa: PLC0415

    # Suppress LiteRT deprecation and XNNPACK initialization messages.
    with suppress_native_output():
        result = evaluate(
            model_path,
            x_validation,
            y_validation,
        )

    required_result_keys = {
        "accuracy",
        "recall",
    }

    missing_result_keys = required_result_keys.difference(result)

    if missing_result_keys:
        missing_text = ", ".join(sorted(missing_result_keys))
        raise RuntimeError(
            f"TFLite evaluation did not return: {missing_text}"
        )

    return result


def main() -> int:
    """Create and validate the M2 full-INT8 PTQ model."""
    validate_required_files()

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_groups,
        validation_groups,
    ) = load_split()

    print(
        f"[PTQ] training trucks={train_groups}; "
        f"validation trucks={validation_groups}; overlap=0"
    )

    print(
        f"[PTQ] training windows={len(x_train)}; "
        f"validation windows={len(x_validation)}"
    )

    model_bytes, calibration = quantise(
        keras_path=M1_MODEL_PATH,
        output_path=M2_MODEL_PATH,
        x_train=x_train,
        y_train=y_train,
    )

    result = evaluate_tflite(
        model_path=M2_MODEL_PATH,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    payload = {
        "variant": "M2_PTQ_INT8",
        "validation_method": "grouped_by_simulated_truck",
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "group_overlap": [],
        "training_windows": int(len(x_train)),
        "validation_windows": int(len(x_validation)),
        "tflite_bytes": int(len(model_bytes)),
        "tflite_kb": float(len(model_bytes) / 1024),
        "calibration_samples": int(len(calibration)),
        "calibration_source": "training_split_only",
        "calibration_seed": CALIBRATION_SEED,
        "input_type": "int8",
        "output_type": "int8",
        **result,
    }

    M2_METRICS_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"[SAVE] INT8 model: {M2_MODEL_PATH}")
    print(f"[SAVE] Metrics: {M2_METRICS_PATH}")

    print(
        f"[PTQ] size={len(model_bytes) / 1024:.2f} KB; "
        f"grouped validation accuracy="
        f"{result['accuracy'] * 100:.2f}%"
    )

    print(
        "[PTQ] recall "
        + " ".join(
            f"{class_name}={value * 100:.1f}%"
            for class_name, value in result["recall"].items()
        )
    )

    print("[PASS] M2 full-INT8 PTQ conversion completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print("\n[STOP] PTQ conversion interrupted by user.")
        raise SystemExit(130)

    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        raise SystemExit(1)
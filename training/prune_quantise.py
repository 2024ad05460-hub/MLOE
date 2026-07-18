"""Create M3 using pruning, structural unit removal, and full INT8 PTQ.

Workflow
--------
1. Load the Keras 3 M1 baseline model.
2. Rebuild the same architecture using legacy tf_keras.
3. Copy every baseline weight into the legacy model.
4. Apply PolynomialDecay magnitude pruning to hidden Dense layers.
5. Strip pruning wrappers.
6. Remove low-importance hidden units structurally.
7. Fine-tune the compact FP32 model.
8. Convert the compact model to full INT8 TensorFlow Lite.
9. Evaluate using the truck-grouped validation holdout.

Calibration samples are selected only from the training split.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Logging configuration
# These values must be set before importing TensorFlow.
# ---------------------------------------------------------------------------
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import contextlib
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Suppress harmless import-time output from legacy TensorFlow/Keras packages.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_native_output() -> Iterator[None]:
    """Temporarily suppress Python and native stdout/stderr output.

    TensorFlow and TensorFlow Lite sometimes write directly to native file
    descriptors instead of Python logging. This context manager is used only
    around operations known to produce harmless framework messages.
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
            sys.stdout.flush()
            sys.stderr.flush()

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


# Import TensorFlow and legacy Keras quietly.
with suppress_native_output():
    import tensorflow as tf
    import keras as keras3
    import tf_keras as keras
    import tensorflow_model_optimization as tfmot

tf.get_logger().setLevel("ERROR")

warnings.filterwarnings(
    "ignore",
    message=r".*sparse_softmax_cross_entropy is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Statistics for quantized inputs were expected.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.lite\.Interpreter is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*LiteRT interpreter.*",
)


# ---------------------------------------------------------------------------
# Project paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_PIPELINE_DIR = ROOT / "data_pipeline"
OPTIMISATION_DIR = ROOT / "optimisation"
TRAINING_DIR = ROOT / "training"
MODELS = TRAINING_DIR / "models"

DATASET_PATH = TRAINING_DIR / "dataset.npz"
STATS_PATH = DATA_PIPELINE_DIR / "training_stats.npy"

M1_MODEL_PATH = MODELS / "m1_fp32.keras"
M3_FP32_PATH = MODELS / "m3_pruned_fp32.keras"
M3_INT8_PATH = MODELS / "m3_pruned_int8.tflite"
M3_METRICS_PATH = MODELS / "m3_metrics.json"
M2_METRICS_PATH = MODELS / "m2_metrics.json"
OPTIMISATION_RESULTS_DIR = OPTIMISATION_DIR / "results"
OPTIMISATION_METADATA_PATH = (
    OPTIMISATION_RESULTS_DIR / "model_optimisation_metadata.json"
)

sys.path.insert(0, str(DATA_PIPELINE_DIR))
sys.path.insert(0, str(OPTIMISATION_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from preprocessing import N_FEATURES, Normaliser  # noqa: E402
from convert_ptq import calibration_indices  # noqa: E402


SPARSITY = 0.35
STRUCTURAL_KEEP_RATIO = 1.0 - SPARSITY
N_CALIB = 250
SEED = 42

PRUNING_EPOCHS = 40
COMPACT_FINE_TUNE_EPOCHS = 40
BATCH_SIZE = 32
PRUNING_LEARNING_RATE = 5e-4
COMPACT_LEARNING_RATE = 5e-4


def validate_required_files() -> None:
    """Check that all input artifacts required by M3 are available."""
    required_files = [
        DATASET_PATH,
        STATS_PATH,
        M1_MODEL_PATH,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "Required M3 input files are missing:\n"
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
    """Load normalized train and truck-grouped validation splits."""
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
            raise RuntimeError(
                "Dataset is missing required fields: "
                f"{', '.join(sorted(missing_keys))}. "
                "Re-run generate_dataset.py."
            )

        x_raw = data["X_raw"]
        labels = data["y"]
        group_ids = data["group_id"]
        train_indices = data["train_idx"]
        validation_indices = data["val_idx"]

    train_groups = set(
        group_ids[train_indices].tolist()
    )
    validation_groups = set(
        group_ids[validation_indices].tolist()
    )

    overlap = train_groups.intersection(validation_groups)

    if overlap:
        raise RuntimeError(
            "Training/validation truck leakage detected. "
            f"Overlapping groups: {sorted(overlap)}"
        )

    normaliser = Normaliser.load(STATS_PATH)
    features = normaliser.transform(x_raw).astype(np.float32)

    x_train = features[train_indices]
    y_train = labels[train_indices].astype(np.int32)

    x_validation = features[validation_indices]
    y_validation = labels[validation_indices].astype(np.int32)

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        sorted(train_groups),
        sorted(validation_groups),
    )


def legacy_baseline() -> Any:
    """Create a legacy tf_keras model and copy all M1 weights into it."""
    with suppress_native_output():
        source_model = keras3.models.load_model(
            M1_MODEL_PATH,
            compile=False,
        )

    model = keras.Sequential(
        [
            keras.layers.Input(
                shape=(N_FEATURES,),
                name="features",
            ),
            keras.layers.Dense(
                32,
                activation="relu",
                name="d1",
            ),
            keras.layers.Dense(
                16,
                activation="relu",
                name="d2",
            ),
            keras.layers.Dense(
                3,
                activation="softmax",
                name="out",
            ),
        ],
        name="logibridge_mlp",
    )

    # Explicitly build the legacy model before copying weights.
    model(
        np.zeros(
            (1, N_FEATURES),
            dtype=np.float32,
        )
    )

    source_weights = source_model.get_weights()
    target_weights = model.get_weights()

    if len(source_weights) != len(target_weights):
        raise RuntimeError(
            "M1 and legacy model weight counts do not match: "
            f"{len(source_weights)} versus {len(target_weights)}."
        )

    for index, (source, target) in enumerate(
        zip(source_weights, target_weights)
    ):
        if source.shape != target.shape:
            raise RuntimeError(
                f"Weight shape mismatch at index {index}: "
                f"M1={source.shape}, legacy={target.shape}."
            )

    model.set_weights(source_weights)

    return model


def copy_baseline_weights(
    baseline_model: Any,
    pruned_model: Any,
) -> None:
    """Copy baseline weights into their corresponding pruning wrappers."""
    pruning_targets: dict[str, Any] = {}

    for layer in pruned_model.layers:
        wrapped_layer = getattr(layer, "layer", None)

        if wrapped_layer is not None:
            pruning_targets[wrapped_layer.name] = layer
        else:
            pruning_targets[layer.name] = layer

    for source_layer in baseline_model.layers:
        source_weights = source_layer.get_weights()

        if not source_weights:
            continue

        target_wrapper = pruning_targets.get(source_layer.name)

        if target_wrapper is None:
            raise RuntimeError(
                "Missing pruning counterpart for baseline layer "
                f"{source_layer.name!r}."
            )

        target_layer = getattr(
            target_wrapper,
            "layer",
            target_wrapper,
        )

        target_weights = target_layer.get_weights()

        if len(source_weights) != len(target_weights):
            raise RuntimeError(
                f"Weight count mismatch for layer {source_layer.name}: "
                f"{len(source_weights)} versus {len(target_weights)}."
            )

        for source_weight, target_weight in zip(
            source_weights,
            target_weights,
        ):
            if source_weight.shape != target_weight.shape:
                raise RuntimeError(
                    f"Weight shape mismatch for layer "
                    f"{source_layer.name}: "
                    f"{source_weight.shape} versus "
                    f"{target_weight.shape}."
                )

        target_layer.set_weights(source_weights)


def calculate_layer_sparsity(
    model: Any,
    layer_name: str,
) -> float:
    """Return kernel sparsity for a Dense layer."""
    kernel = model.get_layer(layer_name).get_weights()[0]
    return float(np.mean(kernel == 0.0))


def sparsify(
    baseline_model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int = PRUNING_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Any:
    """Apply PolynomialDecay pruning and return a stripped model."""
    steps_per_epoch = max(
        1,
        len(x_train) // batch_size,
    )

    end_step = steps_per_epoch * epochs

    pruning_schedule = (
        tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=SPARSITY,
            begin_step=0,
            end_step=end_step,
            power=3.0,
            frequency=100,
        )
    )

    def clone_and_wrap(layer: Any) -> Any:
        """Clone each layer and wrap hidden Dense layers for pruning."""
        cloned_layer = layer.__class__.from_config(
            layer.get_config()
        )

        should_prune = (
            isinstance(cloned_layer, keras.layers.Dense)
            and cloned_layer.name in {"d1", "d2"}
        )

        if should_prune:
            return tfmot.sparsity.keras.prune_low_magnitude(
                cloned_layer,
                pruning_schedule=pruning_schedule,
            )

        return cloned_layer

    pruned_model = keras.models.clone_model(
        baseline_model,
        clone_function=clone_and_wrap,
    )

    pruned_model(
        np.zeros(
            (1, N_FEATURES),
            dtype=np.float32,
        )
    )

    copy_baseline_weights(
        baseline_model=baseline_model,
        pruned_model=pruned_model,
    )

    pruned_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=PRUNING_LEARNING_RATE,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    pruned_model.fit(
        x_train,
        y_train,
        validation_data=(
            x_validation,
            y_validation,
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=True,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
        ],
    )

    stripped_model = tfmot.sparsity.keras.strip_pruning(
        pruned_model
    )

    d1_sparsity = calculate_layer_sparsity(
        stripped_model,
        "d1",
    )
    d2_sparsity = calculate_layer_sparsity(
        stripped_model,
        "d2",
    )

    print(
        f"[PRUNE] d1 unstructured sparsity="
        f"{d1_sparsity * 100:.1f}%"
    )
    print(
        f"[PRUNE] d2 unstructured sparsity="
        f"{d2_sparsity * 100:.1f}%"
    )

    return stripped_model


def select_surviving_units(
    outgoing_kernel: np.ndarray,
    keep_ratio: float,
) -> np.ndarray:
    """Select the most important output units using L2 kernel norm."""
    if outgoing_kernel.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional Dense kernel, "
            f"received shape {outgoing_kernel.shape}."
        )

    unit_scores = np.linalg.norm(
        outgoing_kernel,
        axis=0,
    )

    number_to_keep = max(
        1,
        int(round(outgoing_kernel.shape[1] * keep_ratio)),
    )

    selected = np.argsort(unit_scores)[-number_to_keep:]

    return np.sort(selected).astype(np.int64)


def structurally_prune(
    model: Any,
    keep_ratio: float = STRUCTURAL_KEEP_RATIO,
) -> Any:
    """Remove low-importance hidden units and build a compact model."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(
            f"keep_ratio must be within (0, 1], received {keep_ratio}."
        )

    first_kernel, first_bias = (
        model.get_layer("d1").get_weights()
    )
    second_kernel, second_bias = (
        model.get_layer("d2").get_weights()
    )
    output_kernel, output_bias = (
        model.get_layer("out").get_weights()
    )

    first_survivors = select_surviving_units(
        first_kernel,
        keep_ratio,
    )
    second_survivors = select_surviving_units(
        second_kernel,
        keep_ratio,
    )

    compact_model = keras.Sequential(
        [
            keras.layers.Input(
                shape=(first_kernel.shape[0],),
                name="features",
            ),
            keras.layers.Dense(
                len(first_survivors),
                activation="relu",
                name="d1",
            ),
            keras.layers.Dense(
                len(second_survivors),
                activation="relu",
                name="d2",
            ),
            keras.layers.Dense(
                output_kernel.shape[1],
                activation="softmax",
                name="out",
            ),
        ],
        name="logibridge_mlp_pruned",
    )

    compact_model(
        np.zeros(
            (1, first_kernel.shape[0]),
            dtype=np.float32,
        )
    )

    compact_first_kernel = first_kernel[
        :,
        first_survivors,
    ]

    compact_second_kernel = second_kernel[
        np.ix_(
            first_survivors,
            second_survivors,
        )
    ]

    compact_output_kernel = output_kernel[
        second_survivors,
        :,
    ]

    compact_model.get_layer("d1").set_weights(
        [
            compact_first_kernel,
            first_bias[first_survivors],
        ]
    )

    compact_model.get_layer("d2").set_weights(
        [
            compact_second_kernel,
            second_bias[second_survivors],
        ]
    )

    compact_model.get_layer("out").set_weights(
        [
            compact_output_kernel,
            output_bias,
        ]
    )

    print(
        f"[PRUNE] structural d1 "
        f"{first_kernel.shape[1]} -> {len(first_survivors)}; "
        f"d2 {second_kernel.shape[1]} -> {len(second_survivors)}"
    )

    return compact_model


def select_stratified_calibration_indices(
    labels: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Select deterministic, class-stratified training calibration rows."""
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)

    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero.")

    if sample_count > len(labels):
        raise ValueError(
            f"Requested {sample_count} calibration samples from "
            f"only {len(labels)} training rows."
        )

    classes, class_counts = np.unique(
        labels,
        return_counts=True,
    )

    if classes.size < 2:
        raise RuntimeError(
            "Calibration data must contain more than one class."
        )

    rng = np.random.default_rng(seed)

    raw_targets = (
        class_counts.astype(np.float64)
        / float(class_counts.sum())
        * sample_count
    )
    targets = np.floor(raw_targets).astype(np.int64)

    # Guarantee representation of every class.
    targets = np.maximum(targets, 1)

    while int(targets.sum()) > sample_count:
        reducible = np.where(targets > 1)[0]
        if reducible.size == 0:
            break
        index = reducible[
            np.argmax(targets[reducible] - raw_targets[reducible])
        ]
        targets[index] -= 1

    while int(targets.sum()) < sample_count:
        remaining_capacity = class_counts - targets
        eligible = np.where(remaining_capacity > 0)[0]
        if eligible.size == 0:
            break
        index = eligible[
            np.argmax(raw_targets[eligible] - targets[eligible])
        ]
        targets[index] += 1

    selected_parts: list[np.ndarray] = []

    for class_value, target in zip(classes, targets):
        class_indices = np.flatnonzero(labels == class_value)
        chosen = rng.choice(
            class_indices,
            size=int(target),
            replace=False,
        )
        selected_parts.append(
            np.asarray(chosen, dtype=np.int64)
        )

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)

    if selected.size != sample_count:
        raise RuntimeError(
            "Calibration selection produced "
            f"{selected.size} rows; expected {sample_count}."
        )

    return selected


def quantise_compact(
    model: Any,
    output_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[bytes, np.ndarray]:
    """Convert the compact FP32 model to full INT8 TensorFlow Lite."""
    if len(x_train) < N_CALIB:
        raise RuntimeError(
            f"At least {N_CALIB} calibration samples required; "
            f"found {len(x_train)}"
        )

    # Do not use x_train[:N_CALIB]. The dataset is ordered by scenario,
    # which can make the representative set overwhelmingly Normal.
    calibration = select_stratified_calibration_indices(
        labels=y_train,
        sample_count=N_CALIB,
        seed=SEED,
    )
    calibration_x = x_train[calibration]

    calibration_classes, calibration_counts = np.unique(
        y_train[calibration],
        return_counts=True,
    )
    calibration_summary = ", ".join(
        f"class {int(class_value)}={int(count)}"
        for class_value, count in zip(
            calibration_classes,
            calibration_counts,
        )
    )
    print(
        "[PRUNE] INT8 calibration distribution: "
        f"{calibration_summary}"
    )

    def representative_dataset() -> Iterator[list[np.ndarray]]:
        """Yield deterministic, stratified training-only calibration rows."""
        for sample in calibration_x:
            yield [sample.astype(np.float32)[None, :]]

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT,
    ]

    converter.representative_dataset = (
        representative_dataset
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    with suppress_native_output():
        model_content = converter.convert()

    if not model_content:
        raise RuntimeError(
            "M3 TensorFlow Lite conversion produced an empty model."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(model_content)

    return model_content, calibration


def evaluate_tflite(
    model_path: Path,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> dict[str, Any]:
    """Evaluate M3 while suppressing LiteRT/XNNPACK framework messages."""
    with suppress_native_output():
        evaluation_module = importlib.import_module(
            "tflite_eval"
        )

        result = evaluation_module.evaluate(
            model_path,
            x_validation,
            y_validation,
        )

    required_keys = {
        "accuracy",
        "recall",
    }

    missing_keys = required_keys.difference(result)

    if missing_keys:
        raise RuntimeError(
            "TFLite evaluation result is missing: "
            f"{', '.join(sorted(missing_keys))}."
        )

    return result


def save_compact_fp32_model(model: Any) -> None:
    """Save the compact legacy model without framework status output."""
    M3_FP32_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with suppress_native_output():
        model.save(
            M3_FP32_PATH,
            overwrite=True,
        )


def combined_hidden_sparsity(model: Any) -> float:
    """Return parameter-weighted sparsity across hidden Dense kernels."""
    zero_count = 0
    weight_count = 0

    for layer_name in ("d1", "d2"):
        kernel = model.get_layer(layer_name).get_weights()[0]
        zero_count += int(np.count_nonzero(kernel == 0.0))
        weight_count += int(kernel.size)

    if weight_count == 0:
        raise RuntimeError("Cannot calculate sparsity from empty kernels.")

    return float(zero_count / weight_count)


def load_m2_calibration_samples() -> int:
    """Read and validate the M2 calibration count when metrics exist."""
    if not M2_METRICS_PATH.exists():
        return N_CALIB

    try:
        payload = json.loads(M2_METRICS_PATH.read_text(encoding="utf-8"))
        samples = int(payload.get("calibration_samples", N_CALIB))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Unable to read M2 calibration metadata: {error}"
        ) from error

    if samples < N_CALIB:
        raise RuntimeError(
            f"M2 must use at least {N_CALIB} calibration samples; "
            f"found {samples}. Re-run convert_ptq.py."
        )

    return samples


def save_optimisation_metadata(
    *,
    m2_calibration_samples: int,
    m3_calibration_samples: int,
    achieved_sparsity: float,
    steps_per_epoch: int,
    end_step: int,
) -> None:
    """Save auditable M2/M3 optimisation configuration and results."""
    OPTIMISATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "m2_calibration_samples": int(m2_calibration_samples),
        "m3_calibration_samples": int(m3_calibration_samples),
        "m3_target_sparsity": float(SPARSITY),
        "m3_achieved_sparsity": float(achieved_sparsity),
        "pruning_schedule": "PolynomialDecay",
        "pruning_initial_sparsity": 0.0,
        "pruning_final_sparsity": float(SPARSITY),
        "pruning_begin_step": 0,
        "pruning_end_step": int(end_step),
        "pruning_frequency": 100,
        "pruning_epochs": int(PRUNING_EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "steps_per_epoch": int(steps_per_epoch),
        "calibration_source": "training_split_stratified",
    }

    OPTIMISATION_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    """Run the complete M3 pruning and INT8 quantisation workflow."""
    validate_required_files()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_groups,
        validation_groups,
    ) = load_split()

    print(
        f"[PRUNE] training trucks={train_groups}; "
        f"validation trucks={validation_groups}; overlap=0"
    )

    print(
        f"[PRUNE] training windows={len(x_train)}; "
        f"validation windows={len(x_validation)}"
    )

    baseline_model = legacy_baseline()
    baseline_parameters = int(
        baseline_model.count_params()
    )

    sparse_model = sparsify(
        baseline_model=baseline_model,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    achieved_unstructured_sparsity = combined_hidden_sparsity(
        sparse_model
    )

    print(
        f"[PRUNE] combined hidden-kernel sparsity="
        f"{achieved_unstructured_sparsity * 100:.2f}%"
    )

    compact_model = structurally_prune(
        model=sparse_model,
        keep_ratio=STRUCTURAL_KEEP_RATIO,
    )

    compact_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=COMPACT_LEARNING_RATE,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    compact_model.fit(
        x_train,
        y_train,
        validation_data=(
            x_validation,
            y_validation,
        ),
        epochs=COMPACT_FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=12,
                mode="max",
                restore_best_weights=True,
                verbose=0,
            ),
        ],
    )

    fp32_loss, fp32_accuracy = compact_model.evaluate(
        x_validation,
        y_validation,
        verbose=0,
    )

    save_compact_fp32_model(compact_model)

    tflite_model, calibration = quantise_compact(
        model=compact_model,
        output_path=M3_INT8_PATH,
        x_train=x_train,
        y_train=y_train,
    )

    result = evaluate_tflite(
        model_path=M3_INT8_PATH,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    compact_parameters = int(
        compact_model.count_params()
    )

    parameter_reduction = (
        1.0
        - compact_parameters / max(1, baseline_parameters)
    )

    payload = {
        "variant": "M3_PRUNE35_INT8",
        "validation_method": "grouped_by_simulated_truck",
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "group_overlap": [],
        "training_windows": int(len(x_train)),
        "validation_windows": int(len(x_validation)),
        "target_unstructured_sparsity": SPARSITY,
        "achieved_unstructured_sparsity": achieved_unstructured_sparsity,
        "pruning_schedule": "PolynomialDecay",
        "pruning_frequency": 100,
        "structural_keep_ratio": STRUCTURAL_KEEP_RATIO,
        "baseline_params": baseline_parameters,
        "params": compact_parameters,
        "parameter_reduction": float(parameter_reduction),
        "fp32_acc_after_prune": float(fp32_accuracy),
        "fp32_loss_after_prune": float(fp32_loss),
        "tflite_bytes": int(len(tflite_model)),
        "tflite_kb": float(len(tflite_model) / 1024),
        "calibration_samples": int(len(calibration)),
        "calibration_source": "training_split_stratified",
        "calibration_seed": SEED,
        "calibration_class_counts": {
            str(int(class_value)): int(count)
            for class_value, count in zip(
                *np.unique(
                    y_train[calibration],
                    return_counts=True,
                )
            )
        },
        "input_type": "int8",
        "output_type": "int8",
        **result,
    }

    M3_METRICS_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    steps_per_epoch = max(1, len(x_train) // BATCH_SIZE)
    end_step = steps_per_epoch * PRUNING_EPOCHS
    m2_calibration_samples = load_m2_calibration_samples()

    save_optimisation_metadata(
        m2_calibration_samples=m2_calibration_samples,
        m3_calibration_samples=len(calibration),
        achieved_sparsity=achieved_unstructured_sparsity,
        steps_per_epoch=steps_per_epoch,
        end_step=end_step,
    )

    print(f"[SAVE] compact FP32 model: {M3_FP32_PATH}")
    print(f"[SAVE] compact INT8 model: {M3_INT8_PATH}")
    print(f"[SAVE] M3 metrics: {M3_METRICS_PATH}")
    print(f"[SAVE] optimisation metadata: {OPTIMISATION_METADATA_PATH}")

    print(
        f"[PRUNE] params={compact_parameters}; "
        f"reduction={parameter_reduction * 100:.1f}%; "
        f"size={len(tflite_model) / 1024:.2f} KB; "
        f"grouped validation accuracy="
        f"{result['accuracy'] * 100:.2f}%"
    )

    print(
        "[PRUNE] recall "
        + " ".join(
            f"{class_name}={value * 100:.1f}%"
            for class_name, value in result["recall"].items()
        )
    )

    print("[PASS] M3 pruning and INT8 quantisation completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "\n[STOP] M3 pruning and quantisation "
            "interrupted by user."
        )
        raise SystemExit(130)

    except Exception as error:
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )
        raise SystemExit(1)
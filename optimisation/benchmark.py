"""
Benchmark LogiEdge M1, M2 and M3 model variants.

For each model, the script records:

- Model size
- Held-out validation accuracy
- Critical-class recall
- Mean latency
- p95 latency
- Estimated energy per inference
- Warm-up run count
- Measured run count
- Estimated active CPU power
- Laptop CPU TDP assumption

Outputs:
    optimisation/results/benchmark_results.csv
    optimisation/results/pareto_analysis.csv
    optimisation/results/pareto_chart.png
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score


ROOT = Path(__file__).resolve().parents[1]

DATASET_CANDIDATES = [
    ROOT / "training" / "dataset.npz",
    ROOT / "training" / "assignment_dataset.npz",
    ROOT / "data_pipeline" / "dataset.npz",
]

STATS_PATH = ROOT / "data_pipeline" / "training_stats.npy"

RESULTS_DIR = ROOT / "optimisation" / "results"
OUTPUT_CSV = RESULTS_DIR / "benchmark_results.csv"
PARETO_CSV = RESULTS_DIR / "pareto_analysis.csv"
PARETO_CHART = RESULTS_DIR / "pareto_chart.png"

WARMUP_RUNS = 10
MEASURED_RUNS = 200
EXPECTED_FEATURES = 6
CRITICAL_CLASS = 2
MIN_RECOMMENDED_ACCURACY_PCT = 95.0
MIN_RECOMMENDED_CRITICAL_RECALL_PCT = 95.0

# Override at runtime when required:
#   $env:LOGIEDGE_LAPTOP_TDP_W = "45"
DEFAULT_LAPTOP_TDP_W = 45.0

MODEL_CANDIDATES: dict[str, list[Path]] = {
    "M1_FP32": [
        ROOT / "training" / "models" / "m1_fp32.tflite",
        ROOT / "training" / "models" / "m1_float32.tflite",
        ROOT / "training" / "models" / "model_fp32.tflite",
    ],
    "M2_PTQ_INT8": [
        ROOT / "training" / "models" / "m2_ptq_int8.tflite",
        ROOT / "training" / "models" / "m2_int8.tflite",
        ROOT / "training" / "models" / "model_int8.tflite",
    ],
    "M3_PRUNE35_INT8": [
        ROOT / "training" / "models" / "m3_pruned_int8.tflite",
        ROOT / "training" / "models" / "m3_int8.tflite",
    ],
}


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark CSV row."""

    variant: str
    size_kb: float
    accuracy_pct: float
    recall_critical_pct: float
    mean_latency_ms: float
    p95_latency_ms: float
    energy_mj_per_inference: float
    warmup_runs: int
    measured_runs: int
    estimated_power_w: float
    laptop_tdp_w: float
    cpu_percent: float


def first_existing_path(
    candidates: list[Path],
    description: str,
) -> Path:
    """Return the first existing file."""

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked = "\n".join(
        f"  - {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        f"{description} was not found. Checked:\n{checked}"
    )


def pick_key(
    archive: np.lib.npyio.NpzFile,
    candidates: list[str],
) -> str | None:
    """Return the first matching NPZ key."""

    available = set(archive.files)

    for candidate in candidates:
        if candidate in available:
            return candidate

    return None


def ensure_feature_matrix(
    values: np.ndarray,
    description: str,
) -> np.ndarray:
    """Return a finite two-dimensional six-feature matrix."""

    array = np.asarray(values)

    if array.ndim == 1:
        if array.size % EXPECTED_FEATURES != 0:
            raise ValueError(
                f"{description} cannot be reshaped into "
                f"{EXPECTED_FEATURES} features."
            )

        array = array.reshape(-1, EXPECTED_FEATURES)

    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)

    if array.ndim != 2:
        raise ValueError(
            f"{description} must be two-dimensional. "
            f"Received shape: {array.shape}"
        )

    if array.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{description} must contain exactly "
            f"{EXPECTED_FEATURES} features. "
            f"Received shape: {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{description} contains NaN or infinite values."
        )

    return array.astype(np.float32, copy=False)


def ensure_labels(
    values: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Return integer class labels."""

    labels = np.asarray(values)

    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)

    labels = labels.reshape(-1)

    if labels.shape[0] != sample_count:
        raise ValueError(
            f"Found {labels.shape[0]} labels for "
            f"{sample_count} feature rows."
        )

    return labels.astype(np.int64, copy=False)


def load_validation_data(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw held-out validation features and labels."""

    with np.load(
        dataset_path,
        allow_pickle=True,
    ) as archive:
        print(
            "[BENCH] Dataset keys: "
            + ", ".join(sorted(archive.files))
        )

        feature_key = pick_key(
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

        label_key = pick_key(
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

        if feature_key is not None and label_key is not None:
            features = ensure_feature_matrix(
                archive[feature_key],
                f"Validation features '{feature_key}'",
            )

            labels = ensure_labels(
                archive[label_key],
                features.shape[0],
            )

            return features, labels

        all_feature_key = pick_key(
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

        all_label_key = pick_key(
            archive,
            [
                "y",
                "Y",
                "labels",
                "targets",
                "target",
            ],
        )

        index_key = pick_key(
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
            all_feature_key is None
            or all_label_key is None
            or index_key is None
        ):
            raise KeyError(
                "Unable to locate raw held-out validation data. "
                f"Available keys: {archive.files}"
            )

        all_features = ensure_feature_matrix(
            archive[all_feature_key],
            f"Features '{all_feature_key}'",
        )

        all_labels = ensure_labels(
            archive[all_label_key],
            all_features.shape[0],
        )

        indices = np.asarray(
            archive[index_key],
            dtype=np.int64,
        ).reshape(-1)

        if indices.size == 0:
            raise ValueError(
                "The validation index array is empty."
            )

        if np.any(indices < 0) or np.any(
            indices >= all_features.shape[0]
        ):
            raise IndexError(
                "Validation indices are outside the dataset."
            )

        return (
            all_features[indices],
            all_labels[indices],
        )


def unpack_object_mapping(
    loaded: np.ndarray,
) -> dict[str, Any] | None:
    """Extract a dictionary stored in an object array."""

    if loaded.dtype != object:
        return None

    try:
        item = loaded.item()
    except (ValueError, AttributeError):
        return None

    if isinstance(item, dict):
        return item

    return None


def load_training_stats(
    stats_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load frozen training mean and standard deviation."""

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
                "Frozen statistics dictionary does not contain "
                "recognised mean/std keys."
            )

        mean = np.asarray(
            mapping[mean_key],
            dtype=np.float32,
        ).reshape(-1)

        std = np.asarray(
            mapping[std_key],
            dtype=np.float32,
        ).reshape(-1)

    else:
        numeric = np.asarray(
            loaded,
            dtype=np.float32,
        )

        if numeric.shape == (2, EXPECTED_FEATURES):
            mean = numeric[0]
            std = numeric[1]

        elif numeric.shape == (EXPECTED_FEATURES, 2):
            mean = numeric[:, 0]
            std = numeric[:, 1]

        elif numeric.ndim == 1 and numeric.size == 12:
            numeric = numeric.reshape(2, EXPECTED_FEATURES)
            mean = numeric[0]
            std = numeric[1]

        else:
            raise ValueError(
                "Unsupported training_stats.npy shape: "
                f"{numeric.shape}"
            )

    if mean.shape != (EXPECTED_FEATURES,):
        raise ValueError(
            f"Mean has invalid shape: {mean.shape}"
        )

    if std.shape != (EXPECTED_FEATURES,):
        raise ValueError(
            f"Standard deviation has invalid shape: {std.shape}"
        )

    if np.any(std < 0):
        raise ValueError(
            "Standard deviation contains negative values."
        )

    return mean, std


def normalise(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalise using frozen training statistics."""

    safe_std = np.where(std == 0, 1.0, std)

    values = (features - mean) / safe_std

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Normalised features contain NaN or infinity."
        )

    return values.astype(np.float32, copy=False)


def quantise_input(
    values: np.ndarray,
    dtype: np.dtype,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Quantise a float input tensor when required."""

    if dtype not in (np.int8, np.uint8):
        return values.astype(dtype, copy=False)

    if scale <= 0:
        raise ValueError(
            "Quantised model has an invalid input scale."
        )

    quantised = np.round(
        values / scale + zero_point
    )

    limits = np.iinfo(dtype)

    return np.clip(
        quantised,
        limits.min,
        limits.max,
    ).astype(dtype)


def dequantise_output(
    values: np.ndarray,
    dtype: np.dtype,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Dequantise a TFLite output tensor."""

    if dtype not in (np.int8, np.uint8):
        return values.astype(np.float32, copy=False)

    if scale <= 0:
        raise ValueError(
            "Quantised model has an invalid output scale."
        )

    return (
        values.astype(np.float32) - zero_point
    ) * scale


class TFLiteRunner:
    """Reusable TFLite inference runner."""

    def __init__(self, model_path: Path) -> None:
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
        )

        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        if len(input_details) != 1:
            raise ValueError(
                "Expected exactly one model input tensor."
            )

        if len(output_details) != 1:
            raise ValueError(
                "Expected exactly one model output tensor."
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

        if input_shape[-1] != EXPECTED_FEATURES:
            raise ValueError(
                f"Model expects {input_shape[-1]} features; "
                f"expected {EXPECTED_FEATURES}."
            )

    def infer(
        self,
        feature_row: np.ndarray,
    ) -> np.ndarray:
        """Run one inference and return dequantised output."""

        values = np.asarray(
            feature_row,
            dtype=np.float32,
        ).reshape(1, EXPECTED_FEATURES)

        model_input = quantise_input(
            values,
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

        return np.asarray(
            output,
            dtype=np.float32,
        ).reshape(-1)

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Predict classes for a feature matrix."""

        predictions: list[int] = []

        for feature_row in features:
            output = self.infer(feature_row)

            if output.size == 1:
                predicted = int(output[0] >= 0.5)
            else:
                predicted = int(np.argmax(output))

            predictions.append(predicted)

        return np.asarray(
            predictions,
            dtype=np.int64,
        )


def benchmark_latency(
    run_single_inference: Callable[[], None],
) -> tuple[np.ndarray, float]:
    """
    Execute exactly 10 warm-up runs and 200 timed runs.

    Returns:
        latency array in milliseconds
        average process CPU percentage during the measured loop
    """

    for _ in range(WARMUP_RUNS):
        run_single_inference()

    process = psutil.Process(os.getpid())

    # Prime the CPU percentage counter.
    process.cpu_percent(interval=None)

    latencies_ms: list[float] = []

    for _ in range(MEASURED_RUNS):
        start_ns = time.perf_counter_ns()

        run_single_inference()

        elapsed_ns = time.perf_counter_ns() - start_ns
        elapsed_ms = elapsed_ns / 1_000_000.0

        latencies_ms.append(elapsed_ms)

    cpu_percent = float(
        process.cpu_percent(interval=None)
    )

    latencies = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    if latencies.shape != (MEASURED_RUNS,):
        raise RuntimeError(
            "Benchmark did not create exactly "
            f"{MEASURED_RUNS} measurements."
        )

    if not np.all(np.isfinite(latencies)):
        raise ValueError(
            "Latency measurements contain invalid values."
        )

    if np.any(latencies <= 0):
        raise ValueError(
            "Latency measurements must be positive."
        )

    return latencies, cpu_percent


def get_laptop_tdp_w() -> float:
    """Read the configured laptop TDP assumption."""

    raw_value = os.getenv(
        "LOGIEDGE_LAPTOP_TDP_W",
        str(DEFAULT_LAPTOP_TDP_W),
    )

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "LOGIEDGE_LAPTOP_TDP_W must be numeric."
        ) from exc

    if value <= 0:
        raise ValueError(
            "Laptop TDP must be greater than zero."
        )

    return value


def normalise_cpu_percent(
    process_cpu_percent: float,
) -> float:
    """
    Convert psutil process CPU percentage to a 0-100 system fraction.

    psutil can report more than 100% for a multi-threaded process.
    Dividing by the logical CPU count approximates total-system usage.
    """

    logical_cpu_count = max(
        1,
        psutil.cpu_count(logical=True) or 1,
    )

    system_fraction_percent = (
        process_cpu_percent / logical_cpu_count
    )

    return float(
        np.clip(
            system_fraction_percent,
            0.0,
            100.0,
        )
    )


def evaluate_model(
    variant: str,
    model_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    laptop_tdp_w: float,
) -> BenchmarkResult:
    """Evaluate one model variant."""

    print()
    print("=" * 72)
    print(f"[BENCH] {variant}")
    print(f"[BENCH] Model: {model_path}")
    print("=" * 72)

    runner = TFLiteRunner(model_path)

    predictions = runner.predict(features)

    accuracy_pct = float(
        accuracy_score(
            labels,
            predictions,
        ) * 100.0
    )

    critical_recall_pct = float(
        recall_score(
            labels,
            predictions,
            labels=[CRITICAL_CLASS],
            average=None,
            zero_division=0,
        )[0] * 100.0
    )

    # Use a deterministic representative validation sample for latency.
    # The same sample is used for every model.
    sample = features[0].copy()

    latencies_ms, raw_cpu_percent = benchmark_latency(
        lambda: runner.infer(sample)
    )

    mean_latency_ms = float(
        np.mean(latencies_ms)
    )

    p95_latency_ms = float(
        np.percentile(latencies_ms, 95)
    )

    cpu_percent = normalise_cpu_percent(
        raw_cpu_percent
    )

    cpu_fraction = max(
        cpu_percent,
        0.0,
    ) / 100.0

    estimated_power_w = (
        laptop_tdp_w * cpu_fraction
    )

    # W × ms = mJ
    energy_mj = (
        estimated_power_w *
        mean_latency_ms
    )

    size_kb = (
        model_path.stat().st_size / 1024.0
    )

    result = BenchmarkResult(
        variant=variant,
        size_kb=round(size_kb, 4),
        accuracy_pct=round(accuracy_pct, 4),
        recall_critical_pct=round(
            critical_recall_pct,
            4,
        ),
        mean_latency_ms=round(
            mean_latency_ms,
            6,
        ),
        p95_latency_ms=round(
            p95_latency_ms,
            6,
        ),
        energy_mj_per_inference=round(
            energy_mj,
            6,
        ),
        warmup_runs=WARMUP_RUNS,
        measured_runs=MEASURED_RUNS,
        estimated_power_w=round(
            estimated_power_w,
            6,
        ),
        laptop_tdp_w=round(
            laptop_tdp_w,
            4,
        ),
        cpu_percent=round(
            cpu_percent,
            4,
        ),
    )

    print(
        f"[BENCH] size={result.size_kb:.2f} KB"
    )

    print(
        f"[BENCH] accuracy={result.accuracy_pct:.2f}%"
    )

    print(
        "[BENCH] Critical recall="
        f"{result.recall_critical_pct:.2f}%"
    )

    print(
        "[BENCH] latency="
        f"{result.mean_latency_ms:.4f} ms mean, "
        f"{result.p95_latency_ms:.4f} ms p95"
    )

    print(
        "[BENCH] methodology="
        f"{result.warmup_runs} warm-ups, "
        f"{result.measured_runs} measured runs"
    )

    print(
        "[BENCH] CPU estimate="
        f"{result.cpu_percent:.2f}%, "
        f"power={result.estimated_power_w:.4f} W, "
        f"energy={result.energy_mj_per_inference:.6f} mJ"
    )

    return result


def is_dominated(
    candidate: BenchmarkResult,
    all_results: list[BenchmarkResult],
) -> bool:
    """
    Return True when another model is no worse in all objectives and
    strictly better in at least one objective.

    Objectives:
        Maximise accuracy.
        Minimise mean latency.
        Minimise model size.
    """

    for other in all_results:
        if other.variant == candidate.variant:
            continue

        no_worse = (
            other.accuracy_pct >= candidate.accuracy_pct
            and other.mean_latency_ms
            <= candidate.mean_latency_ms
            and other.size_kb <= candidate.size_kb
        )

        strictly_better = (
            other.accuracy_pct > candidate.accuracy_pct
            or other.mean_latency_ms
            < candidate.mean_latency_ms
            or other.size_kb < candidate.size_kb
        )

        if no_worse and strictly_better:
            return True

    return False


def write_benchmark_csv(
    results: list[BenchmarkResult],
) -> None:
    """Write benchmark results."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "variant",
        "size_kb",
        "accuracy_pct",
        "recall_critical_pct",
        "mean_latency_ms",
        "p95_latency_ms",
        "energy_mj_per_inference",
        "warmup_runs",
        "measured_runs",
        "estimated_power_w",
        "laptop_tdp_w",
        "cpu_percent",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))


def write_pareto_analysis(
    results: list[BenchmarkResult],
) -> None:
    """Write and validate the required Pareto-analysis CSV.

    Objectives:
        - maximise validation accuracy,
        - minimise mean latency,
        - minimise model size.

    The CSV schema is intentionally restricted to the assignment fields:
        variant, accuracy_pct, mean_latency_ms, size_kb,
        pareto_optimal, recommended
    """

    if not results:
        raise ValueError(
            "Cannot create Pareto analysis from an empty result set."
        )

    rows: list[dict[str, Any]] = []

    for result in results:
        pareto_optimal = not is_dominated(
            result,
            results,
        )

        rows.append(
            {
                "variant": result.variant,
                "accuracy_pct": result.accuracy_pct,
                "mean_latency_ms": result.mean_latency_ms,
                "size_kb": result.size_kb,
                "pareto_optimal": pareto_optimal,
                "recommended": False,
            }
        )

    pareto_rows = [
        row
        for row in rows
        if bool(row["pareto_optimal"])
    ]

    if not pareto_rows:
        raise RuntimeError(
            "Pareto analysis produced no Pareto-optimal variants."
        )

    result_by_variant = {
        result.variant: result
        for result in results
    }

    eligible_rows = [
        row
        for row in pareto_rows
        if (
            float(row["accuracy_pct"])
            >= MIN_RECOMMENDED_ACCURACY_PCT
            and result_by_variant[
                str(row["variant"])
            ].recall_critical_pct
            >= MIN_RECOMMENDED_CRITICAL_RECALL_PCT
        )
    ]

    if not eligible_rows:
        raise RuntimeError(
            "No Pareto-optimal model satisfies the deployment quality "
            f"gates: accuracy >= {MIN_RECOMMENDED_ACCURACY_PCT:.1f}% "
            "and Critical recall >= "
            f"{MIN_RECOMMENDED_CRITICAL_RECALL_PCT:.1f}%."
        )

    # Recommend the highest-accuracy eligible Pareto model, followed by
    # lower latency and then smaller size.
    recommended_row = sorted(
        eligible_rows,
        key=lambda row: (
            -float(row["accuracy_pct"]),
            float(row["mean_latency_ms"]),
            float(row["size_kb"]),
        ),
    )[0]
    recommended_row["recommended"] = True

    recommended_rows = [
        row
        for row in rows
        if bool(row["recommended"])
    ]

    if len(recommended_rows) != 1:
        raise RuntimeError(
            "Pareto analysis must contain exactly one recommended variant."
        )

    if not bool(recommended_rows[0]["pareto_optimal"]):
        raise RuntimeError(
            "The recommended variant must be Pareto optimal."
        )

    fieldnames = [
        "variant",
        "accuracy_pct",
        "mean_latency_ms",
        "size_kb",
        "pareto_optimal",
        "recommended",
    ]

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PARETO_CSV.open(
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

    print()
    print("[PARETO] Frontier validation")

    for row in rows:
        print(
            f"[PARETO] {row['variant']}: "
            f"accuracy={float(row['accuracy_pct']):.4f}%, "
            f"latency={float(row['mean_latency_ms']):.6f} ms, "
            f"size={float(row['size_kb']):.4f} KB, "
            f"pareto_optimal={row['pareto_optimal']}, "
            f"recommended={row['recommended']}"
        )

    print(
        f"[PARETO] recommended="
        f"{recommended_rows[0]['variant']}"
    )
    print(f"[PARETO] CSV saved: {PARETO_CSV}")


def create_pareto_chart(
    results: list[BenchmarkResult],
) -> None:
    """Create the required accuracy-latency Pareto chart."""

    figure, axis = plt.subplots(
        figsize=(9, 6),
    )

    for result in results:
        marker = (
            "*"
            if result.variant == "M3_PRUNE35_INT8"
            else "o"
        )

        marker_size = (
            180
            if marker == "*"
            else 90
        )

        axis.scatter(
            result.mean_latency_ms,
            result.accuracy_pct,
            s=marker_size,
            marker=marker,
        )

        axis.annotate(
            (
                f"{result.variant}\n"
                f"{result.size_kb:.1f} KB"
            ),
            (
                result.mean_latency_ms,
                result.accuracy_pct,
            ),
            xytext=(8, 8),
            textcoords="offset points",
        )

    axis.set_title(
        "LogiEdge Model Pareto Comparison"
    )

    axis.set_xlabel(
        "Mean inference latency (ms) — lower is better"
    )

    axis.set_ylabel(
        "Validation accuracy (%) — higher is better"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        PARETO_CHART,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def validate_results(
    results: list[BenchmarkResult],
) -> None:
    """Validate final benchmark integrity."""

    required_variants = {
        "M1_FP32",
        "M2_PTQ_INT8",
        "M3_PRUNE35_INT8",
    }

    actual_variants = {
        result.variant
        for result in results
    }

    if actual_variants != required_variants:
        raise ValueError(
            "Benchmark variants do not match the required set. "
            f"Found: {sorted(actual_variants)}"
        )

    for result in results:
        if result.warmup_runs != 10:
            raise ValueError(
                f"{result.variant} did not use 10 warm-ups."
            )

        if result.measured_runs != 200:
            raise ValueError(
                f"{result.variant} did not use 200 measured runs."
            )

        if not 0 <= result.accuracy_pct <= 100:
            raise ValueError(
                f"{result.variant} accuracy is outside 0-100."
            )

        if not 0 <= result.recall_critical_pct <= 100:
            raise ValueError(
                f"{result.variant} Critical recall is outside "
                "0-100."
            )

        if result.size_kb <= 0:
            raise ValueError(
                f"{result.variant} has invalid model size."
            )

        if result.mean_latency_ms <= 0:
            raise ValueError(
                f"{result.variant} has invalid mean latency."
            )

        if result.p95_latency_ms <= 0:
            raise ValueError(
                f"{result.variant} has invalid p95 latency."
            )

        # p95 is not required to be greater than or equal to the mean.
        # A small number of extreme latency outliers can raise the mean above
        # the 95th percentile. Validate both metrics independently instead.
        if not np.isfinite(result.mean_latency_ms):
            raise ValueError(
                f"{result.variant} has non-finite mean latency."
            )

        if not np.isfinite(result.p95_latency_ms):
            raise ValueError(
                f"{result.variant} has non-finite p95 latency."
            )

        if result.energy_mj_per_inference < 0:
            raise ValueError(
                f"{result.variant} has negative energy."
            )


def main() -> int:
    """Run the complete model benchmark."""

    dataset_path = first_existing_path(
        DATASET_CANDIDATES,
        "Validation dataset",
    )

    stats_path = first_existing_path(
        [STATS_PATH],
        "Frozen training statistics",
    )

    model_paths = {
        variant: first_existing_path(
            candidates,
            f"{variant} model",
        )
        for variant, candidates
        in MODEL_CANDIDATES.items()
    }

    x_validation_raw, y_validation = (
        load_validation_data(dataset_path)
    )

    feature_mean, feature_std = (
        load_training_stats(stats_path)
    )

    x_validation = normalise(
        x_validation_raw,
        feature_mean,
        feature_std,
    )

    if CRITICAL_CLASS not in np.unique(y_validation):
        raise ValueError(
            "Validation split contains no Critical samples."
        )

    laptop_tdp_w = get_laptop_tdp_w()

    print(f"[BENCH] Dataset: {dataset_path}")
    print(
        "[BENCH] Validation shape: "
        f"X={x_validation.shape}, "
        f"y={y_validation.shape}"
    )

    print(
        f"[BENCH] Laptop TDP assumption: "
        f"{laptop_tdp_w:.2f} W"
    )

    results = [
        evaluate_model(
            variant=variant,
            model_path=model_paths[variant],
            features=x_validation,
            labels=y_validation,
            laptop_tdp_w=laptop_tdp_w,
        )
        for variant in (
            "M1_FP32",
            "M2_PTQ_INT8",
            "M3_PRUNE35_INT8",
        )
    ]

    validate_results(results)
    write_benchmark_csv(results)
    write_pareto_analysis(results)
    create_pareto_chart(results)

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)

    for result in results:
        print(
            f"{result.variant}: "
            f"accuracy={result.accuracy_pct:.2f}%, "
            f"Critical recall="
            f"{result.recall_critical_pct:.2f}%, "
            f"mean={result.mean_latency_ms:.4f} ms, "
            f"p95={result.p95_latency_ms:.4f} ms, "
            f"size={result.size_kb:.2f} KB, "
            f"energy="
            f"{result.energy_mj_per_inference:.6f} mJ"
        )

    print()
    print(f"Benchmark CSV: {OUTPUT_CSV}")
    print(f"Pareto analysis: {PARETO_CSV}")
    print(f"Pareto chart: {PARETO_CHART}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"[BENCH][ERROR] {exc}",
            file=sys.stderr,
        )
        raise
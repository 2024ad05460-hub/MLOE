"""LogiEdge Task E1: PSI drift monitoring.

PSI formula
-----------
PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

Bins
----
[0.00, 0.25)
[0.25, 0.50)
[0.50, 0.75)
[0.75, 1.00]

Reference
---------
300 clean Normal-condition windows sampled from multiple simulated trucks.

Monitor
-------
Rolling window of the latest 100 inference results.

Alert
-----
PSI > 0.25

Clear
-----
PSI < 0.10 after an alert has been raised.

Score choices
-------------
normal_prob:
    Probability assigned to the Normal class. This is the default cargo
    health score and is more useful for detecting operational faults.

max_prob:
    Maximum probability across all classes. This can remain high even when
    the model changes confidently from Normal to Critical.

Modes
-----
reference:
    Build monitoring/reference_dist.json.

simulate:
    Run clean, injected-fault and recovered phases.

live:
    Subscribe to MQTT inference messages and monitor live PSI.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Logging configuration
# Must be set before TensorFlow-related modules are imported.
# ---------------------------------------------------------------------------
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import argparse
import contextlib
import json
import sys
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATA_PIPELINE_DIR = ROOT / "data_pipeline"
OPTIMISATION_DIR = ROOT / "optimisation"
MONITORING_DIR = ROOT / "monitoring"
TRAINING_DIR = ROOT / "training"

STATS_PATH = DATA_PIPELINE_DIR / "training_stats.npy"
REF_PATH = MONITORING_DIR / "reference_dist.json"
PSI_METADATA_PATH = MONITORING_DIR / "psi_metadata.json"
DEFAULT_MODEL = TRAINING_DIR / "models" / "m3_pruned_int8.tflite"

sys.path.insert(0, str(DATA_PIPELINE_DIR))
sys.path.insert(0, str(OPTIMISATION_DIR))


BINS = np.array(
    [0.0, 0.25, 0.50, 0.75, 1.0001],
    dtype=np.float64,
)

BIN_LABELS = [
    "[0,0.25)",
    "[0.25,0.50)",
    "[0.50,0.75)",
    "[0.75,1.0]",
]

PSI_ALERT = 0.25
PSI_CLEAR = 0.10

ROLLING_N = 100
REF_WINDOWS = 300
REFERENCE_TRUCKS = 10

EPS = 0.005


@contextlib.contextmanager
def suppress_native_output() -> Iterator[None]:
    """Temporarily suppress harmless native framework output.

    TensorFlow Lite can write warnings and information directly to operating
    system stdout and stderr. Those messages bypass Python logging and are
    interpreted by PowerShell as NativeCommandError when stderr is redirected.

    This context manager is used only around TensorFlow Lite imports and model
    initialization. Python exceptions still propagate normally.
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

        with open(
            os.devnull,
            mode="w",
            encoding="utf-8",
        ) as devnull:
            sys.stdout.flush()
            sys.stderr.flush()

            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

    finally:
        if (
            saved_stdout_fd is not None
            and stdout_fd is not None
        ):
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)

        if (
            saved_stderr_fd is not None
            and stderr_fd is not None
        ):
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)


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


from preprocessing import (  # noqa: E402
    CLASS_NAMES,
    Normaliser,
    WindowFeaturiser,
)
from simulator import ColdChainSimulator  # noqa: E402

with suppress_native_output():
    from tflite_eval import TFLiteModel  # noqa: E402


def validate_common_files(
    model_path: Path,
) -> None:
    """Validate files required by every monitor mode."""
    required_files = [
        model_path,
        STATS_PATH,
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
            "Required drift-monitor files are missing:\n"
            f"{missing_text}\n\n"
            "Run the model training and optimization pipeline first."
        )


def histogram(
    confidences: np.ndarray | list[float],
) -> np.ndarray:
    """Convert scores into a smoothed four-bin probability distribution."""
    confidence_array = np.asarray(
        confidences,
        dtype=np.float64,
    )

    if confidence_array.size == 0:
        return np.full(
            len(BIN_LABELS),
            1.0 / len(BIN_LABELS),
            dtype=np.float64,
        )

    clipped = np.clip(
        confidence_array,
        0.0,
        1.0,
    )

    counts, _ = np.histogram(
        clipped,
        bins=BINS,
    )

    distribution = counts.astype(
        np.float64
    ) / max(1, counts.sum())

    # Apply a small floor so an empty bin cannot create infinite PSI.
    distribution = np.maximum(
        distribution,
        EPS,
    )

    # Renormalize after applying the floor.
    distribution = (
        distribution / distribution.sum()
    )

    return distribution


def calculate_psi(
    expected: np.ndarray | list[float],
    actual: np.ndarray | list[float],
) -> float:
    """Calculate Population Stability Index."""
    expected_array = np.maximum(
        np.asarray(expected, dtype=np.float64),
        EPS,
    )

    actual_array = np.maximum(
        np.asarray(actual, dtype=np.float64),
        EPS,
    )

    if expected_array.shape != actual_array.shape:
        raise ValueError(
            "Expected and actual PSI distributions must have "
            f"the same shape: {expected_array.shape} versus "
            f"{actual_array.shape}."
        )

    expected_array = (
        expected_array / expected_array.sum()
    )

    actual_array = (
        actual_array / actual_array.sum()
    )

    value = np.sum(
        (actual_array - expected_array)
        * np.log(actual_array / expected_array)
    )

    return float(value)


def confidence_score(
    probabilities: np.ndarray,
    kind: str = "normal_prob",
) -> float:
    """Extract the scalar score used for PSI monitoring."""
    probability_array = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if probability_array.size < 3:
        raise ValueError(
            "Expected a three-class probability vector, "
            f"received shape {probability_array.shape}."
        )

    if kind == "max_prob":
        return float(
            np.max(probability_array)
        )

    if kind == "normal_prob":
        return float(
            probability_array[0]
        )

    raise ValueError(
        f"Unsupported confidence score: {kind}"
    )


def create_tflite_model(
    model_path: Path,
) -> Any:
    """Create a TFLite model without deprecation or XNNPACK output."""
    with suppress_native_output():
        model = TFLiteModel(model_path)

    return model


class Scorer:
    """Run the edge feature, normalization and TFLite inference pipeline."""

    def __init__(
        self,
        model_path: Path,
        score: str = "normal_prob",
    ) -> None:
        self.model = create_tflite_model(
            model_path
        )

        self.stats = Normaliser.load(
            STATS_PATH
        )

        self.score_kind = score

    def score_stream(
        self,
        simulator: ColdChainSimulator,
        duration_seconds: float,
    ) -> Iterator[tuple[float, int, float]]:
        """Yield timestamp, predicted class and confidence score."""
        featuriser = WindowFeaturiser()

        for (
            timestamp,
            measurement_kind,
            payload,
        ) in simulator.stream(duration_seconds):
            features = featuriser.push(
                timestamp,
                measurement_kind,
                payload,
            )

            if features is None:
                continue

            normalized_features = self.stats.transform(
                features
            )

            probabilities = self.model.predict(
                normalized_features
            )

            predicted_class = int(
                np.argmax(probabilities)
            )

            score = confidence_score(
                probabilities,
                self.score_kind,
            )

            yield (
                timestamp,
                predicted_class,
                score,
            )


def print_distribution(
    distribution: np.ndarray,
) -> None:
    """Print an ASCII-safe PSI distribution."""
    for label, value in zip(
        BIN_LABELS,
        distribution,
    ):
        bar = "#" * int(
            round(value * 50)
        )

        print(
            f"  {label:<12s} "
            f"{value * 100:5.1f}%  "
            f"{bar}"
        )


def build_reference(
    model_path: Path,
    score: str = "normal_prob",
    number_of_trucks: int = REFERENCE_TRUCKS,
) -> dict[str, Any]:
    """Build a clean fleet-level PSI reference distribution."""
    if number_of_trucks <= 0:
        raise ValueError(
            "number_of_trucks must be greater than zero."
        )

    scorer = Scorer(
        model_path=model_path,
        score=score,
    )

    confidences: list[float] = []
    predicted_classes: list[int] = []

    windows_per_truck = int(
        np.ceil(
            REF_WINDOWS / number_of_trucks
        )
    )

    for truck_index in range(
        number_of_trucks
    ):
        simulator = ColdChainSimulator(
            "none",
            f"REF-{truck_index:02d}",
            seed=2000 + truck_index,
        )

        collected_for_truck = 0

        for (
            _,
            predicted_class,
            confidence,
        ) in scorer.score_stream(
            simulator,
            20 * 60,
        ):
            confidences.append(
                confidence
            )

            predicted_classes.append(
                predicted_class
            )

            collected_for_truck += 1

            if (
                collected_for_truck
                >= windows_per_truck
            ):
                break

        if len(confidences) >= REF_WINDOWS:
            break

    confidences = confidences[:REF_WINDOWS]
    predicted_classes = predicted_classes[:REF_WINDOWS]

    if len(confidences) < REF_WINDOWS:
        raise RuntimeError(
            "Unable to collect the required reference windows. "
            f"Collected {len(confidences)} of {REF_WINDOWS}."
        )

    distribution = histogram(
        confidences
    )

    predicted_class_array = np.asarray(
        predicted_classes,
        dtype=np.int32,
    )

    class_share = {
        CLASS_NAMES[class_index]: round(
            float(
                np.mean(
                    predicted_class_array
                    == class_index
                )
            ),
            4,
        )
        for class_index in range(
            len(CLASS_NAMES)
        )
    }

    reference = {
        "model": model_path.name,
        "score": score,
        "n_windows": REF_WINDOWS,
        "n_trucks": number_of_trucks,
        "bins": BIN_LABELS,
        "bin_edges": [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ],
        "distribution": [
            round(float(value), 6)
            for value in distribution
        ],
        "class_share": class_share,
        "mean_confidence": round(
            float(np.mean(confidences)),
            4,
        ),
        "built_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
    }

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REF_PATH.write_text(
        json.dumps(
            reference,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[REF] {REF_WINDOWS} clean Normal-condition windows "
        f"from {number_of_trucks} trucks"
    )

    print(
        f"[REF] output: {REF_PATH}"
    )

    print_distribution(
        distribution
    )

    print(
        "[REF] Normal-class share "
        f"{reference['class_share']['Normal'] * 100:.1f}%"
    )

    print(
        f"[REF] mean {score}: "
        f"{reference['mean_confidence']:.4f}"
    )

    print(
        "[PASS] PSI reference distribution created"
    )

    return reference


def load_reference(
    expected_score: str,
    expected_model: Path,
) -> dict[str, Any] | None:
    """Load and validate an existing reference distribution."""
    if not REF_PATH.exists():
        return None

    try:
        reference = json.loads(
            REF_PATH.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid PSI reference JSON: {error}"
        ) from error

    required_keys = {
        "model",
        "score",
        "distribution",
        "bins",
    }

    missing_keys = required_keys.difference(
        reference
    )

    if missing_keys:
        raise RuntimeError(
            "PSI reference is missing fields: "
            f"{', '.join(sorted(missing_keys))}"
        )

    if reference.get("score") != expected_score:
        return None

    if reference.get("model") != expected_model.name:
        return None

    distribution = reference.get(
        "distribution"
    )

    if (
        not isinstance(distribution, list)
        or len(distribution) != len(BIN_LABELS)
    ):
        raise RuntimeError(
            "PSI reference distribution must contain "
            f"{len(BIN_LABELS)} bins."
        )

    return reference


class PSIMonitor:
    """Maintain a rolling PSI window and alert state."""

    def __init__(
        self,
        reference: dict[str, Any],
    ) -> None:
        self.expected = np.asarray(
            reference["distribution"],
            dtype=np.float64,
        )

        self.confidence_buffer: deque[float] = deque(
            maxlen=ROLLING_N
        )

        self.class_buffer: deque[int] = deque(
            maxlen=ROLLING_N
        )

        self.alerting = False

    def push(
        self,
        predicted_class: int,
        confidence: float,
    ) -> None:
        """Add one inference result to the rolling window."""
        self.confidence_buffer.append(
            float(confidence)
        )

        self.class_buffer.append(
            int(predicted_class)
        )

    def report(
        self,
        tag: str = "",
    ) -> float | None:
        """Calculate PSI and print drift status."""
        if (
            len(self.confidence_buffer)
            < ROLLING_N
        ):
            print(
                "[PSI] filling rolling window "
                f"{len(self.confidence_buffer)}/{ROLLING_N}"
            )

            return None

        actual = histogram(
            np.asarray(
                self.confidence_buffer,
                dtype=np.float64,
            )
        )

        psi_value = calculate_psi(
            self.expected,
            actual,
        )

        class_array = np.asarray(
            self.class_buffer,
            dtype=np.int32,
        )

        critical_share = float(
            np.mean(class_array == 2)
        )

        bin_summary = " ".join(
            f"{label}={probability * 100:4.1f}%"
            for label, probability in zip(
                BIN_LABELS,
                actual,
            )
        )

        print(
            f"[PSI] {tag:<10s} "
            f"PSI={psi_value:.3f}  "
            f"critical_share={critical_share * 100:5.1f}%  "
            f"{bin_summary}"
        )

        if psi_value > PSI_ALERT:
            if critical_share > 0.30:
                classification = (
                    "REAL CONCEPT DRIFT / FAULT PRESENT"
                )
            else:
                classification = (
                    "DATA OR SENSOR DRIFT / INVESTIGATE PROBE"
                )

            if not self.alerting:
                print(
                    "[LOGIBRIDGE DRIFT ALERT] "
                    f"PSI={psi_value:.3f}"
                )

            print(
                "   classification: "
                f"{classification}"
            )

            self.alerting = True

        elif (
            self.alerting
            and psi_value < PSI_CLEAR
        ):
            print(
                "[LOGIBRIDGE] drift cleared: "
                f"PSI={psi_value:.3f} < {PSI_CLEAR:.2f}"
            )

            self.alerting = False

        return psi_value


def run_simulation(
    model_path: Path,
    reference: dict[str, Any],
    score: str = "normal_prob",
    clean_minutes: int = 25,
    drift_minutes: int = 10,
    recovery_minutes: int = 25,
) -> None:
    """Run the clean, fault-injection and recovery PSI demonstration."""
    scorer = Scorer(
        model_path=model_path,
        score=score,
    )

    monitor = PSIMonitor(
        reference
    )

    trace: list[dict[str, Any]] = []

    phases = [
        (
            "CLEAN",
            "none",
            clean_minutes,
            3000,
        ),
        (
            "INJECTED",
            "combined",
            drift_minutes,
            3100,
        ),
        (
            "RECOVERED",
            "none",
            recovery_minutes,
            3200,
        ),
    ]

    chunk_minutes = 5

    print(
        "\n[SIM] phases: "
        f"clean {clean_minutes} min -> "
        f"combined anomaly {drift_minutes} min -> "
        f"clean {recovery_minutes} min"
    )

    print(
        f"[SIM] score={score}; "
        "PSI reported every simulated 60 seconds\n"
    )

    global_time_seconds = 0.0

    for (
        phase_name,
        anomaly_mode,
        phase_minutes,
        phase_seed,
    ) in phases:
        print(
            f"--- phase {phase_name} "
            f"(anomaly={anomaly_mode}) ---"
        )

        chunk_count = int(
            np.ceil(
                phase_minutes / chunk_minutes
            )
        )

        for chunk_index in range(
            chunk_count
        ):
            remaining_minutes = (
                phase_minutes
                - chunk_index * chunk_minutes
            )

            duration_seconds = (
                60
                * min(
                    chunk_minutes,
                    remaining_minutes,
                )
            )

            simulator = ColdChainSimulator(
                anomaly_mode,
                f"TRK-{chunk_index:02d}",
                seed=phase_seed + chunk_index,
                onset_s=0.0,
            )

            last_report_time = -60.0

            for (
                local_time,
                predicted_class,
                confidence,
            ) in scorer.score_stream(
                simulator,
                duration_seconds,
            ):
                monitor.push(
                    predicted_class,
                    confidence,
                )

                if (
                    local_time - last_report_time
                    >= 60.0
                ):
                    last_report_time = local_time

                    psi_value = monitor.report(
                        phase_name
                    )

                    if psi_value is not None:
                        trace.append(
                            {
                                "phase": phase_name,
                                "t_min": round(
                                    (
                                        global_time_seconds
                                        + local_time
                                    )
                                    / 60.0,
                                    1,
                                ),
                                "psi": round(
                                    psi_value,
                                    4,
                                ),
                            }
                        )

            global_time_seconds += (
                duration_seconds
            )

    trace_path = (
        MONITORING_DIR / "psi_trace.json"
    )

    score_trace_path = (
        MONITORING_DIR
        / f"psi_trace_{score}.json"
    )

    trace_json = json.dumps(
        trace,
        indent=2,
    )

    trace_path.write_text(
        trace_json,
        encoding="utf-8",
    )

    score_trace_path.write_text(
        trace_json,
        encoding="utf-8",
    )

    print_psi_summary(
        trace=trace,
        output_path=trace_path,
        score=score,
    )


def print_psi_summary(
    trace: list[dict[str, Any]],
    output_path: Path,
    score: str = "normal_prob",
) -> None:
    """Print the Task E1 summary and save auditable PSI metadata."""
    clean_values = [
        float(row["psi"])
        for row in trace
        if row["phase"] == "CLEAN"
    ]

    injected_rows = [
        row
        for row in trace
        if row["phase"] == "INJECTED"
    ]

    recovered_values = [
        float(row["psi"])
        for row in trace
        if row["phase"] == "RECOVERED"
    ]

    clean_maximum = max(clean_values, default=0.0)
    injected_maximum: float | None = None
    recovered_final: float | None = None
    detection_lag: float | None = None

    print(
        "\n=========== PSI SUMMARY ==========="
    )

    print(
        "  clean maximum PSI: "
        f"{clean_maximum:.3f}"
    )

    if injected_rows:
        injection_start = min(
            float(row["t_min"])
            for row in injected_rows
        )

        first_crossing = next(
            (
                float(row["t_min"])
                for row in injected_rows
                if float(row["psi"]) > PSI_ALERT
            ),
            None,
        )

        if first_crossing is not None:
            detection_lag = round(
                first_crossing
                - injection_start
                + 1.0,
                1,
            )

        injected_maximum = max(
            float(row["psi"])
            for row in injected_rows
        )

        detection_status = (
            "PASS"
            if (
                detection_lag is not None
                and detection_lag <= 5.0
            )
            else "FAIL"
        )

        lag_text = (
            "not detected"
            if detection_lag is None
            else f"{detection_lag:.1f} min"
        )

        print(
            "  injected maximum PSI: "
            f"{injected_maximum:.3f}"
        )

        print(
            "  PSI 0.25 detection lag: "
            f"{lag_text} ({detection_status}; "
            "requirement less than or equal to 5 min)"
        )

    if recovered_values:
        recovered_final = recovered_values[-1]

        recovery_status = (
            "PASS"
            if recovered_final < PSI_CLEAR
            else "STILL HIGH"
        )

        print(
            "  recovered final PSI: "
            f"{recovered_final:.3f} "
            f"({recovery_status}; required below "
            f"{PSI_CLEAR:.2f})"
        )

    simulation_passed = (
        injected_maximum is not None
        and injected_maximum > PSI_ALERT
        and detection_lag is not None
        and detection_lag <= 5.0
        and recovered_final is not None
        and recovered_final < PSI_CLEAR
    )

    metadata = {
        "reference_windows": int(REF_WINDOWS),
        "rolling_window": int(ROLLING_N),
        "evaluation_interval_seconds": 60,
        "warning_threshold": float(PSI_CLEAR),
        "drift_threshold": float(PSI_ALERT),
        "clean_maximum_psi": float(clean_maximum),
        "injected_maximum_psi": (
            None
            if injected_maximum is None
            else float(injected_maximum)
        ),
        "recovered_final_psi": (
            None
            if recovered_final is None
            else float(recovered_final)
        ),
        "detection_lag_minutes": (
            None
            if detection_lag is None
            else float(detection_lag)
        ),
        "score": score,
        "bins": [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ],
        "bin_labels": list(BIN_LABELS),
        "reference_source": (
            f"{REF_WINDOWS}_clean_normal_condition_windows"
        ),
        "simulation_status": (
            "drift_detected_and_recovered"
            if simulation_passed
            else "validation_failed"
        ),
        "trace_file": str(output_path),
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PSI_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"  trace output: {output_path}"
    )

    print(
        f"  metadata output: {PSI_METADATA_PATH}"
    )

    if not simulation_passed:
        raise RuntimeError(
            "PSI simulation did not satisfy all validation gates. "
            "Inspect psi_trace.json and psi_metadata.json."
        )

    print(
        "[PASS] PSI simulation completed"
    )

def run_live(
    reference: dict[str, Any],
    host: str,
    port: int,
    interval: int = 60,
    score: str = "normal_prob",
) -> None:
    """Run live MQTT PSI monitoring."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as error:
        raise RuntimeError(
            "paho-mqtt is not installed. Run:\n"
            "  pip install paho-mqtt"
        ) from error

    monitor = PSIMonitor(
        reference
    )

    def on_connect(
        client: Any,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        """Subscribe after connecting to MQTT."""
        del userdata, flags, properties

        if int(reason_code) != 0:
            print(
                "[MQTT] connection failed: "
                f"{reason_code}"
            )

            return

        client.subscribe(
            "logibridge/trucks/+/inference",
            qos=1,
        )

        print(
            "[MQTT] subscribed to "
            "logibridge/trucks/+/inference"
        )

    def on_message(
        client: Any,
        userdata: Any,
        message: Any,
    ) -> None:
        """Process one inference MQTT message."""
        del client, userdata

        try:
            record = json.loads(
                message.payload.decode(
                    "utf-8"
                )
            )

            probabilities = record.get(
                "probs"
            )

            if probabilities is not None:
                confidence = confidence_score(
                    np.asarray(
                        probabilities,
                        dtype=np.float64,
                    ),
                    score,
                )
            else:
                confidence = float(
                    record["confidence"]
                )

            predicted_class = int(
                record["class"]
            )

            monitor.push(
                predicted_class,
                confidence,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            print(
                "[MQTT] invalid inference message: "
                f"{error}"
            )

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="psi-monitor",
    )

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(
        host,
        port,
        30,
    )

    client.loop_start()

    print(
        f"[PSI] live monitor: "
        f"mqtt://{host}:{port}"
    )

    print(
        f"[PSI] reporting interval: "
        f"{interval} seconds"
    )

    try:
        while True:
            time.sleep(
                interval
            )

            monitor.report(
                "LIVE"
            )

    except KeyboardInterrupt:
        print(
            "\n[STOP] Live PSI monitor stopped."
        )

    finally:
        client.loop_stop()
        client.disconnect()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Build, simulate or run live PSI drift monitoring."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "reference",
            "simulate",
            "live",
        ],
        default="simulate",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
    )

    parser.add_argument(
        "--host",
        default="localhost",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=1883,
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--score",
        choices=[
            "normal_prob",
            "max_prob",
        ],
        default="normal_prob",
        help=(
            "Output score used to build the PSI histogram."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Execute the selected PSI monitoring mode."""
    arguments = parse_arguments()

    model_path = Path(
        arguments.model
    ).resolve()

    validate_common_files(
        model_path
    )

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if arguments.mode == "reference":
        build_reference(
            model_path=model_path,
            score=arguments.score,
        )

        return 0

    reference = load_reference(
        expected_score=arguments.score,
        expected_model=model_path,
    )

    if reference is None:
        print(
            "[PSI] Building a new reference for "
            f"score={arguments.score!r} and "
            f"model={model_path.name!r}"
        )

        reference = build_reference(
            model_path=model_path,
            score=arguments.score,
        )

    if arguments.mode == "simulate":
        run_simulation(
            model_path=model_path,
            reference=reference,
            score=arguments.score,
        )

    elif arguments.mode == "live":
        run_live(
            reference=reference,
            host=arguments.host,
            port=arguments.port,
            interval=arguments.interval,
            score=arguments.score,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "\n[STOP] Drift monitor interrupted by user."
        )

        raise SystemExit(130)

    except Exception as error:
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )

        raise SystemExit(1)
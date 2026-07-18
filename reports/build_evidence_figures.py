
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    """Load and return a JSON file with a clear error message."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file was not found: {path}"
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON file: {path}\n{exc}"
        ) from exc


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows while normalising header whitespace."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required CSV file was not found: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV file has no header row: {path}"
            )

        rows: list[dict[str, str]] = []

        for raw_row in reader:
            cleaned_row = {
                str(key).strip(): (
                    str(value).strip()
                    if value is not None
                    else ""
                )
                for key, value in raw_row.items()
                if key is not None
            }

            if any(cleaned_row.values()):
                rows.append(cleaned_row)

    if not rows:
        raise RuntimeError(
            f"No data rows were found in CSV file: {path}"
        )

    return rows


def get_first_value(
    row: dict[str, Any],
    possible_keys: tuple[str, ...],
    *,
    default: Any = None,
) -> Any:
    """Return the first available non-empty value."""

    for key in possible_keys:
        if key not in row:
            continue

        value = row[key]

        if value is None:
            continue

        if isinstance(value, str):
            value = value.strip()

            if not value:
                continue

        return value

    return default


def get_text_value(
    row: dict[str, Any],
    possible_keys: tuple[str, ...],
    *,
    default: str,
) -> str:
    """Return a text value using supported alternative column names."""

    value = get_first_value(
        row,
        possible_keys,
        default=default,
    )

    return str(value).strip()


def get_float_value(
    row: dict[str, Any],
    possible_keys: tuple[str, ...],
) -> float:
    """Return a numeric value using supported alternative column names."""

    value = get_first_value(
        row,
        possible_keys,
        default=None,
    )

    if value is None:
        raise KeyError(
            "Required numeric column was not found. "
            f"Accepted columns: {possible_keys}. "
            f"Available columns: {sorted(row.keys())}"
        )

    text = str(value).strip()

    if text.endswith("%"):
        text = text[:-1].strip()

    text = text.replace(",", "")

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Could not convert value {value!r} to float. "
            f"Accepted columns: {possible_keys}"
        ) from exc


# ---------------------------------------------------------------------------
# 1. Exact and grouped dataset distribution
# ---------------------------------------------------------------------------

assignment = load_json(
    ROOT
    / "training"
    / "assignment_dataset_summary.json"
)

split = load_json(
    ROOT
    / "training"
    / "split_manifest.json"
)

labels = [
    "Normal",
    "Warning",
    "Critical",
]

exact = [
    assignment["classes"][index]["windows"]
    for index in range(3)
]

train = [
    split["train_class_counts"].get(
        str(index),
        0,
    )
    for index in range(3)
]

validation = [
    split["validation_class_counts"].get(
        str(index),
        0,
    )
    for index in range(3)
]

x = np.arange(3)

fig, ax = plt.subplots(
    figsize=(9, 5.2),
)

ax.bar(
    x - 0.25,
    exact,
    0.25,
    label="Exact Task D1 run",
)

ax.bar(
    x,
    train,
    0.25,
    label="Grouped training",
)

ax.bar(
    x + 0.25,
    validation,
    0.25,
    label="Grouped validation",
)

for bars in ax.containers:
    ax.bar_label(
        bars,
        fontsize=8,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Feature windows")
ax.set_title(
    "LogiEdge dataset composition and leakage-safe holdout"
)
ax.legend()
ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUT / "dataset_distribution.png",
    dpi=180,
)

plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Confusion matrices
# ---------------------------------------------------------------------------

metrics: list[tuple[str, np.ndarray]] = []

for name, filename in [
    ("M1 FP32", "m1_metrics.json"),
    ("M2 INT8", "m2_metrics.json"),
    ("M3 Pruned INT8", "m3_metrics.json"),
]:
    metric_data = load_json(
        ROOT
        / "training"
        / "models"
        / filename
    )

    confusion = np.asarray(
        metric_data["confusion"],
    )

    if confusion.shape != (3, 3):
        raise ValueError(
            f"{filename} confusion matrix must be 3 x 3, "
            f"but found shape {confusion.shape}."
        )

    metrics.append(
        (
            name,
            confusion,
        )
    )

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4.3),
)

for ax, (name, confusion) in zip(
    axes,
    metrics,
):
    ax.imshow(
        confusion,
        cmap="Blues",
    )

    maximum_value = float(
        confusion.max()
    )

    for row_index in range(3):
        for column_index in range(3):
            value = confusion[
                row_index,
                column_index,
            ]

            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if maximum_value > 0
                    and value > maximum_value / 2
                    else "black"
                ),
                fontsize=10,
            )

    ax.set_title(name)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

fig.suptitle(
    "Grouped validation confusion matrices"
)

fig.tight_layout()

fig.savefig(
    OUT / "confusion_matrices.png",
    dpi=180,
)

plt.close(fig)


# ---------------------------------------------------------------------------
# 3. PSI clean-injection-recovery trace
# ---------------------------------------------------------------------------

trace = load_json(
    ROOT
    / "monitoring"
    / "psi_trace_normal_prob.json"
)

if not isinstance(trace, list) or not trace:
    raise RuntimeError(
        "PSI trace must contain a non-empty list."
    )

phase_color = {
    "CLEAN": "#2e86de",
    "INJECTED": "#c0392b",
    "RECOVERED": "#27ae60",
}

fig, ax = plt.subplots(
    figsize=(10, 4.8),
)

for phase in [
    "CLEAN",
    "INJECTED",
    "RECOVERED",
]:
    phase_rows = [
        row
        for row in trace
        if str(
            row.get(
                "phase",
                "",
            )
        ).upper()
        == phase
    ]

    if not phase_rows:
        print(
            f"Warning: no PSI rows found for phase {phase}."
        )
        continue

    ax.plot(
        [
            float(row["t_min"])
            for row in phase_rows
        ],
        [
            float(row["psi"])
            for row in phase_rows
        ],
        marker="o",
        markersize=3,
        linewidth=1.8,
        label=phase.title(),
        color=phase_color[phase],
    )

ax.axhline(
    0.25,
    linestyle="--",
    linewidth=1.4,
    color="#c0392b",
    label="Drift alert 0.25",
)

ax.axhline(
    0.10,
    linestyle=":",
    linewidth=1.4,
    color="#27ae60",
    label="Recovery 0.10",
)

ax.set_xlabel(
    "Simulated elapsed time (minutes)"
)

ax.set_ylabel("PSI")

ax.set_title(
    "PSI response: clean operation, "
    "combined fault injection and recovery"
)

ax.grid(alpha=0.25)
ax.legend(ncol=2)

fig.tight_layout()

fig.savefig(
    OUT / "psi_drift_recovery.png",
    dpi=180,
)

plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Normalisation sensitivity
# ---------------------------------------------------------------------------

normalisation_csv = (
    ROOT
    / "experiments"
    / "normalisation_experiment.csv"
)

rows = load_csv_rows(
    normalisation_csv
)

available_columns = sorted(
    {
        key
        for row in rows
        for key in row.keys()
    }
)

print(
    "Normalisation experiment columns:",
    available_columns,
)

scenarios = [
    get_text_value(
        row,
        (
            "normalisation",
            "normalization",
            "normalisation_method",
            "normalization_method",
            "normalisation_type",
            "normalization_type",
            "scenario",
            "experiment",
            "model",
            "variant",
            "method",
            "name",
        ),
        default=f"Scenario {index + 1}",
    )
    for index, row in enumerate(rows)
]

accuracy = [
    get_float_value(
        row,
        (
            "accuracy_pct",
            "accuracy_percent",
            "accuracy_percentage",
            "accuracy",
            "validation_accuracy_pct",
            "validation_accuracy",
            "val_accuracy_pct",
            "val_accuracy",
        ),
    )
    for row in rows
]

critical_recall = [
    get_float_value(
        row,
        (
            "recall_critical_pct",
            "critical_recall_pct",
            "critical_recall_percent",
            "critical_recall_percentage",
            "critical_recall",
            "class_2_recall_pct",
            "class_2_recall",
            "recall_class_2_pct",
            "recall_class_2",
        ),
    )
    for row in rows
]

# Convert fractional metrics such as 0.92 into percentages.
accuracy = [
    value * 100.0
    if 0.0 <= value <= 1.0
    else value
    for value in accuracy
]

critical_recall = [
    value * 100.0
    if 0.0 <= value <= 1.0
    else value
    for value in critical_recall
]

x = np.arange(
    len(rows)
)

fig, ax = plt.subplots(
    figsize=(8.6, 4.8),
)

accuracy_bars = ax.bar(
    x - 0.18,
    accuracy,
    0.36,
    label="Accuracy (%)",
)

critical_bars = ax.bar(
    x + 0.18,
    critical_recall,
    0.36,
    label="Critical recall (%)",
)

ax.axhline(
    88,
    linestyle="--",
    linewidth=1,
    label="88% accuracy gate",
)

ax.axhline(
    95,
    linestyle=":",
    linewidth=1.2,
    label="95% Critical recall floor",
)

ax.set_xticks(x)

ax.set_xticklabels(
    scenarios,
    rotation=20,
    ha="right",
)

ax.set_ylim(
    0,
    max(
        105,
        max(
            accuracy
            + critical_recall
        )
        + 5,
    ),
)

ax.set_ylabel("Percent")

ax.set_title(
    "Mandatory frozen-statistics sensitivity experiment"
)

ax.legend(
    ncol=2,
    fontsize=8,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.bar_label(
    accuracy_bars,
    fmt="%.1f",
    fontsize=8,
)

ax.bar_label(
    critical_bars,
    fmt="%.1f",
    fontsize=8,
)

fig.tight_layout()

fig.savefig(
    OUT / "normalisation_sensitivity.png",
    dpi=180,
)

plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Hardware fleet cost and power
# ---------------------------------------------------------------------------

hardware_names = [
    "Raspberry Pi 5\n+ AI HAT+",
    "Jetson Orin\nNano Super",
    "STM32H7\ncustom MCU",
]

unit_cost = np.asarray(
    [
        15000,
        45000,
        3500,
    ],
    dtype=float,
)

pilot_cost_lakh = (
    unit_cost
    * 85
    / 100000
)

fleet_cost_lakh = (
    unit_cost
    * 265
    / 100000
)

power_watts = [
    7.5,
    15.0,
    0.4,
]

x = np.arange(3)

fig, ax = plt.subplots(
    figsize=(9.5, 5.2),
)

pilot_bars = ax.bar(
    x - 0.2,
    pilot_cost_lakh,
    0.4,
    label="85-truck pilot (lakh Rs)",
)

fleet_bars = ax.bar(
    x + 0.2,
    fleet_cost_lakh,
    0.4,
    label="265-truck scale (lakh Rs)",
)

ax.set_xticks(x)
ax.set_xticklabels(hardware_names)

ax.set_ylabel(
    "Hardware acquisition cost (lakh Rs)"
)

ax.bar_label(
    pilot_bars,
    fmt="%.2f",
    fontsize=8,
)

ax.bar_label(
    fleet_bars,
    fmt="%.2f",
    fontsize=8,
)

power_axis = ax.twinx()

power_axis.plot(
    x,
    power_watts,
    marker="o",
    linewidth=2,
    label="TDP / moderate load (W)",
)

power_axis.axhline(
    10,
    linestyle="--",
    linewidth=1.2,
    label="10 W budget",
)

power_axis.set_ylabel(
    "Power (W)"
)

cost_lines, cost_labels = (
    ax.get_legend_handles_labels()
)

power_lines, power_labels = (
    power_axis.get_legend_handles_labels()
)

ax.legend(
    cost_lines + power_lines,
    cost_labels + power_labels,
    fontsize=8,
    loc="upper left",
)

ax.set_title(
    "Constraint Triangle evidence: "
    "cost, power and deployability"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUT / "hardware_cost_power.png",
    dpi=180,
)

plt.close(fig)


print(
    f"Wrote figures to {OUT}"
)

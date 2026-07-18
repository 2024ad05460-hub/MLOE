"""Every number quoted in the reports, computed here so it can be re-derived and
challenged rather than asserted.

Run:
    python experiments/constraint_numbers.py
"""

SEC_PER_DAY = 86_400
RATE_RS_PER_MB = 0.10
PILOT, FLEET = 85, 265

# ---------------------------------------------------------------------------
# Roofline constants required by the assignment
# ---------------------------------------------------------------------------
MODEL_FLOPS = 45_000_000
DATA_BYTES = 18_000_000
COMPUTE_FLOPS_PER_SECOND = 16_000_000_000
BANDWIDTH_BYTES_PER_SECOND = 12_000_000_000


def bandwidth() -> None:
    """Calculate raw-stream and edge-uplink bandwidth and cost."""

    # Raw streams, 4-byte samples (float32).
    # This is a lower bound; JSON-over-MQTT is approximately 3x.
    temp_bps = 1 * 4
    vib_bps = 500 * 3 * 4
    door_bps = 20 * 8 / SEC_PER_DAY

    raw_bps = temp_bps + vib_bps + door_bps
    raw_mb_day = raw_bps * SEC_PER_DAY / 1e6

    # Edge-processed uplink:
    # state transitions, five-minute heartbeats, and alerts only.
    heartbeats = SEC_PER_DAY / 300 * 200
    alerts = 20 * 300
    edge_mb_day = (heartbeats + alerts) / 1e6

    print("=== A1 BANDWIDTH ===")
    print(
        f"raw per truck/day      : {raw_mb_day:,.1f} MB   "
        f"(temp {temp_bps} B/s + vib {vib_bps} B/s)"
    )
    print(
        f"raw cost/truck/day     : "
        f"Rs {raw_mb_day * RATE_RS_PER_MB:,.2f}"
    )
    print(
        f"raw cost 85 trucks/yr  : "
        f"Rs {raw_mb_day * RATE_RS_PER_MB * PILOT * 365:,.0f}"
    )
    print(
        f"raw cost 265 trucks/yr : "
        f"Rs {raw_mb_day * RATE_RS_PER_MB * FLEET * 365:,.0f}"
    )
    print(
        f"edge per truck/day     : "
        f"{edge_mb_day * 1000:,.1f} KB"
    )
    print(
        f"edge cost 85 trucks/yr : "
        f"Rs {edge_mb_day * RATE_RS_PER_MB * PILOT * 365:,.0f}"
    )
    print(
        f"reduction factor       : "
        f"{raw_mb_day / edge_mb_day:,.0f}x "
        f"({100 * (1 - edge_mb_day / raw_mb_day):.4f}% less data)"
    )
    print(
        f"JSON-over-MQTT reality : "
        f"~3x the raw figure ({raw_mb_day * 3:,.0f} MB/day) "
        f"if published un-reduced"
    )
    print()


def roofline() -> dict[str, float | str]:
    """Calculate and print the validated Roofline classification.

    Returns
    -------
    dict
        The calculated arithmetic intensity, ridge point, classification,
        and latency estimates. Returning the values makes the function easy
        to test without parsing console text.
    """

    arithmetic_intensity = MODEL_FLOPS / DATA_BYTES

    ridge_point = (
        COMPUTE_FLOPS_PER_SECOND
        / BANDWIDTH_BYTES_PER_SECOND
    )

    classification = (
        "compute-bound"
        if arithmetic_intensity > ridge_point
        else "memory-bandwidth-bound"
    )

    compute_latency_ms = (
        MODEL_FLOPS
        / COMPUTE_FLOPS_PER_SECOND
    ) * 1000.0

    memory_latency_ms = (
        DATA_BYTES
        / BANDWIDTH_BYTES_PER_SECOND
    ) * 1000.0

    roofline_latency_ms = max(
        compute_latency_ms,
        memory_latency_ms,
    )

    attainable_flops_per_second = min(
        COMPUTE_FLOPS_PER_SECOND,
        BANDWIDTH_BYTES_PER_SECOND * arithmetic_intensity,
    )

    print("=== B2 ARITHMETIC INTENSITY / ROOFLINE ===")
    print("Model operations: 45 MFLOPs")
    print("Data movement: 18 MB")
    print(
        f"Arithmetic Intensity: "
        f"{arithmetic_intensity:.3f} FLOP/byte"
    )
    print(
        f"Ridge Point: "
        f"{ridge_point:.3f} FLOP/byte"
    )
    print(f"Classification: {classification}")
    print(
        f"Attainable performance: "
        f"{attainable_flops_per_second / 1e9:.3f} GFLOP/s"
    )
    print(
        f"Compute latency estimate: "
        f"{compute_latency_ms:.3f} ms"
    )
    print(
        f"Memory latency estimate: "
        f"{memory_latency_ms:.3f} ms"
    )
    print(
        f"Roofline latency estimate: "
        f"{roofline_latency_ms:.3f} ms"
    )
    print(
        f"Latency margin versus 90 s SLA: "
        f"{90_000.0 / roofline_latency_ms:,.0f}x"
    )
    print()

    return {
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "classification": classification,
        "compute_latency_ms": compute_latency_ms,
        "memory_latency_ms": memory_latency_ms,
        "roofline_latency_ms": roofline_latency_ms,
        "attainable_flops_per_second": attainable_flops_per_second,
    }


def ota() -> None:
    """Calculate OTA update bandwidth and cost."""

    model_kb_spec = 280
    model_kb_actual = 3.77

    for label, kb in (
        ("spec 280 KB", model_kb_spec),
        ("actual M3", model_kb_actual),
    ):
        mb = kb / 1024
        full = mb * PILOT * RATE_RS_PER_MB
        canary = (
            mb * 10 * RATE_RS_PER_MB
            + mb * 75 * RATE_RS_PER_MB
        )

        # Ship model to all trucks, then retain seven days of
        # dual-inference comparison telemetry.
        shadow_tel_mb = (
            8_640
            * 40
            * 7
            / 1e6
            * PILOT
        )
        shadow = (
            full
            + shadow_tel_mb * RATE_RS_PER_MB
        )

        print(
            f"=== E3 OTA per update cycle "
            f"({label}, {PILOT} trucks) ==="
        )
        print(
            f"  full replacement : "
            f"{mb * PILOT:8.3f} MB   "
            f"Rs {full:8.3f}"
        )
        print(
            f"  canary 10 -> 75  : "
            f"{mb * PILOT:8.3f} MB   "
            f"Rs {canary:8.3f} "
            f"(same bytes, staged)"
        )
        print(
            f"  shadow mode      : "
            f"{mb * PILOT + shadow_tel_mb:8.1f} MB   "
            f"Rs {shadow:8.2f} "
            f"(model + 7 d dual-inference telemetry)"
        )
        print(
            f"  annual (8.7 cycles): "
            f"full Rs {full * 8.7:.2f} | "
            f"canary Rs {canary * 8.7:.2f} | "
            f"shadow Rs {shadow * 8.7:.2f}"
        )

    print(
        "\n  -> The three strategies differ by rupees. "
        "The decision is therefore not financial:"
    )
    print(
        "     one bad model x 85 trucks x Rs 28 lakh cargo "
        "dwarfs the entire OTA budget."
    )
    print()


def docker() -> None:
    """Calculate Docker layer-cache transfer savings."""

    image_mb = 150.0
    model_mb = 3.77 / 1024

    full = image_mb * PILOT * RATE_RS_PER_MB
    cached = model_mb * PILOT * RATE_RS_PER_MB

    print("=== D2 DOCKER LAYER CACHE (85 trucks) ===")
    print(
        f"  naive full image  : "
        f"{image_mb * PILOT:8.1f} MB   "
        f"Rs {full:8.2f}"
    )
    print(
        f"  cached model layer: "
        f"{model_mb * PILOT:7.3f} MB   "
        f"Rs {cached:8.4f}"
    )
    print(
        f"  saving            : "
        f"{image_mb * PILOT - model_mb * PILOT:8.1f} MB   "
        f"Rs {full - cached:8.2f}   "
        f"({100 * (1 - model_mb / image_mb):.3f}% less)"
    )
    print(
        f"  per truck         : "
        f"{image_mb:.0f} MB -> {model_mb * 1024:.1f} KB"
    )


def main() -> int:
    """Run all assignment constraint calculations."""

    bandwidth()
    roofline()
    ota()
    docker()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
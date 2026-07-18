"""Generates scenario_architecture/system_architecture.png (Task A2) and
data_pipeline/mqtt_topic_tree.png (Report Section 2)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]

EDGE = "#1b4f72"
SENSOR = "#117864"
CLOUD = "#7d6608"
ALERT = "#922b21"
GREY = "#566573"


def box(ax, x, y, w, h, text, fc, ec=None, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec or "#2c3e50", lw=1.1, alpha=0.95))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="white" if bold else "#1c2833",
            fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(ax, p, q, label=None, style="-|>", color="#2c3e50", ls="-", off=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 lw=1.2, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + off, (p[1] + q[1]) / 2 + 0.012, label,
                ha="center", fontsize=7.2, color=color, style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.975, "LogiEdge - Cold-Chain Edge AI Architecture (per truck)",
            ha="center", fontsize=14, fontweight="bold", color="#1c2833")
    ax.text(0.5, 0.945,
            "Raspberry Pi 5 (8 GB) + AI HAT+ | 7.5 W | everything inside the dashed "
            "boundary runs with the modem unplugged",
            ha="center", fontsize=9, color=GREY, style="italic")

    # ---- truck boundary
    ax.add_patch(Rectangle((0.02, 0.09), 0.60, 0.82, fill=False, ec=EDGE,
                           lw=1.8, ls="--"))
    ax.text(0.04, 0.885, "TRUCK — ON-BOARD EDGE NODE", fontsize=9.5,
            fontweight="bold", color=EDGE)

    # ---- sensors
    ax.text(0.075, 0.83, "SENSORS", fontsize=8, fontweight="bold", color=SENSOR)
    box(ax, 0.04, 0.70, 0.155, 0.10,
        "Cargo probe\nPT100 / DS18B20\n1 Hz  ±0.3 °C", "#d1f2eb", SENSOR)
    box(ax, 0.04, 0.575, 0.155, 0.10,
        "Compressor IMU\nADXL355 3-axis\n500 Hz → RMS 0.5 Hz", "#d1f2eb", SENSOR)
    box(ax, 0.04, 0.45, 0.155, 0.10,
        "Door reed switch\ndiscrete OPEN/CLOSE\ninterrupt-driven", "#d1f2eb", SENSOR)

    # ---- edge software stack
    box(ax, 0.235, 0.575, 0.135, 0.225,
        "Mosquitto\nLOCAL BROKER\n:1883\n\nlocalhost only\npersistence on\nQoS 0/1/2",
        "#d6eaf8", EDGE)
    arrow(ax, (0.195, 0.75), (0.235, 0.71), "1 Hz")
    arrow(ax, (0.195, 0.625), (0.235, 0.665), "0.5 Hz")
    arrow(ax, (0.195, 0.50), (0.235, 0.60), "events")

    box(ax, 0.41, 0.665, 0.185, 0.135,
        "PREPROCESSING\n5-tap moving average\n30 s window / 10 s step\n"
        "6 features (fusion)\nz-score · frozen stats", "#d6eaf8", EDGE)
    arrow(ax, (0.37, 0.72), (0.41, 0.73), "sub")

    box(ax, 0.41, 0.50, 0.185, 0.125,
        "INFERENCE  (Docker)\nTFLite INT8 · 3.70 KB\n1.42 µs / window\n"
        "Normal | Warning | Critical", "#d6eaf8", EDGE)
    arrow(ax, (0.5, 0.665), (0.5, 0.625), "6-vector")

    box(ax, 0.235, 0.30, 0.16, 0.13,
        "LOCAL ALERT LOG\nSQLite (WAL)\nalerts.db\ninference + alert ACK flags\n"
        "system of record", "#fadbd8", ALERT)
    box(ax, 0.42, 0.30, 0.175, 0.13,
        "PSI DRIFT MONITOR\nrolling 100 · every 60 s\nalert if PSI > 0.25\n"
        "health-score histogram", "#fdebd0", CLOUD)
    arrow(ax, (0.46, 0.50), (0.36, 0.43), "write FIRST")
    arrow(ax, (0.53, 0.50), (0.52, 0.43), "confidence")

    box(ax, 0.235, 0.135, 0.36, 0.10,
        "STORE-AND-FORWARD SYNC  ·  publishes when the link returns, oldest-first\n"
        "chain-of-custody order preserved for hospital audit",
        "#fadbd8", ALERT, fs=8)
    arrow(ax, (0.315, 0.30), (0.315, 0.235))
    arrow(ax, (0.5, 0.30), (0.5, 0.235))

    # ---- uplink
    box(ax, 0.655, 0.44, 0.10, 0.30,
        "CELLULAR\nUPLINK\n\nM2M SIM\n4G/LTE\n\n35–90 min\nGAPS at 7\nlocations\n"
        "Nashik–\nAurangabad", "#f2f3f4", GREY, fs=7.8)
    arrow(ax, (0.595, 0.185), (0.655, 0.44), "alerts + heartbeat only\n~64 KB/day",
          color=ALERT, ls="--")

    # ---- ops centre
    ax.add_patch(Rectangle((0.795, 0.20), 0.185, 0.62, fill=False, ec=CLOUD,
                           lw=1.6, ls="--"))
    ax.text(0.808, 0.795, "OPERATIONS CENTRE (Pune)", fontsize=8.6,
            fontweight="bold", color=CLOUD)
    box(ax, 0.81, 0.66, 0.155, 0.10,
        "Fleet MQTT broker\nTLS · per-truck creds", "#fdebd0", CLOUD)
    box(ax, 0.81, 0.535, 0.155, 0.10,
        "Alert & escalation\nSMS/call to driver\n+ cold-chain desk", "#fdebd0", CLOUD)
    box(ax, 0.81, 0.41, 0.155, 0.10,
        "Chain-of-custody\ntime-series store\n(audit export, PDF)", "#fdebd0", CLOUD)
    box(ax, 0.81, 0.285, 0.155, 0.10,
        "MLOps: registry,\nPSI dashboard,\nAnsible + Docker\nregistry (OTA)",
        "#fdebd0", CLOUD)
    arrow(ax, (0.755, 0.62), (0.81, 0.70))
    arrow(ax, (0.885, 0.66), (0.885, 0.635))
    arrow(ax, (0.885, 0.535), (0.885, 0.51))

    arrow(ax, (0.81, 0.31), (0.755, 0.50), "OTA: 3.70 KB model layer\nAnsible, canary 10",
          color="#7d3c98", ls=":", style="-|>")

    ax.text(0.5, 0.045,
            "Offline path (bold): sensors → local broker → preprocess → INT8 inference → "
            "SQLite alert log.  No step needs the network.\n"
            "Detection-to-alarm inside the truck: ≤ 40 s window latency + 1.42 µs "
            "inference — the 90 s SLA is met with the modem switched off.",
            ha="center", fontsize=8.6, color="#1c2833",
            bbox=dict(boxstyle="round,pad=0.5", fc="#eafaf1", ec=SENSOR, lw=1))

    fig.savefig(ROOT / "scenario_architecture" / "system_architecture.png",
                dpi=170, bbox_inches="tight")
    plt.close(fig)


def topic_tree():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "LogiEdge MQTT topic tree and QoS policy", ha="center",
            fontsize=13.5, fontweight="bold")

    rows = [
        (0.865, 0.04, "logibridge/", "", "", "#d6eaf8"),
        (0.775, 0.08, "└── trucks/{truck_id}/", "", "", "#d6eaf8"),
        (0.685, 0.12, "├── sensors/temperature", "QoS 0", "1 Hz. Lossy by design: a "
         "dropped sample is re-smoothed by the 5-tap filter;\n     QoS 1 would cost "
         "an ACK per second for nothing.", "#d1f2eb"),
        (0.575, 0.12, "├── sensors/vibration", "QoS 0", "0.5 Hz RMS. Same reasoning. "
         "Raw 500 Hz never leaves the MCU - it is\n     reduced to RMS at source "
         "(6 kB/s -> 4 B/s).", "#d1f2eb"),
        (0.465, 0.12, "├── sensors/door", "QoS 1", "Discrete and rare. A LOST door-open "
         "event mis-attributes a legitimate\n     warm-air bump to a refrigeration "
         "fault. At-least-once, dedup on ts.", "#d1f2eb"),
        (0.355, 0.12, "├── inference", "QoS 1", "One per 10 s window. The ops centre "
         "needs every window for the\n     chain-of-custody record. Duplicates are "
         "idempotent (keyed on ts).", "#fdebd0"),
        (0.245, 0.12, "├── alerts", "QoS 1 + retain", "Warning/Critical escalation. "
         "Retained so a reconnecting ops-centre\n     client immediately sees the "
         "truck's last known state.", "#fadbd8"),
        (0.135, 0.12, "└── health", "QoS 1 + retain\n+ LWT", "Model version, uptime. "
         "Last Will and Testament publishes 'offline'\n     if the node dies - "
         "silence is otherwise indistinguishable from health.", "#fdebd0"),
    ]
    for y, x, topic, qos, note, fc in rows:
        ax.text(x, y, topic, fontsize=11, family="monospace", fontweight="bold",
                color="#1b4f72", va="center")
        if qos:
            ax.add_patch(FancyBboxPatch((0.40, y - 0.028), 0.115, 0.056,
                                        boxstyle="round,pad=0.008", fc=fc,
                                        ec="#2c3e50", lw=0.9))
            ax.text(0.4575, y, qos, ha="center", va="center", fontsize=8.4,
                    fontweight="bold")
        if note:
            ax.text(0.535, y, note, fontsize=8.2, va="center", color="#1c2833")

    ax.text(0.5, 0.045,
            "QoS 2 is used nowhere. Its four-way handshake doubles round trips on a "
            "link that disappears for 90 minutes at a time;\n"
            "exactly-once is bought far more cheaply with QoS 1 + an idempotent "
            "(truck_id, ts) key at the ops centre.",
            ha="center", fontsize=8.6, style="italic", color="#1c2833",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f2f3f4", ec=GREY, lw=0.9))

    fig.savefig(ROOT / "data_pipeline" / "mqtt_topic_tree.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    topic_tree()
    print("wrote scenario_architecture/system_architecture.png")
    print("wrote data_pipeline/mqtt_topic_tree.png")

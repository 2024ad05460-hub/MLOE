"""
LogiEdge - Task C1: Cold-chain truck sensor simulator.

Three streams, physically-motivated:
  temperature   1.0 Hz   N(4.0, 0.3)   setpoint 4 degC
  vibration_rms 0.5 Hz   N(0.45, 0.05) compressor RMS in g
  door_event    discrete OPEN / CLOSE with timestamp

Anomaly modes (--anomaly):
  none        healthy reefer
  temp_drift  refrigeration losing capacity: +0.08 degC per temperature reading,
              clipped at setpoint+3.0 degC  -> stays inside the Warning band
              (1-3 degC outside setpoint) as defined in the problem statement
  vibration   compressor bearing wear: vibration steps to N(1.2, 0.15)
  combined    both faults at once; the drift clip is released to setpoint+10 degC
              because a failed unit no longer holds any setpoint -> Critical

Physical coupling that is always on (all modes): an OPEN door injects a warm-air
bump of about +0.5 degC that decays back with a first-order time constant. This is
what makes the Normal class non-trivial - a naive "temperature > threshold"
rule would false-alarm on every loading bay stop.

Two execution paths share the exact same generator, so the data the model is
trained on is bit-for-bit the data it sees at inference time:

  MQTT mode (default)  publish live to a local Mosquitto broker
  Offline mode (--offline)  yield samples as fast as possible, no broker,
                            used by training/generate_dataset.py

Usage
  python simulator.py --anomaly none --duration 600 --truck-id TRK-01
  python simulator.py --anomaly combined --duration 300 --speed 10
  python simulator.py --anomaly temp_drift --duration 900 --offline --out drift.csv
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------------
# Constants - single source of truth, imported by preprocessing and training
# ----------------------------------------------------------------------------
TEMP_HZ = 1.0            # temperature sample rate
VIB_HZ = 0.5             # vibration RMS sample rate
SETPOINT_C = 4.0         # cold-chain setpoint for pharma (2-8 degC band)
TEMP_SIGMA = 0.3
VIB_NOMINAL_MEAN = 0.45  # g
VIB_NOMINAL_SIGMA = 0.05
VIB_FAULT_MEAN = 1.20    # g, bearing-wear signature
VIB_FAULT_SIGMA = 0.15
DRIFT_PER_READING = 0.08     # degC per temperature reading
WARNING_CLIP_C = 3.0         # temp_drift is clipped here -> Warning band
CRITICAL_CLIP_C = 10.0       # combined mode: unit has failed, no control left
DOOR_OPEN_PROB = 0.0015      # per temperature tick -> roughly 1 stop / 11 min
DOOR_OPEN_MEAN_S = 45.0      # mean door-open duration
DOOR_HEAT_C = 0.5            # warm-air ingress amplitude
DOOR_TAU_S = 60.0            # exponential decay time constant of the bump

MODES = ("none", "temp_drift", "vibration", "combined")

# MQTT topic tree (see data_pipeline/mqtt_architecture.md)
TOPIC_TEMP = "logibridge/trucks/{truck_id}/sensors/temperature"
TOPIC_VIB = "logibridge/trucks/{truck_id}/sensors/vibration"
TOPIC_DOOR = "logibridge/trucks/{truck_id}/sensors/door"


class ColdChainSimulator:
    """Deterministic-with-seed generator for one refrigerated truck.

    Two realism features that matter for honest accuracy numbers:

    * FAULT ONSET. A real compressor does not fail at t=0. The fault starts at
      `onset_s` (default: random 0-120 s). Windows that close before the onset in
      an anomaly run therefore look Normal but still carry the anomaly label, which
      is exactly the label noise a real cold-chain dataset has. Without this the
      model scores a meaningless 100%.
    * PER-TRUCK SENSOR BIAS. Each truck's probe has a small fixed offset
      (N(0, 0.15) degC) and its own compressor a small RMS offset - so the model
      cannot key on an absolutely-calibrated mean.
    """

    def __init__(self, anomaly="none", truck_id="TRK-01", seed=None, onset_s=None, force_door_cycle=False):
        if anomaly not in MODES:
            raise ValueError(f"anomaly must be one of {MODES}")
        self.anomaly = anomaly
        self.truck_id = truck_id
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.force_door_cycle = bool(force_door_cycle)

        # fault onset
        if onset_s is None:
            onset_s = 0.0 if anomaly == "none" else float(self.rng.uniform(0, 120))
        self.onset_s = onset_s
        # per-unit calibration bias
        self.temp_bias = float(self.rng.normal(0.0, 0.15))
        self.vib_bias = float(self.rng.normal(0.0, 0.03))

        self.drift = 0.0          # accumulated refrigeration-loss offset (degC)
        self.door_open = False
        self.door_close_at = 0.0
        self.door_bump = 0.0      # current warm-air bump (degC)
        self.t = 0.0              # simulated seconds since start

    @property
    def fault_active(self):
        return self.anomaly != "none" and self.t >= self.onset_s

    # -- individual stream models ------------------------------------------
    def _temperature(self):
        """One temperature reading at the current simulated time."""
        if self.fault_active and self.anomaly in ("temp_drift", "combined"):
            clip = CRITICAL_CLIP_C if self.anomaly == "combined" else WARNING_CLIP_C
            self.drift = min(self.drift + DRIFT_PER_READING, clip)

        # door dynamics: warm air in while open, first-order decay once closed
        dt = 1.0 / TEMP_HZ
        if self.door_open:
            self.door_bump += (DOOR_HEAT_C - self.door_bump) * (dt / 15.0)
            if self.t >= self.door_close_at:
                self.door_open = False
        else:
            self.door_bump *= math.exp(-dt / DOOR_TAU_S)
            if (not self.force_door_cycle) and self.pyrng.random() < DOOR_OPEN_PROB:
                self.door_open = True
                self.door_close_at = self.t + self.pyrng.expovariate(
                    1.0 / DOOR_OPEN_MEAN_S)

        # slow ambient/route effect (sun load, hill climbs): +/- 0.2 degC, 20 min period
        ambient = 0.2 * math.sin(2 * math.pi * self.t / 1200.0)
        noise = self.rng.normal(0.0, TEMP_SIGMA)
        return (SETPOINT_C + self.temp_bias + self.drift + self.door_bump
                + ambient + noise)

    def _vibration(self):
        if self.fault_active and self.anomaly in ("vibration", "combined"):
            # bearing wear ramps in over ~60 s rather than stepping instantly
            ramp = min(1.0, (self.t - self.onset_s) / 60.0)
            mean = VIB_NOMINAL_MEAN + ramp * (VIB_FAULT_MEAN - VIB_NOMINAL_MEAN)
            sigma = VIB_NOMINAL_SIGMA + ramp * (VIB_FAULT_SIGMA - VIB_NOMINAL_SIGMA)
            return float(self.rng.normal(mean + self.vib_bias, sigma))
        return float(self.rng.normal(VIB_NOMINAL_MEAN + self.vib_bias,
                                     VIB_NOMINAL_SIGMA))

    # -- unified sample stream ---------------------------------------------
    def stream(self, duration_s):
        """Yield (sim_time, topic_kind, payload_dict) in simulated-time order.

        Temperature ticks every 1 s, vibration every 2 s, door events are emitted
        on the edges of self.door_open. This is the ONLY generator in the project.
        """
        next_vib = 0.0
        was_open = False
        for tick in range(int(duration_s * TEMP_HZ)):
            self.t = tick / TEMP_HZ

            # Deterministic OPEN/CLOSE cycle used only by integration evidence.
            # It avoids relying on a low-probability random door event in Task 23.
            if self.force_door_cycle and self.t == 2.0:
                self.door_open = True
                self.door_close_at = 7.0

            temp = self._temperature()
            yield (self.t, "temperature",
                   {"truck_id": self.truck_id, "ts": self.t,
                    "value_c": round(float(temp), 4)})

            if self.door_open != was_open:
                yield (self.t, "door",
                       {"truck_id": self.truck_id, "ts": self.t,
                        "event": "OPEN" if self.door_open else "CLOSE"})
                was_open = self.door_open

            if self.t >= next_vib:
                yield (self.t, "vibration",
                       {"truck_id": self.truck_id, "ts": self.t,
                        "rms_g": round(self._vibration(), 4)})
                next_vib += 1.0 / VIB_HZ



def utc_timestamp() -> str:
    """Return the current UTC time in ISO-8601 format with a trailing Z."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def publish_door_event(
    mqtt_client,
    truck_id: str,
    event: str,
    timestamp: str,
) -> None:
    """Publish a validated door edge event using the assignment MQTT contract."""
    event_value = str(event).upper()

    if event_value not in {"OPEN", "CLOSE"}:
        raise ValueError(
            f"Door event must be OPEN or CLOSE; received {event!r}."
        )

    topic = TOPIC_DOOR.format(truck_id=truck_id)
    payload = {
        "truck_id": truck_id,
        "timestamp": timestamp,
        "event": event_value,
    }

    message_info = mqtt_client.publish(
        topic,
        json.dumps(payload, separators=(",", ":")),
        qos=1,
    )

    # Paho returns an MQTTMessageInfo object. rc == MQTT_ERR_SUCCESS (0)
    # means the message was accepted for transmission.
    if getattr(message_info, "rc", 0) != 0:
        raise RuntimeError(
            f"Failed to publish door event {event_value}; "
            f"MQTT return code={message_info.rc}."
        )


# ----------------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------------
def run_mqtt(sim, duration, speed, host, port):
    import paho.mqtt.client as mqtt

    topics = {
        "temperature": TOPIC_TEMP.format(truck_id=sim.truck_id),
        "vibration": TOPIC_VIB.format(truck_id=sim.truck_id),
        "door": TOPIC_DOOR.format(truck_id=sim.truck_id),
    }
    # QoS per stream - justified in mqtt_architecture.md
    qos = {"temperature": 0, "vibration": 0, "door": 1}

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"sim-{sim.truck_id}")
    client.connect(host, port, keepalive=30)
    client.loop_start()
    print(f"[SIM] {sim.truck_id} anomaly={sim.anomaly} -> mqtt://{host}:{port} "
          f"(speed x{speed})", flush=True)

    t0 = time.time()
    n = 0
    try:
        for sim_t, kind, payload in sim.stream(duration):
            wall_target = t0 + sim_t / speed
            sleep = wall_target - time.time()
            if sleep > 0:
                time.sleep(sleep)
            if kind == "door":
                publish_door_event(
                    mqtt_client=client,
                    truck_id=sim.truck_id,
                    event=payload["event"],
                    timestamp=utc_timestamp(),
                )
            else:
                client.publish(
                    topics[kind],
                    json.dumps(payload, separators=(",", ":")),
                    qos=qos[kind],
                )
            n += 1
            if kind == "temperature" and int(sim_t) % 30 == 0:
                print(f"  t={int(sim_t):5d}s  temp={payload['value_c']:6.2f} C"
                      f"  drift={sim.drift:5.2f}", flush=True)
    except KeyboardInterrupt:
        print("\n[SIM] interrupted", flush=True)
    finally:
        client.loop_stop()
        client.disconnect()
    print(f"[SIM] published {n} messages over {duration}s simulated", flush=True)


def run_offline(sim, duration, out):
    rows = [(t, k, json.dumps(p)) for t, k, p in sim.stream(duration)]
    if out:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim_time_s", "stream", "payload_json"])
            w.writerows(rows)
        print(f"[SIM] offline: {len(rows)} samples -> {out}")
    else:
        for r in rows:
            print(r[0], r[1], r[2])


def main(argv=None):
    p = argparse.ArgumentParser(description="LogiEdge cold-chain sensor simulator")
    p.add_argument("--anomaly", choices=MODES, default="none")
    p.add_argument("--duration", type=int, default=600,
                   help="simulated seconds to generate")
    p.add_argument("--truck-id", default="TRK-01")
    p.add_argument("--speed", type=float, default=1.0,
                   help="wall-clock acceleration (1.0 = real time)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--onset", type=float, default=None,
                   help="seconds before the fault begins (default: random 0-120)")
    p.add_argument(
        "--force-door-cycle",
        action="store_true",
        help="emit a deterministic OPEN event at 2 s and CLOSE event at 7 s",
    )
    p.add_argument("--offline", action="store_true",
                   help="no broker; emit samples as fast as possible")
    p.add_argument("--out", default=None, help="CSV path for --offline")
    a = p.parse_args(argv)

    sim = ColdChainSimulator(
        a.anomaly,
        a.truck_id,
        a.seed,
        onset_s=a.onset,
        force_door_cycle=a.force_door_cycle,
    )
    if a.offline:
        run_offline(sim, a.duration, a.out)
    else:
        run_mqtt(sim, a.duration, a.speed, a.host, a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
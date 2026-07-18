"""On-truck inference service with separate local sensor and remote uplink MQTT.

The local broker remains available when cellular coverage disappears. Sensor
acquisition and inference therefore continue. Every inference is inserted into
SQLite before any remote publish. A row is marked synchronized only after the
remote broker acknowledges the QoS 1 publish. Inference records, complete class
probabilities and required alerts are replayed oldest-first after reconnection.
"""
from __future__ import annotations
import json
import os
import signal
import sqlite3
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
for path in (HERE, ROOT / "data_pipeline", ROOT / "optimisation", ROOT / "monitoring"):
    sys.path.insert(0, str(path))

from preprocessing import CLASS_NAMES, Normaliser, WindowFeaturiser  # noqa: E402
from psi import RollingPSI  # noqa: E402
from tflite_eval import TFLiteModel  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", "/data/model.tflite")
STATS_PATH = os.getenv("STATS_PATH", "/data/training_stats.npy")
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "/data/reference_dist.json")
ALERT_DB = os.getenv("ALERT_DB", "/data/alerts.db")
TRUCK_ID = os.getenv("TRUCK_ID", "TRK-01")
LOCAL_HOST = os.getenv("LOCAL_MQTT_HOST", "127.0.0.1")
LOCAL_PORT = int(os.getenv("LOCAL_MQTT_PORT", "1883"))
UPLINK_HOST = os.getenv("UPLINK_MQTT_HOST", os.getenv("MQTT_HOST", "127.0.0.1"))
UPLINK_PORT = int(os.getenv("UPLINK_MQTT_PORT", os.getenv("MQTT_PORT", "1884")))
PUBACK_TIMEOUT = float(os.getenv("PUBACK_TIMEOUT_S", "8"))

TOPIC_BASE = f"logibridge/trucks/{TRUCK_ID}"
TOPIC_SENSORS = f"{TOPIC_BASE}/sensors/#"
TOPIC_INFERENCE = f"{TOPIC_BASE}/inference"
TOPIC_ALERTS = f"{TOPIC_BASE}/alerts"
TOPIC_HEALTH = f"{TOPIC_BASE}/health"
TOPIC_DRIFT = f"{TOPIC_BASE}/drift"


class DurableLog:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("""CREATE TABLE IF NOT EXISTS inference_records(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                ts REAL NOT NULL,
                truck_id TEXT NOT NULL,
                cls INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                probs TEXT NOT NULL,
                features TEXT NOT NULL,
                door_open INTEGER NOT NULL,
                severity TEXT,
                alert_required INTEGER NOT NULL DEFAULT 0,
                inference_synced INTEGER NOT NULL DEFAULT 0,
                alert_synced INTEGER NOT NULL DEFAULT 1
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_pending ON inference_records(inference_synced, alert_synced, id)")

    def append(self, record: dict) -> int:
        with self.lock:
            cur = self.db.execute(
                """INSERT INTO inference_records(
                    event_id,ts,truck_id,cls,label,confidence,probs,features,
                    door_open,severity,alert_required,inference_synced,alert_synced)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["event_id"],
                    record["ts"],
                    record["truck_id"],
                    record["class"],
                    record["label"],
                    record["confidence"],
                    json.dumps(record["probs"]),
                    json.dumps(record["features"]),
                    int(record["door_open"]),
                    record.get("severity"),
                    int(record["alert_required"]),
                    0,
                    0 if record["alert_required"] else 1,
                ),
            )
            return int(cur.lastrowid)

    def mark(self, row_id: int, field: str) -> None:
        if field not in {"inference_synced", "alert_synced"}:
            raise ValueError("invalid synchronization field")
        with self.lock:
            self.db.execute(f"UPDATE inference_records SET {field}=1 WHERE id=?", (row_id,))

    def pending(self, limit: int = 200):
        with self.lock:
            return self.db.execute(
                """SELECT * FROM inference_records
                   WHERE inference_synced=0 OR (alert_required=1 AND alert_synced=0)
                   ORDER BY id ASC LIMIT ?""", (limit,)
            ).fetchall()

    def pending_count(self) -> int:
        with self.lock:
            return int(self.db.execute(
                """SELECT COUNT(*) FROM inference_records
                   WHERE inference_synced=0 OR (alert_required=1 AND alert_synced=0)"""
            ).fetchone()[0])

    @staticmethod
    def record_from_row(row, replayed=True):
        return {
            "event_id": row["event_id"], "ts": row["ts"], "truck_id": row["truck_id"],
            "class": row["cls"], "label": row["label"], "confidence": row["confidence"],
            "probs": json.loads(row["probs"]), "features": json.loads(row["features"]),
            "door_open": bool(row["door_open"]), "replayed": bool(replayed),
        }


class InferenceService:
    def __init__(self):
        import paho.mqtt.client as mqtt
        self.mqtt = mqtt
        self.model = TFLiteModel(MODEL_PATH)
        self.stats = Normaliser.load(STATS_PATH)
        self.featuriser = WindowFeaturiser()
        self.log = DurableLog(ALERT_DB)
        self.stop_event = threading.Event()
        self.uplink_online = threading.Event()
        self.replay_lock = threading.Lock()
        self.publish_lock = threading.Lock()
        self.last_class = None
        self.sequence = 0
        self.psi = RollingPSI(REFERENCE_PATH) if Path(REFERENCE_PATH).exists() else None

        self.local = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"edge-local-{TRUCK_ID}")
        self.local.on_connect = self._on_local_connect
        self.local.on_message = self._on_local_message
        self.local.on_disconnect = self._on_local_disconnect

        self.uplink = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"edge-uplink-{TRUCK_ID}")
        self.uplink.on_connect = self._on_uplink_connect
        self.uplink.on_disconnect = self._on_uplink_disconnect
        self.uplink.reconnect_delay_set(min_delay=2, max_delay=60)
        username = os.getenv("UPLINK_MQTT_USERNAME")
        if username:
            self.uplink.username_pw_set(username, os.getenv("UPLINK_MQTT_PASSWORD"))
        if os.getenv("UPLINK_MQTT_TLS", "false").lower() == "true":
            self.uplink.tls_set(ca_certs=os.getenv("UPLINK_CA_CERT"))
        self.uplink.will_set(TOPIC_HEALTH, json.dumps({"truck_id": TRUCK_ID, "state": "offline"}), qos=1, retain=True)

    def _on_local_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(TOPIC_SENSORS, qos=0)
        print(f"[LOCAL] sensor broker connected at {LOCAL_HOST}:{LOCAL_PORT}", flush=True)

    def _on_local_disconnect(self, client, userdata, flags, reason_code, properties=None):
        print("[LOCAL] sensor broker disconnected; waiting for local broker recovery", flush=True)

    def _on_uplink_connect(self, client, userdata, flags, reason_code, properties=None):
        # VERSION2 callbacks may provide a ReasonCode object. Only mark the
        # uplink online after a successful MQTT CONNACK.
        if reason_code.is_failure:
            self.uplink_online.clear()
            print(f"[UPLINK] connection rejected: reason={reason_code}", flush=True)
            return

        self.uplink_online.set()
        client.publish(
            TOPIC_HEALTH,
            json.dumps({
                "truck_id": TRUCK_ID,
                "state": "online",
                "model": Path(MODEL_PATH).name,
            }),
            qos=1,
            retain=True,
        )
        print(
            f"[UPLINK] connected at {UPLINK_HOST}:{UPLINK_PORT}; "
            f"backlog={self.log.pending_count()}",
            flush=True,
        )
        threading.Thread(
            target=self.drain_backlog,
            name="backlog-replay",
            daemon=True,
        ).start()

    def _on_uplink_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.uplink_online.clear()
        print("[UPLINK] cellular path unavailable; local inference and SQLite logging continue", flush=True)

    def _on_local_message(self, client, userdata, message):
        try:
            kind = message.topic.rsplit("/", 1)[-1]
            if kind not in {"temperature", "vibration", "door"}:
                return
            payload = json.loads(message.payload)
            features = self.featuriser.push(float(payload["ts"]), kind, payload)
            if features is not None:
                self.infer(features)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"[LOCAL] rejected malformed sensor message: {exc}", flush=True)

    def _publish_local(self, topic: str, payload: dict, retain=False):
        self.local.publish(topic, json.dumps(payload), qos=1, retain=retain)

    def _publish_remote_ack(self, topic: str, payload: dict, retain=False) -> bool:
        if not self.uplink_online.is_set():
            return False
        try:
            with self.publish_lock:
                info = self.uplink.publish(topic, json.dumps(payload), qos=1, retain=retain)
                if info.rc != self.mqtt.MQTT_ERR_SUCCESS:
                    self.uplink_online.clear()
                    return False
                info.wait_for_publish(timeout=PUBACK_TIMEOUT)
                published = bool(info.is_published())
                if not published:
                    # A queued message is not proof of remote delivery. Leave
                    # the SQLite row unsynced and allow reconnect/replay.
                    self.uplink_online.clear()
                return published
        except (RuntimeError, ValueError, OSError):
            self.uplink_online.clear()
            return False

    def infer(self, features):
        probabilities = self.model.predict(self.stats.transform(features))
        predicted = int(probabilities.argmax())
        self.sequence += 1
        alert_required = predicted == 2 or (predicted > 0 and predicted != self.last_class)
        severity = "CRITICAL" if predicted == 2 else "WARNING" if alert_required else None
        record = {
            "event_id": f"{TRUCK_ID}-{time.time_ns()}-{self.sequence}-{uuid4().hex[:6]}",
            "ts": time.time(), "truck_id": TRUCK_ID, "class": predicted,
            "label": CLASS_NAMES[predicted], "confidence": float(probabilities[predicted]),
            "probs": [float(value) for value in probabilities],
            "features": [float(value) for value in features],
            "door_open": bool(self.featuriser.door_open),
            "alert_required": bool(alert_required), "severity": severity,
        }
        row_id = self.log.append(record)
        wire_record = {key: value for key, value in record.items() if key != "alert_required"}

        self._publish_local(TOPIC_INFERENCE, wire_record)
        if self._publish_remote_ack(TOPIC_INFERENCE, wire_record):
            self.log.mark(row_id, "inference_synced")
        if alert_required:
            alert = {**wire_record, "severity": severity}
            self._publish_local(TOPIC_ALERTS, alert, retain=True)
            if self._publish_remote_ack(TOPIC_ALERTS, alert, retain=True):
                self.log.mark(row_id, "alert_synced")

        self._update_psi(probabilities, predicted)
        self.last_class = predicted
        offline = "" if self.uplink_online.is_set() else " [UPLINK-OFFLINE-BUFFERED]"
        print(f"[SVC] #{self.sequence:04d} {CLASS_NAMES[predicted]:<8} conf={probabilities[predicted]:.3f} temp={features[0]:.2f}C roc={features[2]:+.2f}C/min vib={features[3]:.2f}g{offline}", flush=True)

    def _update_psi(self, probabilities, predicted):
        if self.psi is None:
            return
        result = self.psi.update(probabilities, predicted)
        if result is None:
            return
        print(f"[PSI] value={result['psi']:.3f} critical_share={result['critical_share']:.1%}", flush=True)
        if result["state_change"] == "alert":
            payload = {"truck_id": TRUCK_ID, "ts": time.time(), **result}
            print(f"[LOGIBRIDGE DRIFT ALERT] PSI={result['psi']:.3f}", flush=True)
            self._publish_local(TOPIC_DRIFT, payload, retain=True)
            self._publish_remote_ack(TOPIC_DRIFT, payload, retain=True)
        elif result["state_change"] == "clear":
            payload = {"truck_id": TRUCK_ID, "ts": time.time(), **result}
            print(f"[LOGIBRIDGE DRIFT RECOVERY] PSI={result['psi']:.3f}", flush=True)
            self._publish_local(TOPIC_DRIFT, payload, retain=True)
            self._publish_remote_ack(TOPIC_DRIFT, payload, retain=True)

    def drain_backlog(self):
        if not self.replay_lock.acquire(blocking=False):
            return
        try:
            while self.uplink_online.is_set():
                rows = self.log.pending(limit=200)
                if not rows:
                    break
                progress = False
                print(f"[UPLINK] replaying batch of {len(rows)} records", flush=True)
                for row in rows:
                    if not self.uplink_online.is_set():
                        break
                    record = self.log.record_from_row(row)
                    if not row["inference_synced"]:
                        if self._publish_remote_ack(TOPIC_INFERENCE, record):
                            self.log.mark(row["id"], "inference_synced")
                            progress = True
                        else:
                            break
                    if row["alert_required"] and not row["alert_synced"]:
                        alert = {**record, "severity": row["severity"]}
                        if self._publish_remote_ack(TOPIC_ALERTS, alert, retain=True):
                            self.log.mark(row["id"], "alert_synced")
                            progress = True
                        else:
                            break
                if not progress:
                    break
            print(f"[UPLINK] replay complete; backlog={self.log.pending_count()}", flush=True)
        finally:
            self.replay_lock.release()

    def run(self):
        print(f"[SVC] truck={TRUCK_ID}; model={MODEL_PATH}; local={LOCAL_HOST}:{LOCAL_PORT}; uplink={UPLINK_HOST}:{UPLINK_PORT}", flush=True)
        self.local.connect(LOCAL_HOST, LOCAL_PORT, keepalive=30)
        self.local.loop_start()
        self.uplink.connect_async(UPLINK_HOST, UPLINK_PORT, keepalive=30)
        self.uplink.loop_start()
        while not self.stop_event.wait(1.0):
            pass
        self.local.loop_stop()
        self.uplink.loop_stop()
        self.local.disconnect()
        self.uplink.disconnect()

    def stop(self):
        self.stop_event.set()


def main() -> int:
    service = InferenceService()
    signal.signal(signal.SIGTERM, lambda *_: service.stop())
    signal.signal(signal.SIGINT, lambda *_: service.stop())
    service.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

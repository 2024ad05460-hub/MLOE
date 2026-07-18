import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from inference_service import DurableLog


def record(alert=True):
    return {
        "event_id": "TRK-TEST-1", "ts": 1.0, "truck_id": "TRK-TEST",
        "class": 2 if alert else 0, "label": "Critical" if alert else "Normal",
        "confidence": 0.98, "probs": [0.01, 0.01, 0.98],
        "features": [1, 2, 3, 4, 5, 6], "door_open": False,
        "severity": "CRITICAL" if alert else None, "alert_required": alert,
    }


def test_complete_probabilities_and_alert_state_are_persisted(tmp_path):
    log = DurableLog(str(tmp_path / "alerts.db"))
    row_id = log.append(record(alert=True))
    row = log.pending()[0]
    restored = log.record_from_row(row)
    assert restored["probs"] == [0.01, 0.01, 0.98]
    assert row["inference_synced"] == 0
    assert row["alert_synced"] == 0
    log.mark(row_id, "inference_synced")
    assert log.pending_count() == 1
    log.mark(row_id, "alert_synced")
    assert log.pending_count() == 0

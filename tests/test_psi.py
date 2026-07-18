import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "monitoring"))
from psi import RollingPSI


def test_psi_alert_and_recovery(tmp_path):
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps({"distribution": [0.005, 0.005, 0.01, 0.98]}))
    monitor = RollingPSI(ref, rolling_n=10, interval_s=0, alert=0.25, clear=0.10)
    for i in range(10):
        result = monitor.update([0.99, 0.005, 0.005], 0, now=i)
    assert result["psi"] < 0.10
    for i in range(10, 20):
        result = monitor.update([0.01, 0.01, 0.98], 2, now=i)
    assert result["psi"] > 0.25
    assert result["alert_active"]
    for i in range(20, 35):
        result = monitor.update([0.99, 0.005, 0.005], 0, now=i)
    assert result["psi"] < 0.10
    assert not result["alert_active"]

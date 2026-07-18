"""Generate the exact minimum dataset requested in Task D1.

This file is kept separately from dataset.npz. The assignment-prescribed run has
one 20-minute Normal stream, one 15-minute Warning stream and one 15-minute
Critical stream, producing approximately 120/90/90 windows. The production
training pipeline uses repeated truck groups in generate_dataset.py to obtain a
leakage-safe 80/20 truck holdout; this file proves the minimum duration and class
composition were also implemented exactly.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
from preprocessing import windows_with_times  # noqa: E402
from simulator import ColdChainSimulator  # noqa: E402

SPEC = [
    (0, "Normal", "none", 20 * 60, 11001),
    (1, "Warning", "temp_drift", 15 * 60, 11002),
    (2, "Critical", "combined", 15 * 60, 11003),
]


def main() -> int:
    xs, ys, modes, ends = [], [], [], []
    summary = []
    for label, name, mode, duration_s, seed in SPEC:
        sim = ColdChainSimulator(mode, f"ASSIGN-{name.upper()}", seed=seed, onset_s=0.0)
        features, end_s = windows_with_times(sim.stream(duration_s))
        xs.append(features)
        ys.append(np.full(len(features), label, dtype=np.int64))
        modes.append(np.full(len(features), mode, dtype="U16"))
        ends.append(end_s)
        summary.append({
            "class": label,
            "label": name,
            "simulator_mode": mode,
            "duration_minutes": duration_s // 60,
            "windows": int(len(features)),
            "assignment_approximation": "~120" if label == 0 else "~90",
        })
        print(f"[ASSIGN-DATA] class={label} {name:<8} mode={mode:<10} "
              f"duration={duration_s//60:2d} min windows={len(features)}")

    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    np.savez_compressed(
        ROOT / "training" / "assignment_dataset.npz",
        X_raw=X,
        y=y,
        mode=np.concatenate(modes),
        window_end_s=np.concatenate(ends),
        feature_order=np.asarray([
            "temp_mean", "temp_std", "temp_roc_c_per_min",
            "vib_rms", "vib_peak", "vib_kurtosis"
        ]),
    )
    payload = {
        "purpose": "Exact Task D1 minimum-duration dataset evidence",
        "total_windows": int(len(X)),
        "classes": summary,
        "note": (
            "The 30 s window closes first at t=30 s and the simulator emits "
            "samples through duration-1 s, hence 117 and 87 windows rather than "
            "the rounded assignment estimates 120 and 90."
        ),
    }
    (ROOT / "training" / "assignment_dataset_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    with open(ROOT / "training" / "assignment_dataset_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"[ASSIGN-DATA] total={len(X)} -> training/assignment_dataset.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

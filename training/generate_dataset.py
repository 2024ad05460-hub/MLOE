"""Generate labelled LogiEdge windows with leakage-safe grouped validation.

Overlapping 30-second windows advance by 10 seconds, so a random window split
puts nearly identical samples into train and validation. This version assigns
complete simulated trucks to one partition. With the default 10 trucks, two
whole trucks form the required 20% validation holdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
from preprocessing import Normaliser, windows_from_stream, windows_with_times  # noqa: E402
from simulator import ColdChainSimulator  # noqa: E402

SPEC = (("none", 0, 20 * 60), ("temp_drift", 1, 15 * 60), ("combined", 2, 15 * 60))
CLEAN_STATS_SECONDS = 10 * 60
N_COMMISSION = 6


def build(reps: int = 10, seed0: int = 1000) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {
        "X_raw": [], "y": [], "group_id": [], "truck_id": [], "mode": [],
        "run_id": [], "window_end_s": [], "fault_onset_s": []
    }
    for mode, run_label, duration_s in SPEC:
        for rep in range(reps):
            truck = f"TRK-SIM-{rep:02d}"
            run_id = f"{truck}:{mode}"
            sim = ColdChainSimulator(mode, truck, seed=seed0 + 1000 * run_label + rep)
            windows, end_s = windows_with_times(sim.stream(duration_s))
            labels = np.full(len(windows), run_label, dtype=np.int64)
            if run_label:
                labels[end_s < sim.onset_s] = 0
            out["X_raw"].append(windows)
            out["y"].append(labels)
            out["group_id"].append(np.full(len(windows), rep, dtype=np.int64))
            out["truck_id"].append(np.full(len(windows), truck, dtype="U24"))
            out["mode"].append(np.full(len(windows), mode, dtype="U16"))
            out["run_id"].append(np.full(len(windows), run_id, dtype="U48"))
            out["window_end_s"].append(end_s.astype(np.float64))
            out["fault_onset_s"].append(np.full(len(windows), sim.onset_s, dtype=np.float64))
            relabelled = int((labels != run_label).sum()) if run_label else 0
            print(f"  {run_id:<30} {len(windows):3d} windows; onset={sim.onset_s:6.1f}s; pre-onset Normal={relabelled}")
    return {name: np.concatenate(parts) for name, parts in out.items()}


def fit_stats(seed0: int = 7000) -> None:
    windows = []
    for idx in range(N_COMMISSION):
        sim = ColdChainSimulator("none", f"TRK-COMMISSION-{idx:02d}", seed=seed0 + idx)
        windows.append(windows_from_stream(sim.stream(CLEAN_STATS_SECONDS)))
    clean = np.concatenate(windows)
    stats = Normaliser.fit(clean)
    stats.save(ROOT / "data_pipeline" / "training_stats.npy")
    print(f"[STATS] fitted on {len(clean)} clean windows from {N_COMMISSION} separate commissioning trucks")


def grouped_split(group_id: np.ndarray, validation_fraction: float, seed: int):
    groups = np.unique(group_id)
    if len(groups) < 2:
        raise ValueError("At least two simulated trucks are required")
    shuffled = np.random.default_rng(seed).permutation(groups)
    n_val = max(1, min(len(groups) - 1, int(round(len(groups) * validation_fraction))))
    val_groups = np.sort(shuffled[:n_val])
    val_mask = np.isin(group_id, val_groups)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask), val_groups


def counts(y: np.ndarray) -> dict[str, int]:
    values, number = np.unique(y, return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, number)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10, help="simulated trucks; default gives exact 80/20 grouped split")
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_split < 1:
        parser.error("--val-split must be between 0 and 1")

    data = build(args.reps)
    fit_stats()
    train_idx, val_idx, val_groups = grouped_split(data["group_id"], args.val_split, args.seed)
    train_groups = np.unique(data["group_id"][train_idx])
    overlap = np.intersect1d(train_groups, val_groups)
    if overlap.size:
        raise RuntimeError(f"group leakage: {overlap.tolist()}")
    for split_name, split_idx in (("train", train_idx), ("validation", val_idx)):
        if set(np.unique(data["y"][split_idx]).tolist()) != {0, 1, 2}:
            raise RuntimeError(f"{split_name} split does not contain all three classes")

    np.savez_compressed(
        ROOT / "training" / "dataset.npz", **data,
        train_idx=train_idx.astype(np.int64), val_idx=val_idx.astype(np.int64),
        split_method=np.asarray("grouped_by_simulated_truck"),
        split_seed=np.asarray(args.seed), validation_fraction=np.asarray(args.val_split)
    )
    manifest = {
        "split_method": "grouped_by_simulated_truck",
        "overlapping_groups": [],
        "train_groups": train_groups.tolist(),
        "validation_groups": val_groups.tolist(),
        "train_windows": int(len(train_idx)),
        "validation_windows": int(len(val_idx)),
        "train_class_counts": counts(data["y"][train_idx]),
        "validation_class_counts": counts(data["y"][val_idx]),
    }
    (ROOT / "training" / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DATA] {len(train_idx)} train / {len(val_idx)} validation windows; group overlap=0")
    print(f"[DATA] validation trucks={val_groups.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

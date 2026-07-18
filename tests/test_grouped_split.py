from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_grouped_split_has_no_truck_leakage():
    data = np.load(ROOT / "training" / "dataset.npz")
    train_groups = set(data["group_id"][data["train_idx"]].tolist())
    validation_groups = set(data["group_id"][data["val_idx"]].tolist())
    assert train_groups.isdisjoint(validation_groups)
    assert set(data["y"][data["train_idx"]].tolist()) == {0, 1, 2}
    assert set(data["y"][data["val_idx"]].tolist()) == {0, 1, 2}


def test_validation_fraction_is_approximately_twenty_percent():
    data = np.load(ROOT / "training" / "dataset.npz")
    actual = len(data["val_idx"]) / len(data["y"])
    assert 0.18 <= actual <= 0.22

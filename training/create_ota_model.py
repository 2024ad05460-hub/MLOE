"""Create a valid replacement TFLite model for the OTA layer-cache test."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_MODEL = (
    ROOT
    / "training"
    / "models"
    / "m3_pruned_int8.tflite"
)

OTA_MODEL = (
    ROOT
    / "training"
    / "models"
    / "m3_pruned_int8_ota.tflite"
)


def main() -> None:
    """Create the OTA replacement model artifact."""

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(
            f"Source M3 model was not found: {SOURCE_MODEL}"
        )

    OTA_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        SOURCE_MODEL,
        OTA_MODEL,
    )

    if not OTA_MODEL.is_file():
        raise RuntimeError(
            f"OTA model was not created: {OTA_MODEL}"
        )

    if OTA_MODEL.stat().st_size == 0:
        raise RuntimeError(
            f"OTA model is empty: {OTA_MODEL}"
        )

    print(
        f"Created OTA replacement model: {OTA_MODEL}"
    )

    print(
        f"OTA model size: {OTA_MODEL.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
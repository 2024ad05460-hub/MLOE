from pathlib import Path

TRUCKS = 85
COST_PER_MB_INR = 0.10
ASSIGNMENT_MODEL_KB = 280.0

root = Path(__file__).resolve().parents[1]
model_path = root / "training" / "models" / "m3_pruned_int8.tflite"

actual_model_kb = model_path.stat().st_size / 1024.0

assignment_model_total_mb = (
    ASSIGNMENT_MODEL_KB * TRUCKS
) / 1024.0

assignment_model_cost = (
    assignment_model_total_mb *
    COST_PER_MB_INR
)

actual_model_total_mb = (
    actual_model_kb * TRUCKS
) / 1024.0

actual_model_cost = (
    actual_model_total_mb *
    COST_PER_MB_INR
)

print(f"Pilot trucks: {TRUCKS}")
print(
    f"Assignment model size: "
    f"{ASSIGNMENT_MODEL_KB:.2f} KB"
)
print(
    f"Assignment total transfer: "
    f"{assignment_model_total_mb:.2f} MB"
)
print(
    f"Assignment model-only cost: "
    f"INR {assignment_model_cost:.2f}"
)
print(
    f"Actual model size: "
    f"{actual_model_kb:.2f} KB"
)
print(
    f"Actual model transfer: "
    f"{actual_model_total_mb:.2f} MB"
)
print(
    f"Actual model-only cost: "
    f"INR {actual_model_cost:.2f}"
)
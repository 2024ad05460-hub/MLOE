#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11 was not found. Install Python 3.11 and rerun, or set PYTHON_BIN=python3.12." >&2
  exit 1
fi
"$PYTHON_BIN" -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
# TFMOT 0.8.0 declares NumPy ~=1.23 although the tested LogiEdge environment
# uses NumPy 1.26.4 required by TensorFlow 2.20. Install it without dependency
# resolution; its actual runtime dependencies are already pinned above.
python -m pip install --no-deps tensorflow-model-optimization==0.8.0
python - <<'PY'
import tensorflow as tf, keras, tf_keras
import tensorflow_model_optimization as tfmot
print("TensorFlow", tf.__version__)
print("Keras", keras.__version__)
print("tf_keras", tf_keras.__version__)
print("TFMOT", tfmot.__version__)
PY
echo "Environment ready. Activate with: source .venv311/bin/activate"

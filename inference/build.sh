#!/usr/bin/env bash
# Usage: ./build.sh [MODEL_TFLITE] [TAG]
set -euo pipefail
MODEL="${1:-../training/models/m3_pruned_int8.tflite}"
TAG="${2:-logibridge/inference:v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

cp "$HERE/../data_pipeline/preprocessing.py"   "$HERE/preprocessing.py"
cp "$HERE/../optimisation/tflite_eval.py"      "$HERE/tflite_eval.py"
cp "$HERE/../monitoring/psi.py"                "$HERE/psi.py"
cp "$HERE/../monitoring/reference_dist.json"   "$HERE/reference_dist.json"
cp "$HERE/../data_pipeline/training_stats.npy" "$HERE/training_stats.npy"
cp "$MODEL"                                    "$HERE/model.tflite"

docker build -t "$TAG" "$HERE"
echo "built $TAG from $(basename "$MODEL")"

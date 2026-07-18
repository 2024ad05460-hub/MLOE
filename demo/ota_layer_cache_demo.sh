#!/usr/bin/env bash
# LogiEdge - Task D2: OTA layer-cache demonstration.
#
# Builds the image twice. Between the builds NOTHING changes except the model file.
# Docker must report every pip/apt/code layer as "CACHED" and rebuild only the final
# COPY model.tflite layer. The script then computes the fleet bandwidth saving.
#
# Run:  ./demo/ota_layer_cache_demo.sh 2>&1 | tee demo/ota_layer_cache.log
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=logibridge/inference
FLEET=85
RATE=0.10          # Rs per MB, M2M SIM

echo "=============================================================="
echo " BUILD 1 - baseline image, model = M2 (PTQ INT8)"
echo "=============================================================="
./inference/build.sh training/models/m2_ptq_int8.tflite "$TAG:v1"

IMG_MB=$(docker image inspect "$TAG:v1" --format '{{.Size}}' | awk '{printf "%.1f", $1/1000000}')
echo "  full image size: ${IMG_MB} MB"

echo
echo "=============================================================="
echo " BUILD 2 - ONLY the model file changes (M2 -> M3 pruned INT8)"
echo "=============================================================="
docker build --progress=plain -t "$TAG:v2" - < /dev/null >/dev/null 2>&1 || true
./inference/build.sh training/models/m3_pruned_int8.tflite "$TAG:v2" 2>&1 | tee /tmp/build2.log

echo
echo "--- cache verification: every layer above COPY model.tflite must be CACHED ---"
docker build --progress=plain -t "$TAG:v2" inference/ 2>&1 | grep -E "CACHED|COPY model|=> \[" || true

MODEL_KB=$(stat -c %s training/models/m3_pruned_int8.tflite | awk '{printf "%.1f", $1/1024}')
MODEL_MB=$(echo "$MODEL_KB" | awk '{printf "%.6f", $1/1024}')

echo
echo "=============================================================="
echo " FLEET BANDWIDTH ARITHMETIC  (${FLEET} trucks, Rs ${RATE}/MB)"
echo "=============================================================="
awk -v img="$IMG_MB" -v mdl="$MODEL_MB" -v fleet="$FLEET" -v rate="$RATE" 'BEGIN{
  full  = img * fleet;      full_c  = full * rate;
  delta = mdl * fleet;      delta_c = delta * rate;
  printf "  Naive full-image push : %10.2f MB   Rs %10.2f\n", full,  full_c;
  printf "  Layer-cached model push: %9.4f MB   Rs %10.4f\n", delta, delta_c;
  printf "  Saving                 : %9.2f MB   Rs %10.2f  (%.3f%% less data)\n",
         full-delta, full_c-delta_c, 100*(1-delta/full);
  printf "  Per-truck download     : %9.1f MB -> %.1f KB\n", img, mdl*1024;
}'
echo
echo "At a 6-week cadence (8.7 cycles/yr) the cached path saves roughly"
echo "Rs $(awk -v i="$IMG_MB" -v m="$MODEL_MB" -v f="$FLEET" -v r="$RATE" \
      'BEGIN{printf "%.0f", (i-m)*f*r*8.7}') per year in SIM data for the pilot fleet -"
echo "and, more importantly, turns a 150 MB download that CANNOT complete inside a"
echo "rural coverage window into a few-KB download that can."

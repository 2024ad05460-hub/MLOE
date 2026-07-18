# LogiEdge Setup and Execution Guide

## 1. Supported development environment

Use Python **3.11 or 3.12**, Git, Docker Desktop/Engine, Mosquitto and Ansible.
The training setup scripts create `.venv`, install TensorFlow/Keras and install
`tensorflow-model_optimization==0.8.0` using the compatibility path required by
its stale dependency metadata.

### Linux/macOS

```bash
bash setup_training_env.sh
source .venv311/bin/activate
```

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_training_env.ps1
.\.venv311\Scripts\Activate.ps1
```

Manual fallback:

```bash
python -m venv .venv
source .venv311/bin/activate                 # Windows: .venv311\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install --no-deps tensorflow-model-optimization==0.8.0
```

## 2. Run the complete reproducible pipeline

Linux/macOS:

```bash
bash run_all.sh
```

Windows:

```powershell
.\run_all.ps1
```

The pipeline performs:

1. exact Task D1 20/15/15-minute dataset generation;
2. larger grouped dataset generation and overlap audit;
3. M1 training, M2 full-INT8 PTQ and M3 pruning/structural reduction/PTQ;
4. 3σ normalisation sensitivity experiment;
5. five-metric benchmark and Pareto chart;
6. PSI reference, fault injection and recovery;
7. automated tests, diagrams, figures and constraint calculations.

Do not replace the grouped split with a random window split. Adjacent windows
share 20 seconds of data and would otherwise leak across train and validation.

## 3. Start the local and uplink demonstration brokers

```bash
docker compose -f demo/docker-compose.yml up -d
```

- Local sensor broker: `127.0.0.1:1883`
- Simulated operations-centre uplink broker: `127.0.0.1:1884`

## 4. Build the inference image

```bash
cd inference
./build.sh ../training/models/m3_pruned_int8.tflite logibridge/inference:v2
cd ..
mkdir -p /tmp/logibridge
cp training/models/m3_pruned_int8.tflite /tmp/logibridge/model.tflite
cp data_pipeline/training_stats.npy /tmp/logibridge/training_stats.npy
cp monitoring/reference_dist.json /tmp/logibridge/reference_dist.json
```

Run it on Linux with host networking:

```bash
docker run --rm --name logibridge-inference --network host \
  -e TRUCK_ID=TRK-DEMO \
  -e MODEL_PATH=/data/model.tflite \
  -e STATS_PATH=/data/training_stats.npy \
  -e REFERENCE_PATH=/data/reference_dist.json \
  -e ALERT_DB=/data/alerts.db \
  -e LOCAL_MQTT_HOST=127.0.0.1 -e LOCAL_MQTT_PORT=1883 \
  -e UPLINK_MQTT_HOST=127.0.0.1 -e UPLINK_MQTT_PORT=1884 \
  -v /tmp/logibridge:/data \
  logibridge/inference:v2
```

On Docker Desktop, use the broker service names from `demo/docker-compose.yml`
or `host.docker.internal` rather than Linux host networking.

## 5. Run the simulator

```bash
python data_pipeline/simulator.py --anomaly none --duration 180 --speed 20
python data_pipeline/simulator.py --anomaly temp_drift --duration 180 --speed 20
python data_pipeline/simulator.py --anomaly combined --duration 600 --speed 20
```

## 6. Correct cellular-outage test

Keep the local broker and inference container running. Stop only the remote
uplink broker:

```bash
docker compose -f demo/docker-compose.yml stop uplink-broker
# Observe local inference and [UPLINK-OFFLINE-BUFFERED].
docker compose -f demo/docker-compose.yml start uplink-broker
# Observe oldest-first QoS 1 replay until pending rows reach zero.
```

Inspect the outbox:

```bash
sqlite3 /tmp/logibridge/alerts.db \
  "SELECT COUNT(*) AS pending FROM inference_records WHERE inference_synced=0 OR (alert_required=1 AND alert_synced=0);"
```

## 7. OTA Docker layer-cache demonstration

Perform an initial build, replace only the model file with another TFLite
variant, and rebuild with the same tag or a new tag. Capture the BuildKit output
showing requirements/application layers as `CACHED` and the model-copy layer as
rebuilt. Use the measured image and model layer sizes in the video calculation.

```bash
cd inference
./build.sh ../training/models/m2_int8.tflite logibridge/inference:ota-a
./build.sh ../training/models/m3_pruned_int8.tflite logibridge/inference:ota-b
```

## 8. Ansible localhost demonstration

```bash
ansible-galaxy collection install community.docker
cd deployment
ansible-playbook -i inventory.ini logibridge_deploy.yml \
  --limit localhost_demo \
  -e local_registry=localhost:5000 \
  -e uplink_mqtt_host=127.0.0.1 \
  -e uplink_mqtt_port=1884 \
  -e uplink_mqtt_tls=false
```

Run the identical command a second time without changing the model, reference
file, variables or image digest. The second recap must show `changed=0`.

## 9. Final submission details

- Enter the group number and all contributor names/BITS IDs in the DOCX.
- Add the private repository link and tested video link.
- Export the updated DOCX to PDF and use the required LMS filename.
- Complete every item in `FINAL_EXECUTION_CHECKLIST.md`.

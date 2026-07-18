# LogiEdge Assignment Compliance Matrix

| Task | Requirement | Evidence | Status |
|---|---|---|---|
| A1 | 350-500 word analysis of latency, bandwidth, connectivity and privacy | `reports/LogiEdge_Complete_Final_Report.pdf`, Section A1; `experiments/constraint_numbers.txt` | PASS |
| A2 | Complete architecture diagram | `scenario_architecture/system_architecture.png` | PASS |
| B1 | Constraint Triangle for three devices, 10 W limit and fleet costs | Report B1; `hardware/hardware_justification.md`; `reports/figures/hardware_cost_power.png` | PASS |
| B2 | Arithmetic intensity, ridge point and Roofline decision | Report B2; `experiments/constraint_numbers.txt` | PASS |
| C1 | CLI simulator for four modes and three MQTT streams | `data_pipeline/simulator.py` | PASS |
| C2 | 5-tap filter, 30 s/10 s windows, six features, frozen stats and 3-sigma experiment | `data_pipeline/preprocessing.py`; `data_pipeline/training_stats.npy`; `experiments/normalisation_experiment.csv` | PASS |
| C3 | Feature-level fusion justification | Report C3; `data_pipeline/preprocessing.py` | PASS |
| D1 | Prescribed minimum-duration dataset and >88% validation accuracy | `training/assignment_dataset.npz`; grouped `training/dataset.npz`; M1 accuracy 99.48% | PASS |
| D2 | Python 3.11 Docker image, MODEL_PATH, exact MQTT topic and layer-cache design | `inference/Dockerfile`; `inference/inference_service.py`; `demo/ota_layer_cache_demo.sh` | CODE PASS; live build video required |
| D3 | LogiEdge mapping to all ten Edge ML pipeline stages | Report D3 | PASS |
| E1 | PSI reference of 300 windows, rolling 100, 60 s, threshold/recovery demo | `monitoring/reference_dist.json`; `monitoring/psi_trace_normal_prob.json`; PSI 2.614 and recovery 0.037 | PASS |
| E2 | Exactly seven Ansible tasks and second run changed=0 | `deployment/logibridge_deploy.yml`; static test | CODE PASS; live terminal video required |
| E3 | Full, canary and shadow calculations and recommendation | Report E3; `experiments/constraint_numbers.txt` | PASS |
| F1 | M1 FP32, M2 full INT8, M3 35% prune + full INT8 | `training/models/`; training scripts and metrics | PASS |
| F2 | Mean/p95, size, accuracy, energy for 200 runs after ten warm-ups | `optimisation/results/benchmark_results.csv`; Pareto chart | PASS |
| F3 | Recommendation with SLA, memory and Critical recall >95% | Report F3; M3 Critical recall 99.4% | PASS |

## Submission-owned items

The following items require the student's own account or target machine and cannot be truthfully fabricated in a generated package:

1. Group number, complete member names and BITS IDs on the cover page.
2. Private GitHub repository URL and instructor collaborator access.
3. Fifteen-to-twenty-minute demo video URL showing the Docker cache and both Ansible runs.
4. Optional Raspberry Pi 5 target-device measurements if the evaluator requires physical-hardware evidence rather than development-host benchmarking.

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
step() { echo; echo "=============================================================="; echo " $1"; echo "=============================================================="; }

step "1/10 Exact Task D1 assignment-duration dataset"
python training/generate_assignment_dataset.py
step "2/10 Leakage-safe grouped dataset"
python training/generate_dataset.py
step "3/10 Train M1 FP32"
python training/train_model.py
step "4/10 Create M2 full INT8 PTQ"
python training/convert_ptq.py
step "5/10 Create M3 PolynomialDecay pruning + structural removal + INT8"
python training/prune_quantise.py
step "6/10 Normalisation experiment"
python experiments/normalisation_experiment.py
step "7/10 Five-metric benchmark and Pareto chart"
python optimisation/benchmark.py
step "8/10 PSI reference, injection and recovery"
python monitoring/drift_monitor.py --mode reference --score normal_prob
python monitoring/drift_monitor.py --mode simulate --score normal_prob
step "9/10 Automated regression tests"
pytest -q
step "10/10 Diagrams, evidence figures and constraint calculations"
python scenario_architecture/make_diagrams.py
python reports/build_evidence_figures.py
python experiments/constraint_numbers.py | tee experiments/constraint_numbers.txt

echo "DONE - review VERIFICATION.md and FINAL_EXECUTION_CHECKLIST.md before recording the live demo."

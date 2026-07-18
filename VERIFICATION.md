# LogiEdge Package Verification

Verified on **16 July 2026** in the available package-build environment.

## Executed successfully

- Python source compilation: **PASS**
- Automated regression tests: **8 passed**
- Exact assignment-duration data: **117 Normal + 87 Warning + 87 Critical = 291 windows**
- Grouped data: **2,328 training + 582 validation windows**
- Group split audit: **zero truck groups shared between training and validation**
- M1 validation accuracy: **99.48%**
- M2 validation accuracy: **99.31%**
- M3 validation accuracy: **99.14%**
- M3 Critical recall: **99.4%**
- M3 model size: **3.70 KB**
- Latest M3 x86 mean/p95 latency: **1.42/1.46 µs**
- Latest M3 comparative energy estimate: **0.01900 mJ/inference**
- PSI threshold crossing: **3.0 minutes** after combined-fault injection
- PSI recovery: **0.037**, below the required 0.10
- Final report: **16 pages**, rendered and visually inspected as DOCX and PDF

The complete terminal record is in
`evidence/verification_run_2026-07-16.log`; machine-readable values are in
`evidence/verified_results.json` and `optimisation/results/benchmark_results.json`.

## Environment-dependent evidence still required

The build environment did not provide a Docker daemon, local registry,
Mosquitto service, Ansible runtime or a physical Raspberry Pi 5. Therefore the
following must be run and screen-recorded on the student's deployment machine:

1. live local/remote MQTT outage and replay;
2. Docker model-layer cache demonstration;
3. two identical Ansible executions with second-run `changed=0`;
4. target-hardware benchmark, if Raspberry Pi measurements are required;
5. final private-repository and video-link access check.

No terminal output or demo video has been fabricated. The exact commands and
expected evidence are supplied in `demo/demo_script.md`, `SETUP.md` and
`FINAL_EXECUTION_CHECKLIST.md`.

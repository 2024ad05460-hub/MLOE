from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = ROOT / "deployment" / "logibridge_deploy.yml"


with PLAYBOOK.open("r", encoding="utf-8") as handle:
    documents = yaml.safe_load(handle)

if not isinstance(documents, list) or len(documents) != 1:
    raise RuntimeError("Expected exactly one play.")

tasks = documents[0].get("tasks", [])

if len(tasks) != 7:
    raise RuntimeError(
        f"Expected exactly seven tasks; found {len(tasks)}."
    )

text = PLAYBOOK.read_text(encoding="utf-8")

required_terms = [
    "/opt/logibridge",
    "model.tflite",
    "reference_dist.json",
    "docker_container",
    "docker_image",
    "sleep 15",
]

for term in required_terms:
    if term not in text:
        raise RuntimeError(
            f"Required playbook content missing: {term}"
        )

print("Ansible playbook contract: PASS")
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]


def test_docker_application_is_not_hidden_by_data_mount():
    dockerfile = (ROOT / "inference" / "Dockerfile").read_text()
    playbook = (ROOT / "deployment" / "logibridge_deploy.yml").read_text()
    assert "WORKDIR /app" in dockerfile
    assert '"{{ data_dir }}:/data"' in playbook
    assert "{{ logibridge_dir }}:{{ logibridge_dir }}" not in playbook


def test_ansible_has_exactly_seven_top_level_tasks_and_copies_stats():
    play = yaml.safe_load((ROOT / "deployment" / "logibridge_deploy.yml").read_text())[0]
    assert len(play["tasks"]) == 7
    text = str(play)
    assert "training_stats.npy" in text
    assert "REFERENCE_PATH" in text

# Ansible Deployment

The playbook contains exactly seven top-level tasks. Application code stays inside
`/app` in the image; `/opt/logibridge/data` is mounted at `/data` for model,
normalisation stats, PSI reference and SQLite only.

```bash
ansible-galaxy collection install community.docker
cd deployment

ansible-playbook -i inventory.ini logibridge_deploy.yml --limit canary
ansible-playbook -i inventory.ini logibridge_deploy.yml --limit canary
```

The identical second run must show `changed=0`. For a local demonstration use
`--limit localhost_demo` and the variables documented in `../SETUP.md`.

The previous model is retained by Ansible's `backup: true`. Rollback means restoring
that file as `/opt/logibridge/data/model.tflite` and rerunning the playbook/container,
not downloading the full image again.

A complete deployment node map is available in [deployment/node_map.md](node_map.md).

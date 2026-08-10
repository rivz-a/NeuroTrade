#!/usr/bin/env bash
set -euo pipefail
cd /opt/neurotrade

echo "== fetching origin/main =="
git fetch origin main
git reset --hard origin/main

echo "== syncing dependencies =="
venv/bin/pip install --quiet -r requirements.txt

echo "== running test suite (deploy aborts if this fails) =="
venv/bin/python -m pytest -q

echo "== restarting services =="
sudo systemctl restart neurotrade-dashboard.service
sudo systemctl restart neurotrade-runtime.service

echo "== deploy complete: $(git rev-parse --short HEAD) =="

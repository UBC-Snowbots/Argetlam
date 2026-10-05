#!/bin/bash
# Regression gate: run after ANY change to the scene, home pose, gripper, skills, IK or config.
# 1) unit + integration tests, 2) live grasp check through the running app + Isaac (same code path as voice commands).
# Needs the app (udp 47100) and the Isaac bridge running. Exit 0 = everything passed.
set -e
cd "$(dirname "$0")/.."
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_grasps.py --moves --assist

#!/bin/bash
set -e
echo "Install OTVision."

WORKING_DIR=$(pwd)
VENV="$WORKING_DIR"/venv
PYTHON="$VENV"/bin/python
PIP="$VENV"/bin/pip

python3.11 -m venv "$VENV"

$PYTHON -m pip install --upgrade pip
$PIP install -r requirements.txt --no-cache-dir

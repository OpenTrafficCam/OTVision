#!/bin/bash
set -e
echo "Install OTVision development environment."

WORKING_DIR=$(pwd)
VENV="$WORKING_DIR"/.venv
PRE_COMMIT="$VENV"/bin/pre-commit
UV="$VENV"/bin/uv

bash "$WORKING_DIR"/install.sh

$UV pip install -r requirements-dev.txt --python .venv
$PRE_COMMIT install --install-hooks

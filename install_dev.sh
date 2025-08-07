#!/bin/bash
set -e
echo "Install OTVision development environment."

WORKING_DIR=$(pwd)
VENV="$WORKING_DIR"/.venv
PRE_COMMIT="$VENV"/bin/pre-commit

bash "$WORKING_DIR"/install.sh

uv sync --extra dev
$PRE_COMMIT install --install-hooks

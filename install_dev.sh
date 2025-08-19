#!/bin/bash
set -e
echo "Install OTVision development environment."

WORKING_DIR=$(pwd)

bash "$WORKING_DIR"/install.sh

uv sync --extra inference_cpu
uv run pre-commit install --install-hooks

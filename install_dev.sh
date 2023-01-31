#!/bin/bash
set -e

WORKING_DIR=$(pwd)
VENV="$WORKING_DIR"/venv
PYTHON="$VENV"/bin/python
PIP="$VENV"/bin/pip
PRE_COMMIT="$VENV"/bin/pre-commit

bash "$WORKING_DIR"/install.sh

$PIP install -r requirements_dev.txt --no-cache-dir
$PRE_COMMIT install
source "$VENV"/bin/activate

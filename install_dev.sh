#!/bin/bash

OS=$(uname -s)
WORKING_DIR=$(pwd)
PYTHON_VERSION=$(python3.9 --version | grep -Eo '[0-9]\.[0-9]+')
VENV="$WORKING_DIR"/venv
PYTHON="$VENV"/bin/python
PIP="$VENV"/bin/pip
PRE_COMMIT="$VENV"/bin/pre-commit

if [ "$PYTHON_VERSION" = 3.9 ]
then
    python3.9 -m venv "$VENV"
else
    echo "Cannot find python3.9. Using '$(python3 --version)' to create venv."
    python3 -m venv "$VENV"
fi

if [ "$OS" = Linux ]
then
    sudo apt-get install python3-tk
fi
if [ $OS = Darwin ]
then
    if [ $(uname -m) = arm64 ]
    then
        brew install python-tk@"$PYTHON_VERSION"
        brew install gdal
    fi
fi

$PYTHON -m pip install --upgrade pip
$PIP install -r requirements.txt --no-cache-dir
$PIP install -r requirements_dev.txt --no-cache-dir
echo $PRE_COMMIT
$PRE_COMMIT install
source "$VENV"/bin/activate

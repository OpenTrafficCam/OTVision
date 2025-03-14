#!/usr/bin/env bash

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)

set -e
echo "Install OTVision."

echo "$DIR"
cd "$DIR" || exit
WORKING_DIR=$(pwd)
VENV="$WORKING_DIR"/.venv
PYTHON="$VENV"/bin/python
PIP="$VENV"/bin/pip
UV="$VENV"/bin/uv

python3.12 -m venv "$VENV"

$PYTHON -m pip install --upgrade pip
$PIP install uv
$UV pip install -r requirements.txt --index-strategy unsafe-best-match --python .venv

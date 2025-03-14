echo Install OTVision.
@echo off
FOR /F "tokens=* USEBACKQ" %%F IN (`python --version`) DO SET PYTHON_VERSION=%%F

echo %PYTHON_VERSION%
if "x%PYTHON_VERSION:3.12=%"=="x%PYTHON_VERSION%" (
    echo "Python Version 3.12 is not installed in environment." & cmd /K & exit
)

python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install uv
uv pip install --upgrade pip-tools pip wheel
uv pip install -r requirements.txt --index-strategy unsafe-best-match --python .venv%
deactivate

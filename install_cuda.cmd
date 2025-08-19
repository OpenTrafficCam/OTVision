echo Install OTVision.
@echo off
FOR /F "tokens=* USEBACKQ" %%F IN (`python --version`) DO SET "PYTHON_VERSION=%%F"

echo %PYTHON_VERSION%
if "x%PYTHON_VERSION:3.12=%"=="x%PYTHON_VERSION%" (
    echo "Python Version 3.12 is not installed in environment." & cmd /K & exit
)

REM Check if uv is available globally, if not install it
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo uv not found globally, installing uv...
    powershell -ExecutionPolicy Bypass -Command "& {irm https://astral.sh/uv/install.ps1 | iex}"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

uv sync --extra inference_cuda --no-dev

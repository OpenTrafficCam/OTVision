echo Install OTVision development environment.
call install.cmd

uv sync --extra dev --python .venv%
pre-commit install --install-hooks

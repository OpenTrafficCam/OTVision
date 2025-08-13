echo Install OTVision development environment.
call install.cmd

uv sync --only-dev --python .venv%
pre-commit install --install-hooks

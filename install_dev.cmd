echo Install OTVision development environment.
call install.cmd

uv sync --inexact --dev --python .venv
pre-commit install --install-hooks

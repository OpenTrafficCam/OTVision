echo Install OTVision development environment.
call install.cmd

uv sync --extra inference_cpu
uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push

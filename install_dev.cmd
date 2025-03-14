echo Install OTVision development environment.
call install.cmd
call .venv\Scripts\activate
uv pip install -r requirements-dev.txt --python .venv%
pre-commit install --install-hooks
deactivate

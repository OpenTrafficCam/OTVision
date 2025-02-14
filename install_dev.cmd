echo Install OTVision development environment.
call install.cmd
call .venv\Scripts\activate
uv pip install -r requirements-dev.txt%
pre-commit install --install-hooks
deactivate

echo Install OTVision development environment.
call install.cmd
call venv\Scripts\activate
pip install -r requirements_dev.txt --no-cache-dir%
pre-commit install --install-hooks
deactivate

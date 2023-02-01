python -m venv venv
call venv\Scripts\activate
timeout /T 1
REM maybe "set GDAL_VERSION=3.2.3" is needed
pip install -r requirements.txt --no-cache-dir
timeout /T 10
deactivate
pause

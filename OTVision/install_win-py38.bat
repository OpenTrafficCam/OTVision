cd ..
python -m venv venv
call venv\Scripts\activate
timeout /T 1
REM maybe "set GDAL_VERSION=3.2.3" is needed
pip install -r requirements_win-py38.txt
timeout /T 10
deactivate
pause
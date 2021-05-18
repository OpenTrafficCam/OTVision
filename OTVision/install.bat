cd ..
python -m venv venv
call venv/Scripts/activate
REM maybe "set GDAL_VERSION=3.2.3" is needed
pip install -r requirements.txt
deactivate
pause
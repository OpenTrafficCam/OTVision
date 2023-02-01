call %~dp0\venv\Scripts\activate
python %~dp0\view.py
timeout /T 3
deactivate

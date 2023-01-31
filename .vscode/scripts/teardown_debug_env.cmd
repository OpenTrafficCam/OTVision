echo Tear down debug environment.
@echo off
set "CURRENT_DIR=%cd%"
set TMP_DIR=%CURRENT_DIR%\.tmp\

rmdir /s /q "%TMP_DIR%"

for /f "delims=" %%i in ('cd') do set CURRENT_DIR=%%i
set TMP_DIR=%CURRENT_DIR%/.tmp/

rmdir "%TMP_DIR%"

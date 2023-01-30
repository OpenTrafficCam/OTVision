for /f "delims=" %%i in ('cd') do set CURRENT_DIR=%%i
set MP4_FILENAME=Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4
set OTDET_FILENAME=Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.otdet
set MP4_FILE=%CURRENT_DIR%/tests/data/%MP4_FILENAME%
set OTDET_FILE=%CURRENT_DIR%/tests/data/%OTDET_FILENAME%
set TMP_DEBUG_DIR=%CURRENT_DIR%/.tmp/debug/

mkdir "%TMP_DEBUG_DIR%"
xcopy /s "%MP4_FILE%" "%TMP_DEBUG_DIR%%MP4_FILENAME%"
xcopy /s "%OTDET_FILE%" "%TMP_DEBUG_DIR%%OTDET_FILENAME%"

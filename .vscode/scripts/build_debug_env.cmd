echo Setup debug environment.
@echo off
set "CURRENT_DIR=%cd%"
set H264_FILENAME=Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.h264
set MP4_FILENAME=Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4
set OTDET_FILENAME=Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.otdet
set H264_FILE=%CURRENT_DIR%\tests\data\%H264_FILENAME%
set MP4_FILE=%CURRENT_DIR%\tests\data\%MP4_FILENAME%
set OTDET_FILE=%CURRENT_DIR%\tests\data\%OTDET_FILENAME%
set TMP_DEBUG_DIR=%CURRENT_DIR%\.tmp\debug\

mkdir %TMP_DEBUG_DIR%
xcopy /i /q %H264_FILE% %TMP_DEBUG_DIR%
xcopy /i /q %MP4_FILE% %TMP_DEBUG_DIR%
xcopy /i /q %OTDET_FILE% %TMP_DEBUG_DIR%

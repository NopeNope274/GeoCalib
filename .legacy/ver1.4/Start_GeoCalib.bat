@echo off
setlocal

set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\geocalib\python.exe"
set "APP_FILE=%APP_DIR%gradio_app.py"
set "WAIT_SCRIPT=%APP_DIR%_wait_server.py"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Run: where python  to check your env path.
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo [ERROR] App file not found: %APP_FILE%
    pause
    exit /b 1
)

echo.
echo [GeoCalib] Cleaning up port 7860 if in use...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860 " 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo [GeoCalib] Starting server...
echo.

start /b "" "%PYTHON_EXE%" "%APP_FILE%"

"%PYTHON_EXE%" "%WAIT_SCRIPT%"

echo.
start http://127.0.0.1:7860

echo.
echo --------------------------------------------------
echo  Server is running. Close this window to stop.
echo --------------------------------------------------
echo.
pause
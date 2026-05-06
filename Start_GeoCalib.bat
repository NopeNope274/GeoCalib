@echo off
setlocal

set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

:: [FIX] Define Conda Environment Path for DLL resolution
set "CONDA_ENV=%USERPROFILE%\miniconda3\envs\geocalib"
set "PYTHON_EXE=%CONDA_ENV%\python.exe"

:: [FIX] Add Library\bin to PATH to solve "ImportError: DLL load failed while importing _lzma"
:: This ensures that critical DLLs (like liblzma.dll) are found by Python.
set "PATH=%CONDA_ENV%\Library\bin;%CONDA_ENV%\Scripts;%CONDA_ENV%;%PATH%"

set "APP_FILE=%APP_DIR%gradio_app.py"
set "WAIT_SCRIPT=%APP_DIR%_wait_server.py"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Please check if your conda environment is located at the expected path.
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

:: Use a separate window for the server to allow monitoring logs in case of crashes
start "GeoCalib Server" "%PYTHON_EXE%" "%APP_FILE%"

"%PYTHON_EXE%" "%WAIT_SCRIPT%"

echo.
start http://127.0.0.1:7860

echo.
echo ---------------------------------------------------
echo  Server is running. 
echo  - To stop the server: Close the 'GeoCalib Server' window.
echo  - To exit this launcher: Close this window.
echo ---------------------------------------------------
echo.
pause

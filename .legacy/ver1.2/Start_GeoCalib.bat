@echo off
:: 한글 깨짐 방지
chcp 65001 > nul
cd /d %~dp0

:: [중요] 요셉찡의 컴퓨터에 설치된 'geocalib' 전용 파이썬 경로야.
:: 만약 사용자 이름이 'user'가 아니라면 본인 계정명으로 고쳐줘!
set PYTHON_EXE=C:\Users\user\miniconda3\envs\geocalib\python.exe

:: 만약 위 경로가 아니라면 보통 아래 경로 중 하나야 (확인 필수!)
:: set PYTHON_EXE=%USERPROFILE%\miniconda3\envs\geocalib\python.exe
:: set PYTHON_EXE=%USERPROFILE%\anaconda3\envs\geocalib\python.exe

if not exist "%PYTHON_EXE%" (
    echo [에러] 파이썬 실행 파일을 찾을 수 없습니다! 
    echo 현재 설정된 경로: %PYTHON_EXE%
    echo 본인의 미니콘다 설치 경로를 확인하고 배치 파일을 수정해 주세요.
    pause
    exit
)

echo 🚀 GeoCalib 엔진 가동 중...
:: 전용 파이썬으로 직접 실행! (환경 활성화 필요 없음)
start /b "" "%PYTHON_EXE%" gradio_app.py

:WAIT_LOOP
:: 7860 포트가 열렸는지 1초마다 체크해
powershell -Command "try { $t = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 7860); if ($t.Connected) { $t.Close(); exit 0 } } catch { exit 1 }" > nul 2>&1
if %errorlevel% equ 0 (
    echo ✨ 서버가 준비되었습니다! 브라우저를 엽니다.
    start http://127.0.0.1:7860
    goto END
)
echo . (서버 대기 중)
timeout /t 1 /nobreak > nul
goto WAIT_LOOP

:END
echo.
echo 작업을 시작하세요, 요셉님! 창을 닫으면 종료됩니다.
pause
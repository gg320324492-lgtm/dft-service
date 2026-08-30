@echo off
rem dft-service 启动 (幂等 — 8620 已监听则跳过, 防双开)
netstat -ano | findstr /R /C:":8620 .*LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo [dft-service] already listening on 8620, skip
    exit /b 0
)
cd /d E:\dft-service
if not exist data\logs mkdir data\logs
start "dft-service" /min cmd /c ".venv\Scripts\python.exe run.py >> data\logs\service-stdout.log 2>&1"
echo [dft-service] starting on http://127.0.0.1:8620

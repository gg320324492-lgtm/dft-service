@echo off
rem dft-service 本地测试入口 (#38, 无 CI remote 的等价物)
rem   run-tests.bat          只跑 mock 测试 (快, 不碰真实计算)
rem   run-tests.bat smoke    mock 测试 + 五后端真算冒烟 (需服务已启动)
cd /d E:\dft-service
".venv\Scripts\python.exe" -m pytest tests\ -q %1 %2 %3
if errorlevel 1 (
    echo [run-tests] pytest FAILED
    exit /b 1
)
echo [run-tests] pytest PASS
if /i "%1"=="smoke" ".venv\Scripts\python.exe" scripts\smoke_all.py

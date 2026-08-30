@echo off
rem dft-service 停止 — 按端口找 PID 树杀
set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8620 .*LISTENING"') do (
    taskkill /PID %%p /T /F
    set FOUND=1
)
if %FOUND%==0 echo [dft-service] not running

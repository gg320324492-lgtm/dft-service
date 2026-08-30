@echo off
rem 移除登录自启快捷方式
set LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\dft-service.lnk
if exist "%LNK%" (
    del "%LNK%"
    echo [dft-service] autostart shortcut removed
) else (
    echo [dft-service] shortcut not found (already removed?)
)

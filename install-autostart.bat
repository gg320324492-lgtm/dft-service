@echo off
rem 注册登录自启 — Startup 文件夹快捷方式 (无需管理员)
rem schtasks ONLOGON 需要管理员权限 (实测拒绝访问), 故用 shell:startup 方案
set LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\dft-service.lnk
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell; $l = $ws.CreateShortcut('%LNK%'); $l.TargetPath = 'E:\dft-service\start.bat'; $l.WorkingDirectory = 'E:\dft-service'; $l.WindowStyle = 7; $l.Save()"
if exist "%LNK%" (
    echo [dft-service] autostart shortcut created: %LNK%
    echo Remove with uninstall-autostart.bat
) else (
    echo [dft-service] FAILED to create shortcut
)

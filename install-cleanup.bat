@echo off
schtasks /Create /TN "MicroBubble-DFT-Cleanup" /TR "E:\dft-serviceun-cleanup.bat" /SC WEEKLY /D SUN /ST 09:07 /F
if %errorlevel%==0 (
    echo [dft-cleanup] weekly task registered
) else (
    echo [dft-cleanup] schtasks failed
)

@echo off
rem 每周清理 dft-service 过期 job 目录 (保留 14 天)
"E:\dft-service\.venv\Scripts\python.exe" "E:\dft-service\scripts\cleanup.py" --days 14 >> "E:\dft-service\data\logs\cleanup.log" 2>&1

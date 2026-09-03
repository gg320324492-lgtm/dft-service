@echo off
rem 每周清理 dft-service 过期 job 目录 (保留 14 天)
rem 缺口 #35: --archive 删前 zip 到 data/archives; --backup SQLite 在线备份留 7 份
"E:\dft-service\.venv\Scripts\python.exe" "E:\dft-service\scripts\cleanup.py" --days 14 --archive --backup >> "E:\dft-service\data\logs\cleanup.log" 2>&1

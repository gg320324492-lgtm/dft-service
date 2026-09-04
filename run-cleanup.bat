@echo off
rem dft-service weekly cleanup of expired job dirs (keep 14 days)
rem gap #35: --archive zips to data/archives before delete; --backup keeps 7 SQLite online backups
"E:\dft-service\.venv\Scripts\python.exe" "E:\dft-service\scripts\cleanup.py" --days 14 --archive --backup >> "E:\dft-service\data\logs\cleanup.log" 2>&1

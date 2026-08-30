@echo off
rem dft 命令行入口 — 把 E:\dft-service 加入 PATH 后可直接 `dft wait gaussian --smiles O`
"E:\dft-service\.venv\Scripts\python.exe" "E:\dft-service\dft_cli.py" %*

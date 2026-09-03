# dft-service — 项目上下文 (给 Claude Code)

微纳米气泡课题组的独立 DFT/MD 计算服务。5 后端: Gaussian 16W / GROMACS (WSL) /
MACE (GPU) / PySCF (WSL 回退) / Psi4。用户主要在 Claude Code 里通过 `dft_cli.py`
自己调用。

## 常用命令

```bash
cd /e/dft-service
./.venv/Scripts/python.exe run.py                      # 启动 (或 start.bat, 幂等)
./.venv/Scripts/python.exe -m pytest tests/ -q         # 测试 (当前 64 PASS)
./.venv/Scripts/python.exe dft_cli.py tools            # 健康检查
./.venv/Scripts/python.exe dft_cli.py wait pyscf --smiles O --basis sto-3g   # 真算冒烟
./.venv/Scripts/python.exe scripts/smoke_all.py        # 五后端真算一键验收 (改 driver 后必跑)
./.venv/Scripts/python.exe scripts/cleanup.py --days 7 --dry-run   # 清理预览
stop.bat                                               # 停止
```

## 架构铁律 (改代码前必读)

1. **两层热加载语义**: `dft_service/` 包内模块 (api/runners/taskstore/executor/
   main/config) 改动必须**重启服务**; `runners/drivers/*.py` 每次子进程现读,
   改完即生效。分不清就重启。
2. **service 本体零科学计算依赖**: rdkit/ase/mace/pyscf/psi4/torch 全在
   `E:\sci-software\conda-envs\scichem\python.exe` (或 WSL python3), service venv
   只装 fastapi/uvicorn/sqlalchemy/aiosqlite。**永远不要往 service venv 装科学包**,
   新计算能力 = 写新 driver + scichem 里装包。
3. **driver 协议**: `python <driver> <workdir>` — 读 `<workdir>/params.json`,
   永远写 `<workdir>/result.json` (失败也写 status=failed), 打印到 stdout 双通道。
   可选写 `<workdir>/progress.json` (#22 进度, /dft/status 回读)。
   公共骨架在 `_driver_common.py` (run_driver 兜异常)。
4. **WSL 路径**: Windows 路径进 WSL 必须 win_to_wsl_path (`E:\x` → `/mnt/e/x`);
   params 里的路径值在 execute_driver_wsl 里自动转换 (只转真实存在的)。
5. **WSL 残留进程**: 只杀 wsl.exe 留 gmx 继续跑。树杀用 taskkill /T (Windows) +
   wsl pkill -f <workdir.name> (WSL, cmdline 含唯一 /tmp/<workdir.name>)。
   新 WSL 类 driver 的 cmdline 必须含 workdir 名才能被 pkill 命中。
6. **中文 Windows 编码**: 所有 subprocess 显式 `encoding="utf-8", errors="replace"`
   — 默认 GBK 撞 wsl.exe UTF-16 banner 会 UnicodeDecodeError/丢 stderr。
   CLI 入口 stdout/stderr 已 reconfigure UTF-8。
7. **外部依赖修改纪律**: `E:\sci-software\workflows\*.py` 是共享代码, 改动必须
   向后兼容 (新参数带默认值), 注释里写日期和原因。本仓库 drivers 可自由改。
8. **错误哲学**: 编排层和 CLI 永不抛异常 — 失败包成
   `{status: failed|unavailable|timeout, error_msg}` dict。不可用 ≠ 失败。
9. **测试**: tests/ 全部 mock execute_driver (不真跑计算); 真算验证靠 CLI 手动
   冒烟。monkeypatch 目标是 `dft_service.runners.execute_driver` (from-import
   绑定在 runners 包命名空间, 不是 executor 模块 — 类 20.181)。
10. **每任务唯一 workdir** (`<tool>_<task_id>`), 同时是 WSL pkill 的 tag 和
    cleanup 的单位; cleanup 只动终态任务目录。

## 已知边界 (有意为之, 不是 bug)

- GROMACS 拓扑是 demo 级 GROMOS 简易模板 (workflows 里注释明说), 真研究用
  Amber/Charmm 参数化
- Gaussian 真跑依赖 license server, mock 测试只覆盖到 gjf 生成/提交/解析链路
- psi4 driver 不支持溶剂 (psi4_runner 无此包装); PySCF 用 C-PCM, Gaussian 用 SMD
- mdrun 固定 `-ntmpi 1 -nt 4` (小盒子域分解会炸; 大体系换真 force field 时再调)

## 与 microbubble-agent 的接缝

- microbubble 通过 `app/services/dft_client.py` + `app/api/v1/dft.py` 薄代理调用,
  容器内 URL 是 `http://host.docker.internal:8620` (docker-compose 已注入)
- 工具名/schema 与服务端点一一对应; 本服务加参数不破坏 microbubble
  (pydantic 忽略多余字段), 删参数前先改 microbubble

## MCP server (mcp_server.py)

已注册到 Claude Code user 作用域 (`claude mcp list` 可见, 全会话可用):
7 个原生工具 (dft_tools/dft_wait/dft_submit/dft_status/dft_result/dft_cancel/dft_list)。
改 mcp_server.py 后需在新会话生效 (stdio 进程随会话启动); SDK 固定 mcp<2
(2.x 把 FastMCP 改名为 MCPServer)。注册命令见文件头 docstring。

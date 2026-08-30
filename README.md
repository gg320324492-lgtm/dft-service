# dft-service — 微纳米气泡课题组独立 DFT/MD 计算服务

从 microbubble-agent 抽出的完整 DFT 计算系统, 部署在 `E:\dft-service\`。
**service 本体零科学计算依赖** — 所有化学计算通过 `E:\sci-software\conda-envs\scichem\python.exe`
子进程跑 driver 脚本, 任务持久化用 SQLite。

## 支持的后端 (5 个)

| 后端 | 用途 | 依赖 | 实测状态 (2026-08-30) |
|------|------|------|------|
| **Gaussian 16W** | DFT 金标准 (opt/sp/freq + SMD 溶剂 + 热化学量) | g16.exe + license | ✅ g16.exe 就位 |
| **GROMACS 2023.3** | 经典 MD | WSL Ubuntu-24.04 | ✅ EM+MD+轨迹真算通过 |
| **MACE-MP 0.3.16** | 机器学习力场 (秒级优化) | scichem (GPU cu128) | ✅ 真收敛步数输出 |
| **Psi4 1.9.1** | 开源 DFT (properties: 偶极/HOMO-LUMO) | scichem | ✅ 与 PySCF 能量交叉一致 |
| **PySCF** | 开源 DFT (C-PCM 溶剂 + UKS 开壳层) | WSL Ubuntu-24.04 | ✅ **WSL 自动回退已接通**, C-PCM 溶剂化能量物理正确 |

## 启动

```bash
cd E:\dft-service
start.bat          # 幂等启动 (已监听则跳过); 登录自启已注册, 一般不用手动
stop.bat           # 停止
```

首次安装:
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests\ -v
install-autostart.bat   # 注册登录自启 (Startup 快捷方式, 免管理员)
```

可选: 复制 `.env.example` 为 `.env` 或直接 set 环境变量 (鉴权/端口/路径全可配)。
**对外部署必须设置 `DFT_SERVICE_API_KEY`**, 内网本地用可不开。

## CLI (Claude Code / 终端推荐入口)

```bash
CLI=E:\dft-service\.venv\Scripts\python.exe E:\dft-service\dft_cli.py

# 短任务: 阻塞到完成, 人类可读摘要 (退出码 0/1/2/3/4 = 成功/失败/超时/不可用/取消)
%CLI% wait gaussian --smiles O --job opt --solvent water
%CLI% wait pyscf --smiles CCO --basis 6-31g --solvent ethanol
%CLI% wait auto --task optimize --quality fast --smiles O     # 智能选路
%CLI% wait mace --smiles CCO

# 长任务 (Bash 工具 ~10 分钟上限): submit 两段式
%CLI% submit gromacs --smiles O --time-ns 10     # → task_id
%CLI% result <task_id>                            # 稍后任意时刻查
%CLI% cancel <task_id>                            # 树杀运行中进程

# 批量筛选 (#15)
%CLI% wait gaussian --smiles-file mols.txt --summary results.csv

# 产物清理 (#10): 只动终态任务, 运行中绝不碰
%CLI% cleanup --days 7 --dry-run

# 其他
%CLI% tools / list --status success / status <task_id>
```

退出码: 0 成功 / 1 失败 / 2 超时 / 3 服务不可用 / 4 已取消。
stdout 永远只有结果 (`--json` 时为纯 JSON 可直接管道), 进度走 stderr。

## 端点

```
GET  /health                    存活探针 (免鉴权)
GET  /dft/tools                 5 后端健康状态 (免鉴权)
POST /dft/gaussian              Gaussian (SCRF 溶剂 + charge/mult 自动推断)
POST /dft/gromacs               GROMACS MD
POST /dft/mace                  MACE 优化 (真实 converged/n_steps)
POST /dft/pyscf                 PySCF (RDKit 3D + C-PCM + UKS, WSL 自动回退)
POST /dft/psi4                  Psi4 (energy/optimize/properties)
POST /dft/auto                  智能选路 (带 warnings: 参数被丢弃会明说)
GET  /dft/status/{task_id}      查状态 (重启后仍可查 + 启动清扫 interrupted)
GET  /dft/result/{task_id}      拿结果
GET  /dft/jobs                  任务列表
DELETE /dft/jobs/{task_id}      取消 (树杀进程树 + WSL 残留清理)
```

鉴权: 除 `/health` `/dft/tools` 外全部需要 `X-API-Key: <key>` (配置了 key 时)。

### 快速示例

```bash
# 提交 Gaussian 水相单点 (M06-2X/def2-TZVP, 阳离子双重态)
curl -X POST http://127.0.0.1:8620/dft/gaussian \
  -H "Content-Type: application/json" -H "X-API-Key: YOUR_KEY" \
  -d '{"smiles":"CCO","xc":"M06-2X","basis":"def2-TZVP","job":"sp","solvent":"water","charge":1,"multiplicity":2}'

# 轮询
curl -H "X-API-Key: YOUR_KEY" http://127.0.0.1:8620/dft/result/<task_id>
```

## 16 缺口收口记录 (2026-08-30)

1. **Gaussian solvent 真生效** — SCRF=(SMD,Solvent=X) 写进路由关键字; 默认 `none` 气相
2. **charge/multiplicity 自动推断** — SMILES 推断 (与 PySCF/Psi4 同逻辑), 显式传值优先, result 报实际值
3. **freq 结果解析** — 频率列表/虚频计数/ZPE/焓/Gibbs 全提取, 虚频带 warning
4. **重启清扫** — 启动时残留 queued/running → interrupted
5. **dft_cli.py CLI** — wait/submit 两段式适配 Bash 超时, 语义化退出码
6. **DELETE 取消** — taskkill /T 树杀 + WSL pkill 按唯一名清残留
7. **超时进程清理** — 同一树杀机制; 全链 subprocess utf-8/replace (GBK 崩溃根除)
8. **并发信号量** — mace=1 防 GPU 争抢, 其余=2, 超发排队
9. **登录自启** — Startup 快捷方式 (schtasks ONLOGON 需管理员走不通)
10. **cleanup** — 终态任务目录按 mtime 清理, --dry-run/--purge-rows
11. **文件日志** — data/logs/dft-service.log 5MB×3 轮转
13. **auto warnings** — 参数被选中后端丢弃时明确返回
14. **C-PCM 实测** — 水 B3LYP/6-31G: 气相 -76.38546 vs 水相 -76.39870 Ha (稳定化 ~8.4 kJ/mol, 物理正确)
15. **批量** — CLI --smiles-file + --summary CSV
16. **CLAUDE.md/README** — 项目上下文齐备

另外修复 E:\sci-software\workflows 的 7 个真 bug (GROMACS 链路从部署起从未跑通):
pybel bytes / _copy_to_wsl 拷到 Windows 假 /tmp / _copy_from_wsl 静默失败 /
steepest→steep / nsteps 差 1000 倍 + NPT 缺参 / nstxout 写 .trr 不是 .xtc /
.gro 原子数声明≠写入 / mdrun 小盒子域分解 (-ntmpi 1)。

## 架构

```
┌──────────────────────────────────────────────────────────┐
│ 客户端: Claude Code (dft_cli.py) / microbubble (代理) / curl  │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP + X-API-Key
┌──────────────────────▼───────────────────────────────────┐
│ FastAPI (dft_service/api.py)   端点+鉴权+信号量+取消        │
│ taskstore.py   内存 dict + SQLite 双写 (重启可查+清扫)      │
│ runners/       参数组装 + health check + auto 选路          │
│ executor.py    Popen 登记 + 树杀 + 超时清理 + WSL pkill     │
└──────────────────────┬───────────────────────────────────┘
                       │ 子进程 (service 本体零科学依赖)
┌──────────────────────▼───────────────────────────────────┐
│ drivers/ (scichem python 或 WSL python3 下跑)              │
│  gaussian_driver  rdkit→gjf(含 SCRF)→g16.exe→解析(含freq)  │
│  gromacs_driver   workflows/gromacs_runner (WSL gmx)      │
│  mace_driver      rdkit→xyz→mace relax_trajectory         │
│  pyscf_driver     xyz→pyscf RKS/UKS(±C-PCM) 两段式        │
│  psi4_driver      workflows/psi4_runner (energy/opt/prop)  │
└──────────────────────────────────────────────────────────┘
```

复用 `E:\sci-software\workflows\` 的成熟代码 (submit_gjf/parse_log/gromacs_runner/mace_relaxation/psi4_runner), **不重写算法**;
只有 gaussian gjf 生成自建 (原版 route 写死不支持 SCRF)。

## 与 microbubble-agent 的关系

- microbubble 侧保留同名 7 端点的**薄代理** (`app/api/v1/dft.py`) + 5 个同名 agent @tool (HTTP 调用本服务),
  前端 `/dft` 页面与聊天工具调用无感迁移
- 任务数据归属本服务 SQLite (`data/dft_service.db`); microbubble 老的 PostgreSQL `dft_jobs` 表保留不再写入
- 配置: `DFT_SERVICE_URL` (docker 容器内用 host.docker.internal) + `DFT_SERVICE_API_KEY`

## 排错

| 现象 | 处置 |
|------|------|
| `/dft/tools` 某后端 available=false | 看 details: g16 路径 / WSL distro / scichem 包名 |
| pyscf unavailable | `wsl -d Ubuntu-24.04 python3 -m pip install pyscf` (WSL 回退已自动, 一般不用管) |
| gromacs unavailable | `wsl -l -v` 确认发行版; 或设 `DFT_SERVICE_WSL_DISTRO` |
| gaussian 提交卡很久 | 正常 — opt 可能数小时; CLI `submit` 两段式 |
| result.json 报 driver timeout | 调大 request 的 timeout_s (已自动树杀不留孤儿) |
| 任务列表有 interrupted | 上次服务重启时在跑, 结果不可恢复, 重新提交 |
| CLI 报 UnicodeEncodeError | 入口已 reconfigure UTF-8; 旧终端仍出现则 `set PYTHONIOENCODING=utf-8` |

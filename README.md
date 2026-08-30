# dft-service — 微纳米气泡课题组独立 DFT/MD 计算服务

从 microbubble-agent 抽出的完整 DFT 计算系统, 部署在 `E:\dft-service\`。
**service 本体零科学计算依赖** — 所有化学计算通过 `E:\sci-software\conda-envs\scichem\python.exe`
子进程跑 driver 脚本, 任务持久化用 SQLite。

## 支持的后端 (5 个)

| 后端 | 用途 | 依赖 | 实测状态 (2026-08-30) |
|------|------|------|------|
| **Gaussian 16W** | DFT 金标准 (opt/sp/freq + SMD 溶剂) | g16.exe + license | ✅ g16.exe 就位 |
| **GROMACS 2023.3** | 经典 MD | WSL Ubuntu-24.04 | ✅ |
| **MACE-MP 0.3.16** | 机器学习力场 (秒级优化) | scichem (GPU cu128) | ✅ |
| **Psi4 1.9.1** | 开源 DFT (properties: 偶极/HOMO-LUMO) | scichem | ✅ |
| **PySCF** | 开源 DFT (C-PCM 溶剂 + UKS 开壳层) | scichem | ⚠ 需 `pip install pyscf` |

## 启动

```bash
cd E:\dft-service

# 首次: 建独立 venv (不污染宿主)
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt

# 跑测试
.venv\Scripts\python -m pytest tests/ -v

# 启动
.venv\Scripts\python run.py
# → http://127.0.0.1:8620  (Swagger: /docs)
```

可选: 复制 `.env.example` 为 `.env` 或直接 set 环境变量 (鉴权/端口/路径全可配)。
**对外部署必须设置 `DFT_SERVICE_API_KEY`**, 内网本地用可不开。

## 端点

```
GET  /health                    存活探针 (免鉴权)
GET  /dft/tools                 5 后端健康状态 (免鉴权)
POST /dft/gaussian              提交 Gaussian (solvent/charge/multiplicity/nproc/mem 全支持)
POST /dft/gromacs               提交 GROMACS MD
POST /dft/mace                  提交 MACE 优化
POST /dft/pyscf                 提交 PySCF (energy/optimize + C-PCM + UKS)
POST /dft/psi4                  提交 Psi4 (energy/optimize/properties)
POST /dft/auto                  智能选路: task+quality → 最快可用后端
GET  /dft/status/{task_id}      查状态 (重启后仍可查 — SQLite 回退)
GET  /dft/result/{task_id}      拿结果 (同上)
GET  /dft/jobs?tool=&status=&limit=&offset=   任务列表
```

鉴权: 除 `/health` `/dft/tools` 外全部需要 `X-API-Key: <key>` 请求头 (配置了 key 时)。

### 快速示例

```bash
# 健康检查
curl http://127.0.0.1:8620/dft/tools

# 提交 Gaussian 水相单点 (M06-2X/def2-TZVP, 阳离子双重态)
curl -X POST http://127.0.0.1:8620/dft/gaussian \
  -H "Content-Type: application/json" -H "X-API-Key: YOUR_KEY" \
  -d '{"smiles":"CCO","xc":"M06-2X","basis":"def2-TZVP","job":"sp","solvent":"water","charge":1,"multiplicity":2}'

# 智能选路: 快速优化 → MACE; 高精度能量 → Gaussian/Psi4/PySCF 择优
curl -X POST http://127.0.0.1:8620/dft/auto \
  -H "Content-Type: application/json" -H "X-API-Key: YOUR_KEY" \
  -d '{"smiles":"O","task":"optimize","quality":"fast"}'

# 轮询
curl -H "X-API-Key: YOUR_KEY" http://127.0.0.1:8620/dft/result/<task_id>
```

## 相比 microbubble 版修复的 8 个缺口

1. **Gaussian solvent 真生效** — SCRF=(SMD,Solvent=X) 写进路由关键字 (原来只拼 title 是摆设); 默认 `none` 气相, 不再悄悄改变能量
2. **charge/multiplicity/nproc/mem 全透出** — request schema + @tool 层都有 (原来离子/自由基算不了)
3. **PySCF 重写** — 原版把 SMILES 原样塞 `gto.M()` 必报错; 现在 RDKit 生成 3D 几何, spin≠0 自动 UKS, optimize 用 geomeTRIC, 输出 `scf_converged`
4. **MACE 真 converged/n_steps** — 统一走 `relax_trajectory` 拿真实值 (原来硬编码 converged=True + n_steps 造假)
5. **任务状态重启不丢** — `/status` 内存 miss → SQLite 回退 (原来 404, 且与 `/result` 行为不一致)
6. **新增 `/dft/jobs` 列表** — 按 tool/status 过滤 + 分页 (原来只能按 id 单查)
7. **API key 鉴权** — `X-API-Key` 头, 未配置时放行并启动警告 (原来完全裸奔, 未登录也能提交数小时计算)
8. **Psi4 接入 + `/dft/auto` 智能选路** — fast→MACE / accurate→Gaussian→Psi4→PySCF 择优 / md→GROMACS (原来 "multimodel 统一接口" 只有名字)

另外: WSL 发行版自动探测 (原来硬编码 "Ubuntu", 实际机器是 Ubuntu-24.04, 直接不可用)。

## 架构

```
┌────────────────────────────────────────────────────────┐
│ 客户端: microbubble-agent (代理+agent tool) / curl / 前端 │
└──────────────────────┬─────────────────────────────────┘
                       │ HTTP + X-API-Key
┌──────────────────────▼─────────────────────────────────┐
│ FastAPI (dft_service/api.py)     端点 + 鉴权 + 任务登记   │
│ taskstore.py   内存 dict + SQLite 双写 (重启可查)         │
│ runners/       参数组装 + health check + auto 选路        │
│ executor.py    subprocess scichem_python driver.py      │
└──────────────────────┬─────────────────────────────────┘
                       │ 子进程 (service 本体零科学依赖)
┌──────────────────────▼─────────────────────────────────┐
│ drivers/ (scichem python 下跑)                           │
│  gaussian_driver  rdkit→gjf(含 SCRF)→g16.exe→parse_log  │
│  gromacs_driver   workflows/gromacs_runner (WSL gmx)    │
│  mace_driver      rdkit→xyz→mace relax_trajectory       │
│  pyscf_driver     rdkit 3D→pyscf RKS/UKS(±C-PCM)        │
│  psi4_driver      workflows/psi4_runner (energy/opt/prop)│
└────────────────────────────────────────────────────────┘
```

复用 `E:\sci-software\workflows\` 的成熟代码 (submit_gjf/parse_log/gromacs_runner/mace_relaxation/psi4_runner), **不重写算法**;
只有 gaussian gjf 生成自建 (原版 route 写死不支持 SCRF)。

## 与 microbubble-agent 的关系

- microbubble 侧保留同名 7 端点的**薄代理** (`app/api/v1/dft.py`) + 5 个同名 agent @tool (HTTP 调用本服务),
  前端 `/dft` 页面与聊天工具调用无感迁移
- 任务数据归属本服务 SQLite (`data/dft_service.db`); microbubble 老的 PostgreSQL `dft_jobs` 表保留不再写入
- 详细环境变量见 `.env.example`; 日志看 stdout (uvicorn)

## 排错

| 现象 | 处置 |
|------|------|
| `/dft/tools` 某后端 available=false | 看 details 字段: g16 路径 / WSL distro / scichem 包名 |
| pyscf unavailable | `"E:/sci-software/conda-envs/scichem/python.exe" -m pip install pyscf` |
| gromacs unavailable | `wsl -l -v` 确认发行版; `wsl -d Ubuntu-24.04 gmx --version`; 或设 `DFT_SERVICE_WSL_DISTRO` |
| gaussian 提交卡很久 | 正常 — opt 可能数小时; timeout_s 上限受 request 控制 |
| result.json 报 driver timeout | 调大对应 request 的 timeout_s (service 侧留了 +120s 缓冲) |

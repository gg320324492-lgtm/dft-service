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
   (勘误: 当时组装链路实为死代码, freq 任务必崩 — 2026-09-04 #17 才真修复)
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

## 第二批收口记录 #17–#26 (2026-09-04)

17. **Gaussian freq 崩溃修复** — freq 提取块误放在 result 赋值前 (UnboundLocalError,
    #3 实为死代码); 移到赋值后, stub 测试 + 真跑 freq 均验证
18. **Gaussian %chk 并发冲突** — chk 用唯一名 `dft_job_<workdir>`; 正常/异常/超时
    路径清安装目录残留; cancel 树杀场景由 cleanup 按 mtime 兜底扫 `dft_job_*`
19. **timeout 一等状态** — driver/executor 超时返回 status=timeout (原被归成 failed),
    CLI 退出码 2 生效; pyscf geom 阶段超时同样传播
20. **参数上限** — timeout_s≤7d, nproc≤64, mem 正则 `^\d{1,5}(MB|GB|TB)$` (防 gjf 注入),
    gromacs/mace 尺度参数封顶; 超限 422
21. **_TASKS 内存驱逐** — 提交时摊还清扫终态超 24h 条目 (硬上限 2000), DB 回退照旧
22. **进度上报** — driver 写 workdir/progress.json (原子写), /dft/status 在
    queued/running 时回读; gaussian 轮询带 opt_step/scf_cycles, pyscf mf.callback
    累计 SCF 周期, gromacs/mace/psi4 stage 级; CLI wait 进度行显示 stage/步数
23. **批量并发 + 断点续跑** — --smiles-file 全部 submit 后轮询 (吞吐不再被串行砍),
    CSV 逐条增量落盘; `--resume` 复用 summary: 终态跳过, timeout/running 按原
    task_id 重轮询
24. **smoke_all.py** — 五后端真算一键验收 (默认 pyscf/mace/psi4, --with-gaussian
    --with-gromacs 显式开启); 本次实测 5/5 PASS, gaussian freq 出真频率
25. **driver 组装单测** — test_gaussian_driver.py stub rdkit/g16/parse_log 测
    freq 合并/SCRF 路由/唯一 chk/422 边界; test_batch.py fake api 测批量并发+resume
26. **健康探测预热** — lifespan 后台线程启动即填 TTL 缓存, /dft/tools 冷缓存
    几分钟问题消除 (实测启动 6s 内 5/5 available)

测试 28 → 45 PASS。

## 第三批收口记录 #27–#31 (2026-09-04, Phase 2 科学能力)

27. **opt freq 联跑** — `--job "opt freq"` / auto `task=opt+freq` 一次拿几何+频率+热化学;
    **实测揪出并修复完成判定 bug**: G16 多步任务在第 1 步末就写 "Normal termination"
    再接 "Proceeding to internal job step 2", 旧轮询只看字符串会拷回半成品日志
    (频率段缺失) — 现要求进程真正退出才算完成。实测 H₂O: 3 频率/0 虚频/G=-75.319 Ha
28. **extra_route 逃生舱** — 路由行追加任意关键字 (int/scf/geom 等), API+driver 双层
    白名单 (挡换行/#/%/\\), result.route 回显最终路由; TDDFT/NBO/scan 等进阶用法解锁
29. **内联 xyz 输入** — gaussian/pyscf/mace 收 `xyz_content` (与 smiles 二选一;
    xyz 时 charge/mult 必须显式), CLI `--xyz-file` 读文件; MD/优化接力几何可直接精修。
    实测 xyz 水单点 −75.3127 Ha (STO-3G)
30. **ΔGsolv 一键双算** — pyscf `solvation_energy=true`: 同几何气相+C-PCM, 返回
    `delta_solvation_kj_mol` + 两个分量 + 语义标注 (电子静态项, ≠实验全 ΔGsolv)。
    实测水 −27.2 kJ/mol (B3LYP/6-31G*, WSL pyscf)
31. **GROMACS 分析 v1** — `analyze=true`: MD 后 gmx rms (System 组, 不写死 Protein) +
    gmx energy → rmsd_avg/max_nm、potential_avg、temperature_avg + PNG 出图 (无
    matplotlib 静默跳过); 分析失败只降级 warning 不推翻已成功的 MD。实测产物齐全

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

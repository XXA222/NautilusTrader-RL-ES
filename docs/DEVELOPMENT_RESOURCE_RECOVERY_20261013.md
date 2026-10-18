# 本轮开发目标：父进程硬退出后的 CPU worker 租约恢复

依据最新审计与下一阶段 N04。本轮目标：为 Windows Job 管理的 CPU worker 租约提供只读盘点、显式恢复和真实父进程强杀验证，避免只能手工删除 SQLite 记录才能续跑。

## 验收要求（实现与故障运行前声明）

1. 新 lease 在 worker 启动前记录 owner 的机器、Windows session、PID 与原始进程创建时间；记录由 token 派生、不可复用的具名 Job。PID 重用不得被误认作原 owner 仍在或任意释放其他 owner。
2. 仅完整启动、已封存 Job 清单的 CPU worker lease 可恢复。恢复者提供精确 token、owner、CPU/RAM 和非空原因；检查原 owner 已退出，所有原 Job 已销毁或 active_processes=0，再在同一 SQLite 写事务中释放配额并保存回执。
3. 活 owner、活 worker/后代、身份不明、异机器/session、错误描述、GPU、旧 lease、启动未完成或证据损坏必须失败关闭；重复恢复拒绝，不退款研究查询，不清理 GPU owner 文件。
4. 只读盘点不创建数据库、不迁移 schema、不更新文件。新进程恢复后可重新取得同一预算；并发恢复最多一个成功。
5. 使用自有父进程与 worker 做 OS 强杀；受控 KeyboardInterrupt 不能代替。覆盖存活后代、启动失败和正常释放回归。

## 实施边界

本轮原始 CPU worker 恢复覆盖启用 Windows Job 内存限制的 CPUWorkerPool。未启用 Job 的 CPU pool、进程内 learner、旧租约及启动未完整完成的残留仍保留人工处理边界。2026-10-17 已增加单独的受监督 CUDA 应用恢复路径，见下文；它不会把普通 GPU lease 自动转换为可恢复 lease。SQLite 资源配额依然不是协调器/learner 的 OS 内存硬限制；N04 整体不能因此全部关闭。

进程树依据标准 Windows 子进程 Job 继承和不允许 breakaway 的限制；不声称能够约束不受信任代码通过外部服务创建的进程。恢复只检查已登记的自有进程，不执行 PID 强杀。官方语义依据：[Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)。

## 使用方式

```powershell
.venv/Scripts/python.exe -m rles_compute_runtime.recovery inspect --database <run>/quota.sqlite
.venv/Scripts/python.exe -m rles_compute_runtime.recovery recover --database <run>/quota.sqlite --token <精确token> --owner <原owner> --cpu-cores <检查中的CPU> --ram-bytes <检查中的RAM> --reason "自有父进程硬退出后恢复"
```

`ready` 只表示启动已登记，不能说明 owner 或进程树已退出。恢复会重新检查实际 OS 状态，并将原因、完整租约描述和检查证据写入配额库 `lease_recovery_log`。该命令不启动训练、不打开 Validation/Test/Holdout；训练续跑仍须通过原检查点及研究账本门禁。

受监督 CUDA 训练可使用同一 `inspect` 查看 `application_ready` 状态。确认是预期 run 后，只有原 owner 已退出、登记的 Windows Job 为空或已销毁、共享 GPU 内核锁能够被独占取得时，才可执行：

```powershell
.venv/Scripts/python.exe -m rles_compute_runtime.recovery recover-supervised-gpu --database <run>/quota.sqlite --token <精确token> --owner <原owner> --cpu-cores <检查中的CPU> --ram-bytes <检查中的RAM> --reason "受监督应用树已退出并审核通过"
```

该入口只适用于通过 `rles_compute_runtime.supervisor` 启动并在运行前成功登记的 Windows CUDA 应用。它会写入同一恢复收据；旧格式、普通未受监督和未完成登记的 GPU lease 仍失败关闭。它不证明显存已归还到特定数值，也不自动重试训练。

## 本轮结果

已实现：`resources.py` 新增只接受精确 lease 描述的恢复事务与持久回执；`windows_recovery.py` 读取真实 owner 生命周期和具名 Job；`workers.py` 在启动前登记、全部 READY 后封存；`recovery.py` 提供独立只读 `inspect` 与显式 `recover` 命令。具名 Job 与外来同名对象冲突时拒绝复用。

针对性及集成验证 **106 passed、20 warnings，64.47 秒**，覆盖新故障用例、既有 Windows 内存限制/强杀、worker pool、设备配额、原生历史工厂和 WFO 边界。新用例覆盖真实 OS kill、保留 Job 句柄导致活后代阻断、恢复后预算复用、重复/并发恢复、错误 owner/token/预算、PID 重用分类、访问拒绝、异机器/session、旧 schema 只读盘点及 GPU 保留。Ruff 全目录通过；现有 MyPy 配置下 `packages qa` **107 source files** 通过。

**真实原生父进程强杀验收通过**：训练协调器 PID 53312、原生 worker PID 50240；实际 Engine reset 返回 23 列观测、初始原生权益 10000。OS 强杀后协调器 exit -15，原具名 Job 已销毁且 active_processes=0，但 SQLite CPU lease 仍存在。显式恢复后 active leases=0，并成功重新取得/释放同一 1 CPU / 2 GiB 预算。研究数据库前后字节一致，**4 次 ES lane requests 保留**；Validation/Test/Holdout 查询均为 0，无 bundle、无 COMPLETED。故障探针 exit 0 表示观察到预期失败及正确恢复，不表示训练成功。

证据：`.tmp/native-parent-recovery-20261013/{config.json,native-ready.json,acceptance.json,run/quota.sqlite}`、`.tmp/native-parent-recovery-20261013.log`、`.tmp/resource-recovery-integration.{log,xml}`。此探针使用原探索 Train，成本/规则仍为 scenario-only；未读取新的评估结果，也未重试失败的一代。

最终全量 CPU 回归 **390 passed、4 skipped、45 warnings，282.31 秒**。4 项跳过仍为 CUDA 不可用；Ruff 全目录通过，现有 MyPy 范围下 `packages qa` 107 个源码文件通过。完整证据为 `.tmp/resource-recovery-full.{log,xml}`。没有修改 NautilusTrader 上游源码。

另已离线构建 `rles-compute-runtime` / `rles-contracts` 两个 wheel，在全新 `.tmp/resource-recovery-installed/` 非 editable 环境中独立安装并通过 **10 passed / 2.71 秒** 的恢复用例；导入确认为该环境 `Lib/site-packages/rles_compute_runtime`，未使用原工作区 `.pth`。这证明本功能的独立 wheel 路径，不代表全项目发布、Linux/CUDA 或项目整体验收。

nightly 完整表达式现收集 **57/394** 项，其中 `artifact_restore` **40** 项；标记存在重叠，不能简单相加。日志/结果继续按现有 nightly 工件上传配置保留。

复现命令（测试临时目录/原生 probe 根目录需使用全新路径）：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_resource_recovery.py tests/test_windows_worker_limits.py tests/test_compute_worker_pool.py tests/test_compute_device.py tests/test_historical_factory.py tests/test_research_wfo.py -q -p no:cacheprovider --basetemp .tmp/pytest-resource-recovery-integration --tb=short
.venv/Scripts/python.exe .tmp/native_parent_recovery_20261013.py --work-dir .tmp/native-parent-recovery-20261013
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp .tmp/pytest-resource-recovery-full --junitxml .tmp/resource-recovery-full.xml -o junit_family=xunit1 --tb=short
.tmp/resource-recovery-installed/Scripts/python.exe -m pytest tests/test_resource_recovery.py -q -p no:cacheprovider --basetemp .tmp/pytest-resource-recovery-installed --tb=short
```

## N04 协调器/learner OS 内存硬限制

继续推进 N04 的协调器/learner OS 内存硬限制：优先研究在独立监督进程下启动整个训练应用并将其进程树放入 Job，避免在已运行的交互进程上直接设置不可撤销限制。需要覆盖超限失败、子进程清理、检查点保留及原研究查询预算；不得把本轮 worker lease 恢复当作该项已经验收。N02 的新鲜多折策略研究、N03 外部历史事实和 N05 CUDA/90 天吞吐仍保留原优先级与门禁。

## 原生故障探针的冻结范围

新增 `.tmp/native_parent_recovery_20261013.py`，使用既有 `core-execution-20261013/config-a4-clean-hard-risk-fixed.json` 的原 Train 行情与明确场景规则，派生 A1/population 4/generations 1，`train(evaluate=False)`；不是新鲜策略研究。在原生 Nautilus Engine 完成真实 reset 后进入故障屏障，外部监督进程强杀自有训练协调器。预期原 Job 清空、SQLite 保留 CPU lease，显式恢复后可重新取得预算；4 次已预留 ES lane requests 及研究数据库原样保留，无 bundle/COMPLETED，也不继续失败的一代。

### 整体应用进程树监督器（2026-10-17）

新增 `WindowsWorkerJob.run_suspended()` 和 `rles_compute_runtime.supervisor` CLI。监督器以 `CREATE_SUSPENDED` 创建应用主进程，先把进程加入配置了 process/tree commit-memory 上限和 `KILL_ON_JOB_CLOSE` 的 Windows Job，再恢复主线程；所有应用后代继承该 Job。监督器结束时终止残留后代并记录最终活动进程数。`--evidence` 以独占创建方式保存 JSON；记录命令 SHA-256 而不写入可能含敏感参数的完整命令，拒绝覆盖现存证据。

使用方式：

```powershell
python -m rles_compute_runtime.supervisor `
  --memory-bytes 8589934592 `
  --evidence .artifacts/run-123/supervisor.json `
  -- python -m rles_orchestrator.application train --config configs/frozen-train.json
```

Windows 集成回归 20 项通过，覆盖 128 MiB 限额触发终止且保留预先写入的 checkpoint、主进程退出后清理存活后代、超时清理；Ruff 和两个 compute-runtime 源文件的 Windows MyPy 通过。实际以 2 GiB Job 启动 orchestrator `--help`，进程树峰值提交内存 983,347,200 字节，退出码 0，Job 清理后活动进程为 0；记录在 `.artifacts/audit-development-20261017/native-supervisor-help.json`。compute-runtime wheel/sdist 构建并在隔离 wheel 目标安装后，监督器和资源恢复回归另有 13 项通过。

本节首次验收只用 `--help` 验证监督器，没有启动训练或读取行情；这一局部状态已由下方 2026-10-17 follow-up 补充。follow-up 在 6 GiB Windows Job 中运行了实际 64 步 Train-only CUDA 命令并成功完成，见其独立 verification artifact。N04 因此已覆盖当前 Windows 真实训练成功路径，但仍保持 `PARTIAL`：它没有证明 GPU 显存上限、Linux 或新机器能力，也没有把监督器本身纳入 Job 限额。

#### 2026-10-17 follow-up: real train command under the Job cap

在隔离 test registry 中运行实际 `application train --train-only`，使用现有 `tests/fixtures/real_ethusdt_20240101` 作为 CI 集成夹具，窗口 2024-01-01 00:05–08:05 UTC；不构造 evaluation factory。64 步 PPO 明确请求 CUDA，并在 6 GiB Windows Job 内成功结束。监督器记录峰值 tree commit **3,802,705,920 bytes**、进程数 2、最终 active processes 0、exit code 0；bundle 明确写入 `rl_device=cuda` 和 64 个 R1 steps。隔离 registry 仅有一条 development reservation。`verification.json` 独立检查完成收据/bundle SHA、四个归档源 SHA、零 evaluation query 和 registry 记录。证据和完成产物位于 `.artifacts/audit-development-20261017/n04-supervised-train/`。这补齐真实应用成功路径；超限后的 checkpoint 保留仍由独立 OS 故障回归证明，不据此声称 GPU 显存受 Job 限制。

#### 受监督 GPU 租约的正常释放

正常释放受监督 CUDA 配额时，先释放当前 owner 的 GPU 内核锁，再由配额层验证原 owner 仍在精确命名且内存限制未变的 Job 中，并尝试重新独占同一 GPU 锁。若锁仍被占用，不退还配额；若 Job 内还有活动应用后代，也不退还配额。Windows 的 Python 启动器可能令 launcher parent 与实际 owner 同时留在 Job 中，因此 PID 检查只允许 owner 及其进程祖先；这不把任意活动后代视为安全。正常释放、持锁拒绝、活动后代拒绝及硬退出恢复回归均通过：`tests/test_resource_recovery.py tests/test_compute_device.py` **27 passed**，Ruff 和 Windows MyPy 通过。compute-runtime 0.4.1 wheel/sdist 已构建，隔离 wheel import 命中本次源码。以上是租约和进程树边界，不构成 VRAM 上限或 CUDA context 的总显存测量。

# L1–L6 工程持续修复验证

日期：2026-10-11。本记录覆盖当前 Windows CPython 3.14.7 CPU 工程路径，不是 Q0–Q10 发布签核。

最新功能开发见 [原生交易与 RL/ES 集成记录](QA_NATIVE_FUNCTIONS_20261011.md)：已补原生提交器、Episode 装配与日志回放恢复、进程隔离多环境 PPO（含 VecNormalize）、L1 MarketEvent 原生解码与双向 funding 结算金标、真实 native 环境短训/续训/候选评估、L6 WFO 到 L2 `0.4.0` 嵌套 FoldManifest 校验、EvoTorch 0.6.1 CPU 的 PGPE/CMA-ES 状态恢复、L5 E1/E2 完整代际批量评估经 L6 spawn ES lane、真实 Native Engine 在 L6 worker 内执行候选，以及 A0–A4 编排分支测试。本轮补齐 `StageBudget.max_lane_requests` 与 `StageUsage.lane_requests`，将实际 `EvaluationBudget.consumed` 和 candidate objective evaluations 分别写入 L6 receipt；A4 每候选两个 episode，真实核对 4 次 objective evaluations 与 8 次 lane requests。SelectionLedger 在 worker dispatch 前持久预留请求，失败 attempt 消耗协议级预算；checkpoint schema v3 绑定 durable reservation scope，恢复前要求账本请求集合与 checkpoint 精确相等。`ProviderStageOutput` 强制 lane 计数与合法 lane evidence hash 配对。新增 Native Engine 在 spawn pool 重建后的 checkpoint 恢复集成，连续运行与恢复运行的候选结果、evidence 及 L6 receipt 一致。全套 CPU 209 passed / 4 CUDA-only skipped，全项目 MyPy（88 个源码文件）和 Ruff 通过。下文早期 122 项结果与 JSON 属于此前安装验证快照。

## 本轮实际修复

- 将 MyPy 2.3.0 固定到 dev group 和 `uv.lock`，安装到项目解释器；CI 通过 `python -m mypy` 调用，避免误用系统 Python 3.12 的外部命令。当前锁解析 88 个候选包、Windows 环境安装 68 个包。无类型声明的 PyArrow 导入增加模块级明确豁免；它与 Nautilus/EvoTorch/psutil 的第三方 API 仍不受完整静态类型验证。
- 修正 GPU CI 的 `setup-python` action 来源为 `actions/setup-python@v5`。CPU CI 现先构建全部包，再在锁定环境内安装 11 个 wheel（`--no-deps --reinstall`），检查 wheel 来源和导入路径，然后执行测试；后续命令使用 `--no-sync`，保持实际 wheel 安装状态。
- nightly 必需 marker 收集改为 Python `subprocess.run(check=True)`，任一失败立即中止。修复 Windows 多命令步骤可能用后一条成功覆盖前一条失败的问题。本机直接执行该 workflow 的 Python preflight，确认缺 `real_golden` 时因 pytest 退出码 5 被拒绝。
- Compute benchmark 的 9 个新增反例曾全部失败：超大整数泄漏 `OverflowError`，小数工作量计数被接受，错误的重复次数泄漏 `TypeError`，字符串审批被接受，布尔 schema 版本被接受。现统一拒绝非法值，并严格校验 summary 和 sample budget，避免 `False == 0` 绕过报告一致性检查。
- Worker pool 现提前拒绝非整数/布尔资源计数、NaN/Infinity 及超过平台锁上限的 timeout；同一线程重入 dispatch 也会被拒绝。新增 13 项真实 spawn/IPC 测试，覆盖两 worker 状态隔离、step 顺序、ES 独立 candidate session、结果归属、异常/超时关闭及配额释放。session 为合成夹具，不是原生 Engine golden。
- L6 WFO 校验器原先假设 L2 提供扁平字段，与实际嵌套 `train/validation/test: Window` 不兼容。现兼容 L2 dataclass 与 `asdict()` 结构，核对 schema、三个窗口的精确 warmup/score 边界、purge/embargo、数据和特征指纹、可选 scaler 指纹及 episode 状态隔离标志；折 ID 由 L6 按确定顺序派生。补充了真实 L2 `build_folds()` 的配对回归与错误输入拒绝测试。
- L5 EvoTorch 0.6.1 CPU 原生恢复现使用版本固定的白名单 JSON/NPZ 编码，不载入 pickle；覆盖 PGPE 分布、ClipUp 动量、CMA-ES 协方差/进化路径、population、问题生成器及 Python/NumPy/Torch RNG、最佳候选和预算计数。PGPE/CMA-ES 两代中断再续跑的候选序列及最终结果与连续运行精确一致；检查点只允许成功代际边界，身份/配置漂移拒绝，恢复异常回滚 RNG 并废弃实例。CUDA codec 与非零数据游标仍未实现。
- `rles-es-evotorch` 本轮构建 sdist + wheel 成功；将最新 wheel 直接置于导入路径后，checkpoint/resume 等价性及 E1/E2→L6 ES lane 用例 5 passed，验证打包产物包含候选 schema 与新 codec。该检查不是 clean-environment 依赖安装复现。
- L5/L6 ES 契约现区分 `candidate_kind` 与 `candidate_schema_hash` 和冻结 `actor_signature`：E1 评估 ActorGenome，E2 保持 R1 Actor 身份、独立校验 SoftRiskSpace 维度/指纹。合成 spawn/IPC 回归实际通过 L6 `ESEvalLane` dispatch E1 与 E2 的 candidate×episode 矩阵；错误 actor/schema 在派发前被拒绝，预算按逻辑请求 key 防止换 genome 后以新 request hash 重复计费。另为 L5 Orchestrator 补齐 A0–A4 五条分支和 E2 修改 R1 actor 的 fail-closed 用例。以上为跨包调度/状态机证据，不是 Native Engine 或真实 E1→R1→E2 研究证据。
- EvoTorch Provider 新增可选 `Problem(vectorized=True)` 批量目标；`GenerationLaneEvaluator` 自动构建完整 generation×episode 请求矩阵，`WorkerLaneDispatcher` 将其映射到注入式 L6 `ESEvalLane`，结果按原始种群顺序回填。PGPE/E1 Actor 与 CMA-ES/E2 SoftRisk 各跑两代、每代四候选、每候选两个场景，真实 spawn worker 返回 genome 相关 fitness 和 evidence；优化器候选计数为 8，逻辑评估预算精确消耗 16。`rles_context` 将请求哈希、seed、genome、隔离账号和 Train-inner 等元数据自动传入 worker，且必须由 worker 回显请求哈希与账号身份。评估配方指纹并入 checkpoint configuration，载荷 builder 身份、场景、seed、schema 或 actor/run 变化都会改变签名。该集成使用合成 L3 evaluate payload 和 session，尚不证明 Native Engine 的生产装配或 worker 真实执行了隔离语义。
- E2 的 CandidateRequest 现在明确携带物理 SoftRisk 乘数，派发前再次调用 `SoftRiskSpace.decode` 检查实际上下界；新增范围外请求拒绝回归，避免调用方绕过 CMA-ES latent 限幅。
- ES worker 不再将 L3 shaped reward 当作候选 fitness：Gym reset/step info 暴露初始/最终 native equity、未整形逐步 log return、drawdown 与风险干预，L5 `FitnessSpec.score_episode` 生成净收益、worst-tail loss、最大回撤和约束项分数。`tail_fraction`、FitnessSpec 权重与 payload builder ID 纳入 WorkerLaneDispatcher 指纹；验证覆盖 tail 计算、非法事实和 Validation 拒绝。
- 批量 evaluator 的逻辑 request keys 和预算上限现在随 EvoTorch CPU checkpoint v2 持久化。恢复时校验 run/attempt/fold/seed/schema/dispatch 身份，再还原历史 key 集；重放已用代际（包括换 genome 保持相同逻辑身份）会被拒绝，后续代继续消耗原预算。新增中断→恢复→拒绝旧代重放→完成剩余代并与连续结果相等的回归。
- L5 正式 `nautilus_rles.es/evotorch` Provider 工厂现在接受 `batch_fitness` 与冻结的 `batch_fitness_id`，支持不提供伪标量 evaluator 的 batch-only 配置；标量 evaluate 在该模式下明确拒绝。新增从已安装 entry point 创建 batch-only Provider 并执行完整一代 PGPE 的回归。
- 修复 E1 的 ActorGenome 搜索 schema 与 R1 Actor contract 被错误要求同一哈希的问题。`candidate_schema_hash` 继续校验 L5 genome 布局，`actor_signature` 可单独钉住下游 L4 Actor contract；二者都进入请求与 checkpoint 配方身份。新增候选×episode 派发回归，确认不同签名完整传递并分别核验。
- L4 `RLProvider.transfer` 现分别接收 L4 destination mapping signature 与可选 L5 `source_actor_signature`；层映射签名仍独立严格核对，验证通过的 parity 报告绑定到源 Actor identity。新增真实 SB3 TD3 policy 权重经 L4 导出/加载和 L5 `transfer()` 的 DIRECT 集成回归，确认精确 tanh Actor 跨包映射及动作 parity。
- L4 对“权重/映射验证通过但动作 parity 不过”新增专用 `ActorActionParityMismatch` 类型，区别于签名、覆盖、shape/dtype 错误；失败加载会回滚。测试适配器只将此类型翻译为 L5 `StructuralTransferMismatch`，令 L5 generic transfer 执行 Train-inner 蒸馏，并核验教师数据哈希、实测 imitation MSE 与独立固定观测 parity。SB3 PPO + 确定性合成教师回归通过；真实 E1 source、用户批准及生产蒸馏预算仍未由该合成 fixture 证明。
- 将上述 translation 固化为 `rles_orchestrator.adapt_l4_load` / `adapt_l4_distillation` 公共生产 API：异常类由装配方显式注入，L5 不依赖或导入 L4。成功 L4 exact-transfer 报告须包含 source identity、passing parity 和完整 tensor counts，才能生成 L5 load evidence；蒸馏 adapter 从 L4 passing report、L5 student 实际输出生成数据 hash/MSE。SAC/TD3/DDPG DIRECT 与 PPO 回退集成测试均已改用此 API。最新 Orchestrator sdist/wheel 隔离安装后，直接从 wheel 验证两个 adapter 的导入、typed fallback、证据转换和 hash/MSE 复算。
- L6 新增持久化 `SelectionLedger` 阶段收据，并可作为 L5 `Orchestrator` 的 `attest_usage` 回调。它验证 E1/E2 与 R1 计数语义、冻结预算、RunSpec 身份和 StageArtifact 收据哈希，以 SQLite 事务记录唯一阶段/制品并写入 audit chain；新增真实 SQLite 跨 L5/L6 workflow 测试。该项验证接口与持久化机制，阶段 Runner 仍为测试夹具，其 usage 仍是自报数据，不代表生产 provider/lane 用量已接入或 ICR-L6-008 有效 trial 策略获批。
- L5 `StageUsage` 增加 provider-result adapters：R1 可从一个或多个 L4 `Performance.samples` 合计实际 train timestep delta；E1/E2 从 L5 `EvolutionResult.generations/evaluations` 与 provider 的 frozen population size 派生用量。真实 Native Engine 上的 SB3 PPO/SAC/TD3/DDPG 短训、SB3 restore→续训、EvoTorch PGPE/CMA-ES episode 测试现验证转换结果与 Provider 输出一致。生产 Runner 仍须把这些 adapters 用于 StageArtifact，并将 ESEvalLane 的 candidate×episode dispatch 实际消耗与 ES candidate 计数区分核对。
- `StageArtifact.for_stage` 现提供统一装配工厂：从 RunSpec 复制运行指纹、父制品和冻结 data/feature/risk/execution hashes，并调用注入式 L6 receipt provider 预先绑定实际 usage receipt。L5 A4→SelectionLedger 集成测试已切换到此工厂，减少 Runner 手工复制身份字段导致漂移或错账的机会；算法模型导出与真实 L6 lane 计量仍由各生产 StageRunner 完成。
- 新增 `ProviderStageOutput` / `ProviderBackedStageRunner`，将 Provider 结果→用量→预算核验→L6 receipt→StageArtifact 统一起来。L5 不导入 RL、L6 包；执行回调按 StageContext 注入各自的 Provider 和 Env/worker factory。A4 跨包 SQLite 联测现走此 Runner，通过 SB3 `Performance` 与 EvoTorch `EvolutionResult` DTO 覆盖三阶段编排，并固定 E1=PGPE、E2=CMA-ES 搜索器身份。此集成组合测试仍用轻量 DTO 作为 Provider 输出；真实 Provider 输出格式另由 Native Engine 短训/恢复与 ES episode 测试覆盖。更新后的 Orchestrator wheel 构建并从 wheel 路径导入 smoke 通过。生产数据源、Transfer Actor payload 与候选 WorkerLane payload factory 仍未整体装入单一 production runner。
- 新增完整 synthetic A4 Native Engine + L6 spawn 联测：PGPE/E1 与 CMA-ES/E2 都经真实 `ESEvalLane`/`CPUWorkerPool` 评估 candidate×episode，并在 worker 内执行 L3 Gym/Nautilus Engine；E2 worker 初始化时接收 R1 训练后冻结 Actor 的数值快照。SB3 PPO 通过 L4 exact-load parity rejection 与 L5 Train-inner distillation 完成 E1→R1 初始化，再真实训练 4 步；R1 Provider 释放其 VecEnv/Engine 后才启动 E2 lane，保证单进程 Engine 隔离。三个阶段均由 ProviderBackedStageRunner 从 Provider result 建 StageArtifact，并在 L6 SQLite SelectionLedger 持久化。该测试证明本地合成数据功能闭环，不替代真实行情、生产应用数据/风险/配置工厂或外部签核。
- `GenerationLaneEvaluator` 现在将每代 CandidateAggregate/worker evidence 聚合成可核验的 evidence manifest hash，并把按代摘要同 logical request budget 一起纳入 EvoTorch checkpoint。中断恢复会校验 generation 顺序、每代 candidate×episode request 数与预算已派发 keys 一致，随后继续累积同一个 evidence hash。A4 spawn 联测把 E1/E2 lane evidence hash 合入 StageArtifact payload hash，L6 阶段 receipt 因而绑定候选 episode evidence；checkpoint/resume 回归验证连续与恢复运行得到相同 evidence manifest。
- 此次 evaluator/checkpoint 更新后的 `rles-es-evotorch` sdist + wheel 重新构建成功，直接将 wheel 放入导入路径后验证 `GenerationLaneEvaluator.evidence_hash` 新 API 可从发行包导入。
- E1/PGPE 与 E2/CMA-ES 的 L6 spawn 集成测试现在调用真实 L3 Gym 装配及 Nautilus Engine，逐候选执行 2 步并完成原生终止结算；worker 验证传入的 `rles_context`，evidence 绑定请求哈希、隔离账号、最终原生权益和 fitness。它使用合成行情，但补上了“Native Engine 能否在 L6 ES worker 生命周期中运行”的直接证据。
- 当前全仓回归为 200 passed / 4 CUDA-only skipped；MyPy 全部 88 个源码文件通过，Ruff `packages tests` 全部通过。最新 L5 wheel 对应的 E1 identity/batch-only smoke 已通过。SAC、TD3、DDPG 的 L4→L5 DIRECT 编解码/parity，以及 PPO clip/tanh 差异触发 L5 Train-inner 蒸馏回退均有集成测试；另有 A4 E1→R1→E2 合成 Native Engine 全流程与 SQLite 收据联测。PPO 正式 hybrid 放行仍需外部 ICR 与 parity 证据。

- 新增可安装 `rles_orchestrator.native_es_worker` 作为 L6 spawn ES worker 工厂，应用通过公开 L3 环境工厂注入冻结的行情与 Native Engine 组装。E1/E2 候选在真实 Native Engine 上完成一代评估；伪造请求在建 Engine 前拒绝。L6 ES 派发去除不可靠的跨进程 `Queue.empty()` 死亡判断，保留完成结果直至候选 deadline。worker 工厂引用、配置、worker 数、线程数和超时进入 L5 dispatcher checkpoint 身份，配置漂移恢复在派发前失败。安装依赖增加 `native-es-worker` extra 并更新 `uv.lock`；RL 与 ES 包仍保持独立。

- 新增 `ESStageExecutor` 生产装配接口，将 StageContext、EvoTorch Provider、L6 spawn lane、SelectionLedger 请求预留、按代不可变 checkpoint 和 `ProviderStageOutput` 接成 E1/E2 执行路径。A4 合成行情联测已改用它运行真实 Native Engine E1/E2，R1 继续由独立 L4 Provider 训练并由 Orchestrator 持久化三个阶段收据。Native Engine E1 中断一代后的恢复测试现由新的执行器在全新 pool 中续跑，并由 Orchestrator 完成阶段收据；连续/恢复结果、evidence 和 8 条请求对账一致。安装依赖增加 `native-es-runtime` extra；真实行情与 R1 应用装配仍缺。

## 当前验证证据

| 验证路径 | 结果 |
| --- | --- |
| 当前开发环境 CPU 全套 | 200 passed / 4 CUDA-only skipped |
| 当前 Orchestrator sdist + wheel | 构建成功；wheel 隔离安装后 L4 load/distillation adapters smoke 通过 |
| 当前项目解释器 MyPy | 88 个源码文件，0 errors |
| 当前 packages/tests Ruff | 通过 |
| 原开发环境 CPU 全套（早期快照） | 122 passed / 4 CUDA-only skipped |
| 全新环境按 `uv.lock` 安装后 CPU 全套 | 122 passed / 4 CUDA-only skipped |
| 隔离环境替换为 11 个实际 wheel 后 CPU 全套 | 122 passed / 4 CUDA-only skipped |
| wheel 环境 PR marker 集 | 119 passed / 7 deselected |
| 项目解释器 MyPy 2.3.0 | 83 个源文件，0 errors |
| Ruff 当前配置 | 通过 |
| `uv lock --check` / 68 包 `uv pip check` | 通过 |
| 全部运行包构建（前一轮） | 11 sdist + 11 wheel，成功 |
| 当前 L5 源码构建 | sdist + wheel，成功；隔离安装 wheel 的 E1 identity dispatch 与 batch-only entry point smoke 通过 |
| 最新 L5 wheel 路径导入 + episode fitness smoke | 通过 |
| 上一轮 L5 wheel checkpoint/resume suite | 4 passed；本轮仅改评估请求 identity 与 entry point，CPU 源码 checkpoint 回归包含在 193 项全套中 |
| 11 个 wheel 导入路径和 direct URL 来源 | 全部位于隔离环境，来源为本地 wheel，非 editable |
| nightly `real_golden` / `perf` | 两者均 0 用例，pytest exit 5；门禁仍拒绝 |

全新环境使用项目 CPython 3.14.7，Torch distribution 元数据为 2.14.1、运行模块为 2.14.1+cpu、CUDA runtime 为 None。先完成锁定依赖安装和 editable 回归，再卸载 11 个 editable 包并安装构建 wheel，完整回归再次通过。源文件/锁文件 SHA-256、22 个构建包哈希、JUnit 统计和 wheel 导入路径保存在 `qa/reports/engineering_20261011.json`。该 JSON 是工程记录，不具备发布签署效力。

## 仍需保留的阻断

本次补上了 Windows CPU 工作区锁的干净安装和 wheel 回归证据。独立 `requirements/cpu.lock` 的全新 hash-only 安装、Linux CUDA clean install、受控 wheelhouse、SBOM/NOTICE、许可证审查和外部签署仍未完成。GitHub Actions runner 尚未实跑修改后的 workflow。

真实 ETHUSDT 数据/来源证据、订单成交/资金费/保证金规则与交易所对照金标、阶段级 RLES Env/VecEnv checkpoint 恢复、EvoTorch CUDA 原生恢复、真实行情 L3 环境工厂及生产 StageRunner 装配、真实 WFO/留出统计、双版本回放/rollback 和模拟盘验证仍缺相应证据。SelectionLedger 现可在 worker 执行前持久预留每条 candidate×episode 请求，按 protocol 的 ES budget 跨 attempt 累计，失败 attempt 没有阶段收据时仍计费；成功阶段 receipt 会复核持久预留与 usage 相等。A4 synthetic integration 已接入该回调。PGPE checkpoint 跨 pool 恢复会在继续前核对 SQLite reservation keys；此外，Native Engine 候选已实测在新 spawn pool 中从 checkpoint 恢复，和连续运行的 fitness/evidence 一致，并核对 8 条 lane request reservation 与阶段 receipt。当前覆盖的是正常结束第一组 worker 后重建 pool；worker 被操作系统强杀、评估中断电仍未验证。E1/E2 已由可安装阶段执行器接入 L5 Provider、L6 lane 和 SelectionLedger；正式生产仍需 R1 训练/转移配置、真实行情 L3 环境工厂及相应运行签核。有效独立 trial 数/DSR 政策仍等待 ICR-L6-008。EvoTorch 0.6.1 CPU 原生恢复现有候选序列等价性证据；Native Engine 恢复和 A4 均使用合成行情；`real_golden` 与 `perf` 门禁需要固定外部行情证据/资源预算后再实现，不以合成测试关闭这些门禁。

## R1→E2 实际策略修复（本次更新）

A4 已将手工 R1 回调换成可安装的 `RLStageExecutor`。它使用独立 L4 `RLProvider` 与现有 L5 transfer 接口，训练后导出 L0 `InferenceActor`。E2 Native worker 只接受该 L0 工件和 R1 完成标识，执行 L4 原生动作变换与冻结归一化；这修复了先前 PPO 训练权重在 E2 被 L5 `tanh` Actor 解释而导致的策略漂移。旧 E2 canonical 配置在 `ESStageExecutor` 预检阶段失败；缺失或损坏的 L0 工件也在预留 L6 候选额度前失败。RL 和 ES 包仍可分别使用。

当前验证：A4 定向测试 1 passed；全仓 CPU 209 passed、CUDA 4 skipped；MyPy 89 个源码文件通过；Ruff `packages tests` 通过；`uv lock --offline`、`uv pip check` 与 orchestrator wheel 离线构建通过。wheel 包含 `rl_stage.py`。NautilusTrader 上游 Git 工作树的已跟踪文件无未提交或暂存修改；存在一个未跟踪的 `python/nautilus_trader/nautilus_pyo3.pdb` 构建调试文件，不属于源码改动。

这些结果只覆盖确定性合成行情的 Native Engine 闭环。上游仓库自带行情测试文件不等于经项目验收的 ETHUSDT 永续数据；真实数据来源、历史 Mark/Index/Funding/规则和交易所对照金标、生产 L3 工厂、WFO/留出统计、GPU/双版本恢复仍未验收。

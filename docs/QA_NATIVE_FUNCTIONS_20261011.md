# L3 原生交易与独立 RL / ES 功能集成

日期：2026-10-11。以功能联调为主；没有修改 `nautilus_trader/` 上游源码。RL、ES 保持独立运行包，通过 L3 环境与适配器使用锁定的官方 NautilusTrader。

## 已实现

- `NativePlanSubmitter`：将冻结的 L1 ExecutionPlan 转为原生 Strategy 的 Market/GTC 订单与取消命令；校验标的、数量单位/步长、执行规格、延迟身份和动作幂等。终止平仓使用 reduce-only，并在提交前核对原生持仓的精确反向数量。
- `PublicBacktestBuilder`：绑定实际注册的 Strategy 和提交器；每个 episode 使用全新的 builder、原生执行模型和 engine，禁止跨折复用有状态模型。
- `NativeFactsReader`：修复 NETTING 持仓 ID 复用后丢失旧成交的问题。成交读取原生订单事件历史；资金费读取当前持仓及原生归档，按事件 ID 去重并拒绝矛盾事实。
- `BarrierEpisode`：装配 L2 可见特征、风险→计划回调、原生回放、已成交敞口、固定观测、决策时钟与终止结算。保留两个额外数据帧用于 cancel/ACK 和退出；取消确认后重新规划平仓，避免使用可能已变化的旧持仓数量。
- `EpisodeReplay`：记录每个活跃 episode 的动作序列，重建新的原生会话后回放。恢复奖励峰值、回撤、前次干预及已经消耗的步数预算，防止恢复时重新获得完整 max_steps。
- Gym 环境现在可在首次 reset 前提供冻结的 transition identity，满足 SB3 初始化顺序；reset 后继续核对实际 episode。
- Planner 的幂等名称增加 `action-` 前缀，修复 SHA-256 以数字开头时不能构造 L1 ExecutionPlan 的运行错误。这改变了草案格式的计划身份，旧动作日志不能混用。
- L3 SoftRiskParams 增加中立 multiplier 映射，接收 E2 的 exposure/drawdown/volatility 三个缩减倍率，不接收 HardRiskConfig。修正原生敞口因价格变动越过动作区间时无法执行风险缩减的问题。

## 实际功能验证

`tests/test_nautilus_adapter_smoke.py`：22 项真实 native API / 合成行情检查，包括多空开平仓、两笔原生手续费、取消 ACK、带挂单的终止关闭、错误平仓数量拒绝、成交历史跨两次 round trip 保留、L1 MarketEvent→Nautilus Bar/Mark/Index/Funding 解码，以及 ETHUSDT 永续多空实际资金费结算。正 funding rate 下多仓扣款、空仓收款；结算时资金费只入账一次，NativeAccountFacts 的权益/持仓调整与入场保证金经实际引擎验证。另检查 DQ-D、标的漂移和同结算时刻费率矛盾均被拒绝。该金标采用合成事件，不替代真实交易所数据核对。

`tests/test_native_training_episode.py`：22 项功能检查，使用真实 L2 FeatureRow、L3 Soft/HardRisk、数量 planner、Gym 环境及 Nautilus 账户：

- PPO、SAC、TD3、DDPG 各训练 4 步。
- PPO、SAC 的 4 步保存 → 重建原生环境 → 续训 2 步，与连续 6 步的模型权重逐项相等。
- 日志恢复后的原生账户、观测、奖励、终止时刻及剩余步数预算一致。
- PGPE 与 CMA-ES 各运行 1 代、4 个候选，fitness 实际经过原生交易环境和终止结算。
- E1/PGPE 与 E2/CMA-ES 另各通过真实 L6 spawn `ESEvalLane` 执行 4 个候选；每个候选在 worker 内新建完整 L3 Gym episode 和 Nautilus Engine，经 `rles_context` 校验 Train-inner、Actor/SoftRisk schema、R1 Actor 身份与账号请求，最终清算后回传受请求/账号身份绑定的 evidence。该项使用本地合成市场事件，证明真实原生 Engine 路径可在 L6 worker 生命周期内执行，不证明外部真实数据或生产应用环境工厂装配。
- ES 批次从 episode native equity、未整形 log-return、drawdown 和 risk-guard intervention 汇总 `FitnessSpec` 分数；tail fraction 明确传入并加入 dispatcher 配方签名，避免将 shaped Gym reward 当成经济 fitness。
- Actor 推理、E2 软参数映射及市场漂移后的风险缩减实际可执行。
- 两个独立原生 Engine 进程进行 PPO 训练并启用 VecNormalize；一个 episode 结束时只重置该 worker，另一个保持当前 episode 连续推进，归一化运行统计正常更新。

`tests/test_spawn_sb3_vec_env.py`：2 项进程 VecEnv 边界检查，覆盖 selective autoreset、终止观测保留、Gym Wrapper 属性查询和批次超时后整组 worker 回收。多环境 SB3 接口现由 L4 的 `BoundedSubprocessVecEnv` 提供，worker 使用 spawn 隔离每个 Native Engine；startup 与决策 barrier 分别设置超时，超时/worker 异常会废弃整个同步尝试。只允许有限的基础值、NumPy 数组与 Gym spaces 跨边界传输，Native 对象不会作为查询返回或调用参数穿过 RPC。

L6 Research/WFO 继续集成验证：新增 `tests/test_research_wfo.py`，使用 L2 实际 `build_folds()` 产物配对验证 L6 请求与嵌套 `FoldManifest`，并检查 dataclass/asdict 输入、边界漂移、schema、warmup、hash、隔离标志、错误时间类型及请求结构。校验器覆盖当前 L2 `0.4.0` 源码结构，不替代 L2 DQ gate 或正式 ICR 签核。

L5 ES 状态恢复验证：新增 `tests/test_es_checkpoint_resume.py`，在 EvoTorch `0.6.1` CPU 下分别中断 PGPE 和 CMA-ES，原子检查点保存后创建全新 Provider 恢复；后续候选序列和最终结果与连续运行逐项相等。另覆盖数据/算法指纹漂移拒绝与失败后禁止继续。codec 不使用 pickle，CUDA 恢复仍未支持。

L5/L6 ES 及 Hybrid 集成回归：E1 ActorGenome 与 E2 SoftRisk 候选分别通过真实 spawn/IPC `ESEvalLane` 执行 candidate×episode 矩阵；专门集成测试已进一步在 L6 ES worker 内运行真实 Nautilus Engine 和 L3 Gym episode。E2 actor/schema 漂移在派发前失败；`EvaluationBudget` 按逻辑请求 identity 防止篡改 genome 后更换 request hash 绕过一次性计费。`tests/test_orchestrator_workflow.py` 覆盖 A0–A4 全部路由、usage/lineage receipt、SQLite 持久化 SelectionLedger 回调，以及 E2 修改冻结 R1 actor 时失败关闭。该 SelectionLedger 联测中的阶段 Runner 仍是合成回调，并非生产 provider/payload 工厂。研究市场仍是确定性合成数据，正式数据和生产用量计费未完成。

L6 lane request accounting：ES StageBudget 新增独立 `max_lane_requests` cap，StageUsage 新增 `lane_requests` 实际计数；ProviderBackedStageRunner 将其与 `EvolutionResult.evaluations` 分开校验，SelectionLedger 将 cap 与 usage 一并写入 receipt。ProviderStageOutput 要求非零 lane 用量与合法 `lane_evidence_hash` 同时存在，禁止 RL 输出携带 ES lane 计量。`EvaluationBudget` 新增预留回调；SelectionLedger 在 worker 执行前将每个 candidate×episode request 写入 SQLite，并在研究协议级 ES 预算中累计，失败 worker 或换 attempt 不会重置额度。成功阶段的收据要求 usage 与预留记录数量一致。L5 evaluator checkpoint schema v3 将 reservation scope 纳入签名并写入预算 checkpoint；恢复时缺少 durable callback、协议/运行/尝试/阶段 scope 不同，或 SQLite 中的 request-key 集合与 checkpoint 不完全匹配，都会在继续评估前拒绝。A4 原生 Engine/spawn lane 测试由冻结 StageContext cap 创建带持久预留和恢复核对回调的 `EvaluationBudget`，每个候选在两个 episodes 执行，并断言 E1/E2 收据各记录 4 个 candidate objective evaluation 与 8 个 candidate×episode lane request；另有回归验证失败 attempt 无收据仍会耗尽协议预算、重试被拒绝。PGPE checkpoint 已覆盖跨进程池恢复：首个真实 spawn pool 完成一代并退出，新 pool 恢复后与连续运行的候选结果一致；账本缺失时会在续跑前拒绝。新增真实 Native Engine 恢复集成：每个候选在 L6 spawn worker 中运行完整 L3 Gym episode 与 Nautilus Engine，首个 pool 完成一代后关闭，新的 pool 从 checkpoint 继续第二代；恢复结果、lane evidence hash 与连续两代运行一致，SQLite 中 8 条 candidate×episode 预留与最终阶段收据逐项对账。另有真实 spawn worker 执行失败回归，确认 worker 进程退出、整批预留留在 SQLite、quota 释放，超预算的新 attempt 在派发前被拒绝。当前覆盖的是有序关闭 pool 后恢复；操作系统强杀进程、评估途中断电，以及真实行情生产 worker 的恢复仍未验收。

本轮 CPU 全套：**209 passed / 4 CUDA-only skipped**；项目解释器 MyPy：**89 source files，0 errors**；Ruff `packages tests` 全通过。L5 E1 派发现分别绑定 ActorGenome 搜索 schema 和下游 R1 Actor signature，隔离安装 wheel 回归通过；batch-only Provider entry point 完成一代 PGPE、4 个候选评估。SAC、TD3、DDPG 的 SB3 参数经 L5 Actor Genome 编解码、L4 destination mapping 检查和 L5 DIRECT parity 回归通过；PPO clip/tanh mismatch 触发 L4 typed rejection 与 L5 Train-inner 蒸馏回退，MSE/hash/parity 复核通过。L5 用量 adapter 已对真实 L4 `Performance`（含恢复后多次 train 调用）及 EvoTorch `EvolutionResult` 做 Native Engine 测试；A4 端到端联测现经真实 L6 spawn ESEvalLane 执行 E1 PGPE 与 E2 CMA-ES 候选，在 R1 PPO 蒸馏/训练后持久化全部阶段收据。另验证 Native Engine ES 在新 spawn pool 中 checkpoint 恢复，候选结果与连续运行一致，L6 request reservations 与阶段收据对账一致。该 A4 和恢复测试使用确定性合成行情、测试授权和可安装 worker 适配器和测试用 L3 环境工厂；还需接入生产数据、配置及经签核的运行工厂。所有行情均为合成数据，不表示真实市场策略、保证金方案或交易所费率规格已经验收。

## 安装级 Native ES worker 与派发修复

新增 `rles_orchestrator.native_es_worker:make_native_es_session`，作为可安装的 L6 ES worker 工厂。应用注入受审查的 L3 `NautilusBarrierEnv` 工厂及冻结数据/执行配置；worker 逐候选核对 L5 request、Actor/SoftRisk schema、观察列顺序、E2 冻结 R1 Actor 权重身份、transition 身份和原生终止结算，再从 Native Engine 账户事实回传未整形经济收益、回撤、干预量及 evidence。E1/PGPE 与 E2/CMA-ES 均在真实 L6 spawn 进程中各运行一代、4 个候选，且伪造请求在建 Engine 前被拒绝。当前注入的是合成行情环境工厂；生产数据和应用工厂仍待接入。

`ESEvalLane.runtime_signature` 现在包含 worker 工厂引用、配置、worker 数、Torch 线程数和候选超时；L5 `WorkerLaneDispatcher` 将其并入 checkpoint 配方，恢复时更换 worker 配置会在派发前拒绝。L6 不再用跨进程 `Queue.empty()` 与进程状态组合判断已完成候选是否丢失，而是读取队列并以候选 deadline 为失败边界。对应 spawn 回归验证了状态观察先于队列结果到达时仍能收到结果。

新增 `ESStageExecutor`，把 StageContext 冻结预算、EvoTorch E1/E2 Provider、L6 spawn lane、SelectionLedger 预留和 `ProviderStageOutput` 装入可复用的阶段执行器。A4 合成行情联测现由该执行器实际运行 E1 和 E2，R1 仍调用独立 L4 Provider，三个阶段通过 Orchestrator 持久化收据。执行器每完成一代便保存到独立的不可变 checkpoint 目录；真实 Native Engine 的一代中断后，新的执行器从旧 checkpoint 接续第二代并完成阶段收据，结果及 evidence 与连续运行一致。应用仍须提供真实市场数据、L3 环境工厂以及 R1 生产训练/转移配置。

## R1→E2 策略一致性修复

`RLStageExecutor` 现在负责 A4 的实际 L4 训练、E1→R1 transfer、L0 推理工件导出与动作一致性检查。E2 的安装级 Native worker 只接收 R1 导出的 L4 `InferenceActor` 工件及其完成标识；它按 L4 原生输出变换和冻结归一化推理，避免将 PPO 的 `clip` 动作头误作 L5 canonical `tanh` Actor。旧 canonical E2 配置在阶段预检被拒绝；缺失或损坏的 L0 工件也在预留 L6 候选额度前被拒绝。R1/E2 仍通过独立的 L4、L5 公共接口组合，没有修改 NautilusTrader 上游源码。

验证：`uv run --locked --no-sync pytest -q tests/test_native_training_episode.py -k a4_provider_backed --disable-warnings --tb=short --show-capture=no` 为 1 passed；全套 `pytest -q --disable-warnings --tb=short --show-capture=no` 为 209 passed / 4 CUDA-only skipped；89 个源码文件 MyPy 与 `ruff check packages tests` 通过。`uv build --offline packages/orchestrator --out-dir .artifacts/build-r1-final` 成功，wheel 含 `rl_stage.py`。所用行情和 L3 工厂仍为确定性合成测试样本；实际 ETHUSDT 永续数据、历史费率/保证金和外部签核仍阻断正式训练及历史回测验收。

## 装配与恢复约定

完整可运行装配示例在 `tests/test_native_training_episode.py` 的 `_episode()` / `_env()`。生产装配应替换该示例的固定价格、合成事件、来源身份和测试授权，提供实际 L2 数据、历史规则及 native mark/funding 能力；风险和 planner 仍通过回调注入，不放进 RL/ES 模块。

每个 episode 构造新的 NativePlanSubmitter、PublicBacktestBuilder、EngineSession 和 BarrierEpisode。L4 stage.v3 checkpoint 保存逐环境 seed schedule；`RLProvider.load` 显式 reset 一次，由 env factory 注入并消费对应的 L3 `EpisodeReplay` 以重建已保存边界，然后核对 L3 token/观测。恢复 replay 只应用于该次加载 reset，之后 episode 正常重新开始。外部装配仍须按 worker 保存和供应各自的 `EpisodeReplay` 及冻结数据/执行配置。

已复现 rc6 同一进程中两个活跃 engine 的注册表/消息路由串扰。因此 EngineSession 明确限制每进程一个活跃实例。L4 多环境训练通过 spawn 为每个环境建立独立进程，并经 L3 同步 barrier 汇总；没有通过修改上游源码规避此问题。

还需继续实现/验证：生产级 R1 装配及真实数据驱动的 E1→R1→E2 工作流、真实数据的 mark/funding/margin replay 对照与 WFO；本轮已提供完整合成 A4 functional workflow，并提供可安装的 Native ES worker 适配器，但仍未连接生产 L3 环境工厂或真实研究数据。EvoTorch 0.6.1 CPU native searcher resume 已通过分代候选序列 parity，CUDA codec 和外部 evaluator 非零数据游标尚未实现。L4 已把 Python/NumPy/Torch RNG 及每环境 episode seed schedule 放入 stage.v3；应用层仍须正确供应逐 worker 的 L3 replay 与数据游标。当前权重连续性和 funding 结算证据来自确定性合成 episode，不替代真实数据端到端恢复验收。

之前 `qa/reports/engineering_20261011.json` 是上一轮干净安装和 wheel 测试快照，不包含本轮功能改动；本轮没有重复生成全包哈希或发布门禁材料。

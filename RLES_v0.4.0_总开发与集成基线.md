# Nautilus RLES v0.4.0 总开发与集成基线

更新日期：2026-10-11。依据《独立 RL-ES 优化设计方案 v0.4.0》和 R1 并行开发工作包整理。本文件是集成控制清单；完成状态只依据本机代码和验证证据填写。

## 目标与边界

建立独立于 NautilusTrader 上游源码的 ETHUSDT USDT-M 永续合约研究系统。以中立 Contracts 串联数据、Nautilus 引擎事实、执行、风控、Gym、RL、ES、计算和研究层。研究发布需 Q0–Q9 全部通过；Q10 是额外的模拟盘准备门禁。工程验收不等于盈利证明，也不授权真实资金交易。

用户本次委托我承担各部分功能汇总和总开发集成。R1 工作包原先把最终集成分配给用户；这里按本次直接委托执行集成工作，保留包边界、接口变更记录和用户最终发布签核。

## 十一个包的功能汇总

| 包 | 开发线 | 应交付的生产功能 | 输入 / 输出与边界 |
| --- | --- | --- | --- |
| `rles-contracts` | L1 | 不可变 DTO、Protocol、版本化 schema、规范序列化、配置与运行指纹、SeedTree、插件发现及结构化错误 | 所有包共享；不导入 Nautilus、SB3 或 EvoTorch，不包含交易计算 |
| `rles-data` | L2 | RAW/CURATED/FEATURE/FOLD 数据链、可用时间、质量分级、特征规格、只用训练段拟合的 scaler、FoldManifest | 向环境提供决策时刻可见事件；不创建引擎，不读取未来数据 |
| `rles-nautilus-adapter` | L3 | 用官方公开 API 创建每折独立 EngineSession，推进事件屏障并产出订单、持仓、权益等事实快照 | 唯一直接导入 Nautilus 的包；不复制或修改上游源码 |
| `rles-execution` | L3 | 目标净敞口到订单的唯一映射、精度与最小名义值、在途单、延迟、滑点、终止平仓规则 | 生成 ExecutionPlan；撮合和账户事实仍由 adapter/引擎承担 |
| `rles-risk` | L3 | 硬风险限额与软风险变换；记录原始、软调整、硬限制后三层目标 | E2 只优化软参数；硬限额只能缩小或拒绝暴露 |
| `rles-gym-bridge` | L3 | 单环境 reset/step、观测与奖励、终止结算、同步向量环境屏障 | 一步为一个决策屏障区间；不自建第二套交易账本 |
| `rles-rl-sb3` | L4 | PPO/SAC/TD3/DDPG 的原生训练、评估、保存恢复、回放指纹、Actor 迁移目标 | 使用冻结 Gym 契约；不改 SB3 更新公式或环境语义 |
| `rles-es-evotorch` | L5 | CanonicalActor/Genome、PGPE、CMA-ES、候选评估、状态恢复 | E1 进化 Actor；E2 只搜索 SoftRiskParams；不得访问引擎私有状态 |
| `rles-compute-runtime` | L6 | CPU Worker、同步 RL lane、候选级异步 ES lane、GPU 独占、资源配额、故障处理与端到端基准 | 实现 Contracts 的计算接口；设备改变必须产生可追踪的新 attempt |
| `rles-orchestrator` | L5 | 插件装配、A0–A4 路线、E1→R1→E2 单次状态机、迁移/蒸馏与血缘 | 只编排 Provider；不实现算法更新和市场事实 |
| `rles-research` | L6 | WFO、B0–B3 基准、指标与统计、选择账本、最终留出锁、工件完整性与模型/数据卡 | 查询预算先冻结；测试和最终留出不回流训练或调参 |

## 跨包主链路

`rles-data` 用 `MarketEvent.available_at_ns <= decision_ts_ns` 过滤可见事件 → `EngineSession` 在统一事件屏障推进 → Gym 形成固定 schema 的 Observation → 策略输出 PolicyAction → SoftRiskTransform → HardRiskGuard → ExecutionPlanner → adapter/官方引擎执行 → ExecutionSnapshot → economic/shaping reward 分离记录。每次 step 都必须留有输入、动作三层目标、计划、订单/成交/费用/资金费、账户和奖励的关联标识。

RL 的多个环境按同一 batch barrier 前进，超时即终止 attempt；ES 只在独立候选评估粒度异步。迁移必须验证四种算法各自 Actor 的参数映射和固定观测动作一致性；不通过时只能明确进入蒸馏路线。CPU/GPU/Hybrid 比较必须固定样本数、更新预算、随机种子和环境配置。

## 集成顺序与放行条件

| 阶段 | 集成内容 | 放行证据 |
| --- | --- | --- |
| G0 / P0 | Python、Nautilus、SB3、PyTorch、EvoTorch、CPU/CUDA 锁与许可证；Contracts 草案 | Q0：真实安装、导入、最小训练、保存/加载和依赖清单；仅有 wheel 不算通过 |
| G1 / P1 | Contracts → Data；核对可用时间、质量、折边界、scaler | Q1：未来扰动不改变当前观测，warm-up/purge/embargo 和终止边界可复现 |
| G2 / P1 | Adapter → Execution/Risk → Gym | Q2/Q3：事件、费用、资金费、在途单、账户、终止平仓金标；single/VecEnv 轨迹一致 |
| G3 / P2 | 四个 SB3 算法、PGPE/CMA-ES、Compute lanes | Q4/Q7：独立训练与恢复；同步 RL、异步 ES 和故障路径不改变采样语义 |
| G4 / P3 | 四算法迁移、E1→R1→E2、A0–A4 | Q5：参数覆盖、动作一致性、阶段冻结、预算与 lineage |
| G5 / P3 | WFO、B0–B3、统计、选择账本、原子工件库 | Q6/Q8：查询预算和最终留出锁、统计报告、L0/L1 恢复与缓存完整性 |
| G6 / P4 | 双版本 adapter 回放、依赖升级回归、回滚、发布组装 | Q9：差异归因、批准记录、回滚演练；Q0–Q9 全部通过后才可声明研究系统完成 |

任何 Q1/Q2/Q3 未通过时，不启动市场数据上的正式 RL/ES 训练。最终留出只在 recipe 冻结并用全部 pre-holdout 数据重训后打开一次；失败也是有效结论。

## 每份模块交付的集成检查

1. 核对所属包、Task ID、输入 Contracts 版本、前置 Gate、文件清单及已知限制。
2. 对 DTO、schema、配置和 CLI 的变化登记 ICR；先合并 Contracts，再合并生产者，最后合并消费者。未经确认的破坏性接口不进入集成基线。
3. 只把模块生产源码及功能说明纳入模块交付；集中 QA 另存测试、fixture、golden、日志和运行证据。R1 文档同时要求 Gate 证据且禁止把证据放进模块交付，本清单以“模块交付”和“集中 QA 证据”分开执行。
4. 审查失败路径：未知配置、缺 mark/funding/rules、schema 不一致、Worker 超时、CUDA OOM、未完成工件均须结构化失败，不静默回退。
5. 在统一集成环境运行消费者契约检查、最小相关测试和跨包固定动作回放；把实际命令、版本、哈希和结果写入集中 Gate 记录。未运行的 Gate 标为 BLOCKED/未验证，不写 PASS。

## 当前工作区盘点

- `nautilus_trader/` 是上游 Git 仓库，当前分支 `develop`；本轮未改它的源码。发现既有未跟踪文件 `python/nautilus_trader/nautilus_pyo3.pdb`，保持原状。
- 已合入 L1 R2 与 L2–L6 的生产源码；11/11 个功能包现位于 `packages/` 标准 workspace 目录。跨线 ICR 与功能验收尚未关闭，不能宣称任何 Q Gate 通过。
- 上游 `nautilus_trader/python/.venv` 仍为 Python **3.14.6**；已另建 RLES 根环境 `.venv`，使用隔离 CPython **3.14.7**，详见本目录 `docs/QA_G0_CPU_20261011.md`。
- 上游仓库 `version.json` 标识 **v2.0.0rc7**，方案写的冻结候选是 **rc6**。需在独立 RLES 环境明确选择并锁定版本；版本变更要走 Q9 双版本回放，不直接把当前源码当作 rc6 wheel。
- 本机可见 NVIDIA GeForce RTX 5070 Laptop GPU、驱动 582.05、显存约 8 GiB。CUDA 可见不代表 PyTorch/EvoTorch/SB3 的训练与恢复验证已通过。

## L1 R2 接入记录

- 交付来源：`01_L1_Contracts_Runtime_v0.4.0_DeepReview_R2_L1_ONLY.zip`。23 个归档条目路径安全、无重复、无目标覆盖冲突；交付范围未发现 tests、测试数据、Nautilus 源码、wheel 或运行结果。
- 合入位置：本文件所在的 RLES 工作区根目录。`nautilus_trader/` 上游 Git 仓库未作为目标目录，也未被修改。
- 代码范围：中立 DTO/Protocol、严格 canonical JSON 与可选 Arrow IPC、配置/Schema、provider registry、运行指纹与 SeedTree、结构化错误、风险配置 DTO，以及只读工作区依赖校验器和离线依赖/许可证审计工具。
- 版本状态按交付说明保留为 `rles-contracts 0.4.0 Draft`；`.python-version` 指向 3.14.7，但当前已有的 `nautilus_trader/python/.venv` 是 3.14.6，不能代替独立 G0 环境。
- 未运行训练、安装或测试，也未生成 CPU/CUDA 锁、SBOM、证据包；L1 自述 `PARTIAL / Q0 BLOCKED` 和 ICR-L1-001–007 仍有效。合入代码不等于代码审查完成或 Q0 PASS。

## L2–L5 接入记录

- 交付来源为用户提供的 L2 DataTruth v0.4.1、L3 AI-03 dev2、L4 SB3 v0.4.1、L5 ES/Hybrid dev2 四个 ZIP。共 56 个原始文件按工作区路径安全检查后合入；压缩包内未发现测试目录、测试数据、wheel 或 NautilusTrader 源码。
- L2 源码由归档根目录 `rles_data/` 移至 `packages/data/src/rles_data/`；归档未提供包构建元数据，故添加 `packages/data/pyproject.toml`（rles-data 0.4.1、Python 3.14.7、pyarrow 作为可选 arrow extra）。暂未声明对 `rles-contracts` 的依赖，因为 L2 的 `ICR-L2-001` 尚未冻结 DTO/hash 对接。
- L3 四包进入 `packages/nautilus_adapter`、`execution`、`risk`、`gym_bridge`；L4 放入 `packages/rl_sb3`；L5 放入 `packages/es_evotorch` 与 `orchestrator`。四份功能说明均保留在 `docs/` 下。
- L2–L5 包目录都有 `pyproject.toml`；L6 交付补上了 `compute_runtime` 和 `research`，目前 11/11 个功能包都有构建元数据。`ci/l6` 是独立发布工具，不属于 11 个运行功能包。
- 接口风险保持待办：L2 仍使用自身事件/清单模型；L3 适配器目标为 NautilusTrader rc6，而当前上游仓库标识 rc7；L4/L5 分别列有与 L1/L3/L5/L6 的待签 ICR；L5 明确标记 PARTIAL，ES 原生恢复未获 G0 验证。
- 本次只完成代码归位和工作区集成，不运行安装、训练或测试；L2 Q1、L3 Q2/Q3、L4 Q4-RL、L5 Q4-ES/Q5 均不能据此标记 PASS。

## L6 接入记录

- 交付来源：`Nautilus-RLES_L6_AI06_Production_Source_v0.4.1.zip`。32 个归档文件通过路径安全、重复项、覆盖冲突和交付白名单检查后并入；未发现 tests、测试数据、wheel 或上游 NautilusTrader 源码。
- 两个功能包进入 `packages/compute_runtime` 和 `packages/research`；Gate/attestation、upgrade、security、release 工具进入 `ci/l6`；功能说明保留于 `docs/L6_功能与接口说明.md`。
- 可复用 CI workflow 已按本次委托放入 RLES 根项目 `.github/workflows/l6-ci-reusable.yml`，只响应 `workflow_call`；未改动 `nautilus_trader/.github`。
- 来源校验：32 个归档文件中 31 个与交付源 SHA-256 完全一致；唯一差异是本说明文档的两处 workflow 路径/集成状态文字，已随目标布局校正。
- L6 文档标记 Q0/G0 及跨线任务 BLOCKED，ICR-L6-001–008 待签。L6 的独立 DTO/mapping 尚未与 L1–L5 冻结契约核实；实际 adapter 双版本回放、资源/故障验证、研究统计与发布证据仍待集中验收。
- 未运行安装、测试、训练或 CI；本次为源码归位和工作区集成，不代表 Q0–Q9 任何 Gate 通过。

## L1–L6 源码复核与 G0 CPU 实测

- 再次逐项对照六份来源 ZIP：共 111 个文件，全部在项目目标位置；110 个 SHA-256 与交付源一致，唯一差异为 L6 功能说明中的两处 workflow 路径文字已按实际集成位置调整。没有需要补合并的源码。
- 建立 CPython 3.14.7 项目隔离环境并安装 11 个功能包及 Arrow extras；`uv.lock`、包导入、compileall、uv 包构建、依赖一致性检查及 CPU 版 SB3 四算法合成环境短训练/保存/加载均有记录。
- 已在 `pyproject.toml` 增加 workspace source 映射，修复首次 `uv sync` 暴露的 Contracts 本地依赖解析缺口。
- 已修复审计发现的 Provider descriptor 兼容、workspace 必需依赖边、一维 CMA-ES 边界；增加 L2→L1 MarketEvent 显式映射、逐标的缺口与资金/规则审计、不可变审计元数据、L3/L4 动态 transition/checkpoint 边界、L4 SB3 Provider 与 L6 统计输入缺陷修复。CPU 最新 60 项通过，PR 57 项通过，CUDA 专项 4 项通过；四算法阶段恢复与连续 CPU 训练权重一致；ruff、依赖一致性和 11 个 workspace 包构建也通过。真实 Nautilus/L3 环境证据仍未关闭，详见 `docs/QA_G0_CPU_20261011.md`。
- Q0 仍为 BLOCKED：还没有干净环境锁文件复现、Linux CUDA runner 实测、wheelhouse、SBOM/NOTICE、许可证审查、Nautilus ABI/Engine 集成与签署证据。Q1–Q9 仍缺真实市场数据、原生金标、真实环境/恢复、研究统计、GPU/Hybrid 故障、工件恢复、双版本回放与 rollback drill 证据。
- 当前完成的是 L1–L6 集成缺陷修复与基础工程验证，不等于设计方案的研究验收完成或模拟盘准备；完整门禁状态见 `docs/QA_G0_CPU_20261011.md`。
- 后续功能修复已接通 L2 FeatureRow/每折 Scaler 到 L3 Gym 观测的显式适配，并统一 L3 planner 与 L1 `ExecutionPlan` 的订单/取消/ACK 字段；Nautilus rc6 官方 Engine 的两 Bar 流式回放与原生账户权益读取通过。当前 CPU 全套 95 passed / 4 CUDA-only skipped，详见同一 QA 记录。RL、ES 仍为独立包，`nautilus_trader/` 上游源码未修改。

## 立即执行队列

1. 审查 L2–L6 的待决 ICR，先冻结 L1→L2/L3 DTO、L3→L4 Gym、L4/L5→L6 lane 与研究证据映射。
2. 保持独立 RLES 项目在本工作区根目录，并以 Nautilus 官方 wheel 作为外部依赖；上游源码只用于阅读公开 API 与升级比较。
3. 先处理 G0 版本矩阵与 Q0，再冻结 L1/L2/L3 消费契约；Q3 前不开展正式算法市场训练。
4. 每次接入一个任务后更新本文件的状态，并在集中 QA 记录 Gate 证据与 ICR；发布声明始终以实际通过的最高能力等级为准。

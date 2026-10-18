# Nautilus-RLES L1–L6 当前状态与全面审计报告

- 审计日期：2026-10-11（Asia/Shanghai）。
- 审计对象：`D:\Program Files\NautilusTrader_RL_ES` 当前源码、现有 CPU/CUDA 环境、测试、交付归档、安装工件与设计要求。
- 需求基线：用户提供的《独立 RL-ES 优化设计方案 v0.4.0》及各层功能说明。附件中的开发指令作为待核对的项目材料，不作为本轮额外执行授权。
- 本轮执行：只做审计、验证和报告；未修复生产代码、未发布、未执行正式研究训练。

## 1. 总体结论

**当前状态：L1–L6 交付源码已合并，核心组件和使用合成行情的原生交易引擎链路已打通；项目仍处于研发集成阶段，尚未完成设计方案的完整研究与发布验收。**

本次重新检查没有发现六个交付包的文件漏合并。当前源码已能通过独立 RL、独立 ES 和 A4（E1→R1→E2）的功能测试，实际调用 NautilusTrader 原生引擎、SB3、EvoTorch 和跨进程评估池。这些进展有测试证据，不能简单归为“只有接口”。

但正式 ETHUSDT 永续数据驱动的应用装配、完整观测、执行模型绑定、WFO/Validation/Test/Holdout 流程、完整交易工件、算力基准与全包发布仍不闭环。本次还复现了 DQ 能力授权过宽、ObservationSchema 可变、归一化迁移输入域混用三项问题。因此，**现阶段不能宣布“全部设计完成”，也不能把现有测试通过等同于真实历史回测验收或策略有效性证明。**

### 1.1 本次实际结果

| 检查 | 结果 | 结论边界 |
|---|---|---|
| 六个 L1–L6 原始交付包逐文件存在性 | 111/111 文件存在，0 缺失 | 证明已导入；不证明后续修改全部正确 |
| CPU 全套 pytest | 209 passed，4 skipped，24 warnings；43.91 秒 | 共 213 个用例；CUDA 用例在 CPU 环境跳过 |
| 独立 CUDA 环境 SB3 冒烟 | 4 passed，2 warnings；6.50 秒 | 四算法 Pendulum 训练/保存/加载/预测，非原生交易集成 |
| MyPy | 89 个源码文件，0 errors | packages 范围；部分未标注函数体未检查 |
| Ruff：packages/ci/qa/tests | 通过 | 配置只启用 E4、E9、F |
| CPU 环境依赖一致性 | 68 个安装包，无依赖冲突 | 非最新版全包干净 wheel 安装验收 |
| real_golden 或 perf 收集 | 0 个，213 deselected，非零退出 | 当前 nightly 的必备测试门禁不具备通过条件 |
| short_wfo/artifact_restore/dependency_resolver 收集 | 6 个 | marker 名称不能证明完整 WFO 已执行 |
| 上游 Git 跟踪文件 | 工作区与暂存区 diff 均为空 | 有一个未跟踪构建 PDB，详见第 3 节 |
| 架构直接导入扫描 | 0 项违规 | 只验证静态直接导入规则 |
| 定向功能探针 | 复现 3 项缺陷 | 详见 F01–F03，结果已另存 JSON |

### 1.2 审计问题总览

本报告列出 **15 项**后续事项：**4 项 P0、10 项 P1、1 项 P2**。

- P0：阻断设计要求的正式研究/整体验收，须先补齐或修复；不表示已经发生生产事故。
- P1：功能正确性、恢复、安装或运行保障问题，须在正式交付前关闭。
- P2：项目可追溯性与文档治理问题。

## 2. 方法、证据与限制

执行了源码审查、交付 ZIP 路径映射和文件对比、依赖/解释器检查、全套 CPU 回归、独立 GPU 冒烟、静态检查、测试 marker 收集、上游 Git 检查、已有 wheel 与源码比较，以及三项针对性功能探针。

设计文档已从原始 DOCX 文本中读取；审计同时查阅 `docs/QA_NATIVE_FUNCTIONS_20261011.md`、`docs/QA_ENGINEERING_CONTINUATION_20261011.md` 和旧 `qa/reports/engineering_20261011.json`，并区分历史记录与本次重跑结果。

证据等级：

1. **已实测**：本轮实际执行得到的结果。
2. **源码确认**：当前调用链/实现可以直接确认的问题，但未在完整生产场景重放。
3. **未验收**：缺少实现闭环或足够测试材料，不推断其已经成功或必然失败。

本轮未重新进行全部 11 包的干净安装、正式市场数据研究、长时间稳定性/吞吐测试、操作系统强杀或断电恢复、外部漏洞库扫描。两份旧 wheel 在当前审计身份下不可读，详见第 8 节。以下结论明确受这些范围限制。

## 3. 源码合并、架构和上游状态

### 3.1 六份交付包核对

| 交付层 | 原始交付文件数 | 当前存在 | 与原始字节内容不同 |
|---|---:|---:|---:|
| L1 Contracts Runtime R2 | 23 | 23 | 11 |
| L2 DataTruth v0.4.1 | 9 | 9 | 8 |
| L3 AI-03 dev2 | 20 | 20 | 13 |
| L4 RL SB3 v0.4.1 | 11 | 11 | 9 |
| L5 ES Transfer Hybrid dev2 | 16 | 16 | 12 |
| L6 AI06 Production v0.4.1 | 32 | 32 | 22 |
| 合计 | **111** | **111** | **75** |

映射核对包含 L2 顶层源码进入 `packages/data/src/rles_data/`、L4/L5 包目录重定位、L2 功能文档重命名，以及 L6 workflows 进入 `.github/workflows/`。75 个文件发生变化，与累计集成开发相符；文件存在性和字节对比不能替代变更正确性审查。

### 3.2 当前 11 包

| 归属 | 包 | 当前版本 |
|---|---|---|
| L1 | contracts | 0.4.0 |
| L2 | data | 0.4.1 |
| L3 | execution / risk / gym_bridge / nautilus_adapter | 0.4.0.dev2 |
| L4 | rl_sb3 | 0.4.1 |
| L5 | es_evotorch / orchestrator | 0.4.0.dev2 |
| L6 | compute_runtime / research | 0.4.1 |

另有 `ci/l6` 的 CI/发布门禁代码。项目总体版本仍为 0.4.0，各子包混合稳定版与 dev2，不应仅凭版本号推断完成度。

### 3.3 架构边界

AST 直接导入扫描未发现 adapter 之外的包直接导入 `nautilus_trader`；L4 未直接导入 L5，L5 的 ES 包未直接导入 L4。编排包通过公共接口组合二者，符合独立 RL/ES 的基本方向。

NautilusTrader 上游 HEAD：`b82f3e3c801f4d7d0423f11c34fea8806c09e68a`。跟踪文件工作区 diff、暂存 diff 均为空；仅有未跟踪文件 `python/nautilus_trader/nautilus_pyo3.pdb`。因此本次证据支持“上游跟踪源码未被修改”，不支持“上游目录完全无额外文件”。

当前业务环境从 `.venv/Lib/site-packages/nautilus_trader` 导入已安装原生包。根工作区自身没有 Git 仓库，不能提供涵盖全部 RLES 源码的统一 commit/tag 身份。

## 4. 安装与运行环境

### 4.1 CPU 主环境

| 项目 | 本次读取值 |
|---|---|
| Python | 3.14.7，Windows 64 位 |
| NautilusTrader | 2.0.0rc6 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.14.1；此环境无 CUDA runtime |
| EvoTorch | 0.6.1 |
| Gymnasium | 1.3.0 |
| NumPy / PyArrow | 2.5.4 / 25.0.1 |
| pytest / MyPy / Ruff | 9.1.1 / 2.3.0 / 0.17.0 |

锁文件与已安装依赖能够支持本次 CPU 测试，`uv pip check` 无冲突。该检查不证明每个依赖的所有功能、外部兼容性和最新源码的 wheel 安装都已验收。

### 4.2 GPU 环境

`.venv-cuda` 实际存在并可用：Python 3.14.7、PyTorch **2.14.1+cu130**、CUDA runtime 13.0，`torch.cuda.is_available()` 为真；GPU 为 **NVIDIA GeForce RTX 5070 Laptop GPU**。

`tests/test_cuda_smoke.py` 四项通过：PPO/SAC/TD3/DDPG 在 `Pendulum-v1`、`DummyVecEnv` 下各训练 128 步并保存/加载/预测。

准确结论是 **GPU 硬件与 SB3 基础 CUDA 路径可用**。尚未验收 L3 原生环境上的 GPU RLProvider、GPU ES 检查点、GPU Hybrid、OOM 新 attempt 恢复和端到端性能。CPU 测试的 4 项跳过不能解释为电脑没有 GPU。

## 5. L1–L6 分层状态

| 层 | 已具备且有证据的能力 | 主要未完成/问题 | 状态判断 |
|---|---|---|---|
| L1 | 中立 DTO、配置冻结、序列化、schema/hash、种子、provider 注册与跨包契约 | 下游 schema 的冻结不完整；应用/工件仍未把所有契约连成可执行闭环 | 基础契约可用；整体验收未完成 |
| L2 | 事件规范、Parquet/Arrow、raw/curated、可见性、DQ、特征/scaler、fold manifest | DQ 时间覆盖缺口；真实数据与历史规则验收缺失 | 可用于组件集成；能力授权须修复 |
| L3 | 官方引擎启动、动作→订单、取消 ACK、原生账户/手续费/funding、净持仓、风险/planner、Gym barrier、终止清算和 replay | 完整观测不足；执行配置未完整绑定 native 数值模型；生产环境工厂缺失 | 合成原生交易链路可用 |
| L4 | 四算法训练/保存/加载/评估/actor 导出；独立 RL；多进程 VecEnv；归一化基础路径 | R1 迁移与归一化域混用；应用级续训、生产 GPU 装配未完成 | 核心 provider 可用 |
| L5 | 实际 PGPE/CMA-ES；Actor/SoftRisk 搜索；严格迁移/蒸馏；A0–A4 路由；原生 worker；CPU ES 分代恢复 | 完整生产 Hybrid 未验收；GPU checkpoint 不支持；R1 整阶段续跑缺失 | CPU 合成联测可用 |
| L6 | spawn 池、同步 RL/异步 ES lane、资源预留、GPU lease、持久预算、指标/统计函数、WFO 校验、工件/发布门禁 | 可执行 WFO/研究报告、真实金标/性能、OS 资源保障、全包发布链缺失 | 运行与研究基础设施部分完成 |

### 5.1 原生引擎验证的实际深度

现有 `test_nautilus_adapter_smoke.py` 覆盖真实 native API 下的多空开平仓、手续费、取消确认、终止清算、NETTING 往返成交保留、L1 事件解码及永续 funding 结算等。正资金费率下多仓扣款、空仓收款和结算只入账一次已有合成事件测试。

`test_native_training_episode.py` 使用真实 L2 FeatureRow、L3 风险/planner/Gym 和原生账户：四种 SB3 算法各训练 4 步；PPO/SAC 的保存 4 步→续训 2 步与连续 6 步权重相等；PGPE/CMA-ES 各运行一代、4 个候选。多环境 PPO 使用 spawn 隔离多个 Engine，并测试错开终止和 VecNormalize。

已知 rc6 同进程多个活跃 Engine 会串扰，当前 EngineSession 明确限制每进程一个活跃 Engine，多环境通过 spawn 隔离。该限制有测试支撑，仍须在生产入口保持。

### 5.2 Hybrid 与恢复证据

A4 集成用实际 PGPE→PPO 蒸馏/训练→CMA-ES，E1/E2 通过 L6 spawn 运行原生 Engine，R1 导出 L4 推理工件，E2 使用该工件而非错误替换 PPO 输出头。每个候选执行两个 episode，E1/E2 收据各区分 4 次 objective evaluation 与 8 次 lane request；SQLite 预留与收据对账有覆盖。

CPU ES 跨新进程池的分代恢复，候选/evidence 与连续运行一致；失败 worker 的预留留存、quota 释放及超预算重试拒绝也有测试。

边界：行情、授权、L3 工厂仍是合成测试样本；A4 的实际 R1 算法覆盖 PPO，不是四算法与全部 A0–A4 的完整生产组合矩阵。A0–A4 路由单测、四算法单独冒烟与一个真实 provider A4 联测是三种不同证据。原生权重连续恢复覆盖 PPO/SAC，不应扩写为原生四算法全部恢复通过。

## 6. 详细发现与修复要求

### F01 · P0 · L2：DQ-A 没有证明整个研究区间的经济事实覆盖

**证据：已实测 + 源码确认。** `core.py:479` 的 `quality()` 对 funding、mark/index 主要检查每标的是否至少存在事件；规则检查没有证明研究区间内持续有效。`core.py:601` 的 `require_usable()` 据 DQ grade 授权研究能力。

探针构造 1,440 根连续分钟 K 线，但仅有第一分钟的 mark/index/funding update/settlement，规则在第 2 分钟失效。实际输出仍为 `DQ-A`、`findings=[]`，且 `require_usable(report, "liquidation")` 成功。

**影响：** 文件/事件存在会被误当成全时段可结算、可杠杆/清算研究的证据。不能据此断言 L3 实际已使用错误数据交易，但 L2 gate 本身确有漏检。

**修复与验收：** 按市场规则确定预期 funding 结算时间，检查缺失/重复；校验 mark/index 最大陈旧时间与覆盖区间；校验历史 instrument rule 的完整有效期和切换边界；将明确能力证据绑定 DQ report；覆盖上述反例，并确认不合格数据无法授权 liquidation/settlement/leveraged。

### F02 · P1 · L3/L1 契约边界：ObservationSchema 实际可变

**证据：已实测。** `observation.py:13–31` 使用 frozen dataclass，但未把 `names/units` 转为 tuple 或拒绝可变列表。传入列表合法，外部修改 `units[0]` 后，已构造对象的 schema hash 随之变化。

**影响：** schema/hash 在对象存活期间不稳定，可能破坏观测契约、缓存、检查点与工件身份一致性。这是 L3 schema 实现问题，不能据此断言所有 L1 DTO 都可变。

**修复与验收：** 边界拷贝为不可变 tuple 或严格拒绝 list；逐项检查非空字符串；验证调用者修改原始输入不能改变 schema/hash。

### F03 · P1 · L4/L5 R1：raw observation 与 policy observation 混用

**证据：局部 API 探针已实测，R1 调用链已确认。** `rl_stage.py:113–150` 将固定原始观测直接交给 teacher、`provider.transfer()` 和 `provider.distill()`；L4 `provider.py:328–345` 的后二者明确接受 policy observations，而 `provider.py:230–239` 的公开预测对 raw observations 应用 VecNormalize。

TD3、normalize=True 探针中，传权重的 policy-space parity 通过；同一 raw corpus 经正式 `provider.predict()` 后，动作最大差为 **0.5830942392**。teacher 动作约 0.642/0.682/0.581，正式预测均约 0.0993。

**影响：** 开启归一化时，迁移/蒸馏训练域与最终推理域可能不一致，造成不必要蒸馏、迁移拒绝或无法达成正确 parity。L5 后续还会使用公开 student 做 parity 并在失败时阻断；本次并未证明整个 R1 会静默接受错误最终策略。当前成功 A4 样本未覆盖这一配置。

**修复与验收：** 明确原始域与 policy 域的数据类型/接口；teacher 在其约定输入域产标签，student 训练输入使用与正式推理相同的冻结变换；绑定 scaler/normalizer 身份；覆盖 normalize=True、显著不同输入尺度、DIRECT 和蒸馏路径，以及四算法 R1 最终 raw-input parity。

### F04 · P0 · 跨层：缺少正式数据驱动的应用入口与 L3 生产工厂

**证据：源码确认。** 11 包没有项目级 train/backtest/wfo/report console script。存在 provider 插件 entry points 和 L2 `__main__`，但二者不能替代完整应用。当前完整装配在 `tests/test_native_training_episode.py` 的 `_episode()`/`_env()`；安装级 native ES worker 仍要求应用注入 L3 工厂。

**影响：** 用户尚不能通过一个冻结配置，完成真实数据读取→环境构建→独立 RL/ES 或 Hybrid→结算/评估→报告。剩余工作同时包括代码装配与外部数据，不能描述为“仅差行情文件”。

**修复与验收：** 提供可安装应用入口、配置校验和生产 factory；连接 L2 manifest/真实历史规则、L3 NativeBoot/事件/特征/风险、L4/L5 和 L6 gate；先做有界真实数据回测，再运行四算法和 A0–A4 必需组合。入口不得使用测试授权、固定价格或默认补造缺失经济事实。

### F05 · P1 · L2/L3：观测未覆盖设计最低信息集

**证据：源码确认。** `episode.py:207–217` 只装配 equity/peak_equity/drawdown、guard_intervention、prev_executed_target，以及输入 market/time。未装配设计所列净敞口、现金、未实现盈亏、保证金/可用余额、风险 headroom、前次 raw target、挂单量、fill rate/slippage 等状态。

ObservationBuilder 按 schema 查找列；仅扩展 schema 无法从当前装配获得这些值。市场输入支持扩展，也不等于实际收益率/波动/成交量/spread/funding/basis/多周期特征配方已完成和验收。

**修复与验收：** 建立 native account/order/fill/risk facts 到完整观测的映射，明确单位/时点/缺失策略；冻结市场特征配方与列顺序；验证开仓、部分成交、挂单、费用/funding、风险裁剪和跨 fold reset 下各列正确且无未来信息。

### F06 · P0 · L3：执行情景身份尚未完整绑定原生模型参数

**证据：源码确认/集成未验收。** `execution/stress.py:23` 的 `resolve_scenario()` 生成 model ID、latency/slippage 数值与 hash；除导出外未发现生产调用。`nautilus_adapter/bootstrap.py:13` 的 NativeBootConfiguration 接受注入 fee/fill/latency/margin model，未见完整工厂把已声明情景/历史规则与这些原生对象的实际参数一致绑定。

**影响：** 当前合成引擎成交和费用可工作，但不足以证明“声明的 base/stress 配置就是实际执行的数值模型”，尤其是真实费用、延迟、保守滑点、历史保证金和清算能力。

**修复与验收：** 从冻结配置/规则构造官方模型并记录可核查参数；订单计划身份绑定实际配置；完成 base/stress 成交时点、数量、价格、费用、funding、保证金的事件级对照；数据粒度不足时明确限制能力。不得在引擎外二次扣成本来补偿模型缺口。

### F07 · P0 · L6/编排：完整 WFO 与研究决策流程未实现闭环

**证据：源码确认。** `research/wfo.py:72/112` 提供窗口计划与 L2 manifest 校验；`test_research_wfo.py` 验证边界/结构，未执行完整逐 fold 训练研究。基线是动作策略，统计是函数，stress 是冻结请求计划。未见统一 runner 执行全部研究步骤。

**缺口：** 60/15/15、360/15/15、step 15 的实际运行；warmup/purge/embargo 与 scaler 只拟合 Train-inner；Validation 预算选择；一次性 Test/Holdout；holdout 前最终重训；同市场成本的 B0–B3；bootstrap/DSR/PBO 与全部试验计数关联；压力场景执行与统一报告。

现有 SelectionLedger 的一次性许可和预算预留是有效基础，但 metadata/调用许可还不等于对所有实际数据读取路径的隔离证明。

**修复与验收：** 实现逐 fold runner 和研究报告；让输入数据视图/哈希、拟合状态、预算/候选/选择/重训/测试状态关联到实际执行；完成小规模真实数据全流程，再扩到完整窗口。用反例验证未来数据、重复 holdout、失败重试重置预算和跨 fold scaler/账户污染均被拒绝。

### F08 · P1 · L3/L6：原生账户事实到研究指标缺少生产桥接

**证据：源码确认。** `metrics.py:59` 要求等间隔快照以及每期 fee/funding/slippage/turnover/external_cashflow 增量；当前 NativeFactsReader 产出原生账户/事件事实，未发现生产调用把这些事实规范为指标所需行。`from_account_facts` 的使用位置仅是定义、导出和测试。

**影响：** 指标函数本身通过测试，不等于实际原生回测能生成口径一致的经济报告。尤其须防止把累计费用逐期重复相加或将已入权益成本再次扣除。

**修复与验收：** 增加明确的原生事实采样/增量归并层，保存现金流、费用、signed funding、周转和成交 PnL 的来源；验证 NETTING 多次往返、跨 funding、外部现金流与不规则采样；输出账户权益与报告逐项对账证据。

### F09 · P1 · L4/L5：R1 与整条工作流的应用级恢复缺失

**证据：源码确认。** RLProvider 已支持底层保存/加载；真实 PPO/SAC 恢复测试成立。但 `RLStageExecutor.__call__` 当前执行 create→transfer→train→export→close，没有 stage checkpoint/load/resume 装配。逐 worker 的 EpisodeReplay/数据游标仍由调用者供应。

**修复与验收：** 给 R1 加入版本化阶段恢复与实际用量对账；保存/重建各 worker 的 replay、seed schedule、模型/优化器/replay buffer、normalizer、RNG 和数据配置；让 A4 从已完成 E1、未完成 R1、已完成 R1 等边界续跑。按设计验收阶段边界恢复，不额外承诺任意环境 mid-step 逐位恢复。

### F10 · P1 · L5/L6：CUDA ES 的检查点路径与 GPU ownership 尚不闭环

**证据：源码确认；未运行完整 CUDA ES stage。** `EvoTorchProvider._state_components()`（provider.py:396–401）与 resume 路径明确只认证 CPU；而 ESStageExecutor 在第一代完成后必定调用 save（es_stage.py:235 起）。CUDA EvolutionConfig 可构造，因此当前执行器可能先消耗第一代预算再因 CPU-only codec 失败，预检不足。

L6 已有 GPULearnerLease/ComputeAllocation，但 R1 执行器未装配它们；L4 CUDA 创建主要检查硬件可用；ES config 的 gpu_lease_id 非空检查本身不验证活动 lease 所有权。

**修复与验收：** 支持 CUDA codec，或在任何候选执行/计费前明确拒绝不支持的组合；应用持有并验证独占 GPU lease，失败时释放；测试真实 native GPU RL/ES/Hybrid、冲突 owner、OOM 和新 attempt 策略。禁止静默降级 CPU。

### F11 · P1 · L6：实际资源上限、完整故障恢复与性能验收不足

**证据：源码与测试范围确认。** `allocation.py:1–5` 明示配额是 reservation，不是 OS cgroup/Job Object 限制。当前有 worker timeout/故障回收、有序关闭后恢复；没有完整生产 native 流程的进程内存硬上限/超额监测、强杀/断电、磁盘写满/OOM 恢复验收。

没有收集到 perf 用例，CPU/GPU/Hybrid 的固定同任务端到端 benchmark、三次重复与吞吐/峰值内存/耗时对照尚无完整证据。

**修复与验收：** 定义并落实 Windows 进程资源监控/上限和超额中止；故障注入验证预算不回退、checkpoint 原子性、worker/lease 清理及新 attempt；用相同任务、数据、预算和可比结果执行 CPU/GPU/Hybrid 三次基准，包含 IPC、环境构建、评估和工件 I/O 时间。

### F12 · P1 · L4/L6：当前 L0 导出尚不是完整交易工件

**证据：源码确认。** `actor.py:272–287` 导出 actor、policy、fingerprint 和内部 normalization；外部 L2 feature scaler 仅记录 SHA256。函数文档也明确为 actor-only，不是 trading artifact。

**影响：** actor 动作可以独立复现，但仅凭这份目录无法重建完整特征、外部 scaler、SoftRisk/HardRisk、执行配置与历史规则。研究 ArtifactStore 的存在不能代替自动绑定和装配。

**修复与验收：** 生成独立可加载的完整 L0 bundle，包含特征配方/列单位、scaler 状态、actor/normalizer、风险/执行/规则身份与实际配置、依赖锁和完整性清单；在无 tests 依赖的干净安装中加载并重放，核对观测/动作/订单/权益。L1 续训工件另保留完整阶段状态。

### F13 · P1 · 发布：dist 多份 wheel 落后于当前源码

**证据：已实测字节比较。** 详见第 8 节。旧 gym wheel 缺 episode.py；adapter wheel 缺 market_data.py/submitter.py；compute wheel 缺 validation.py；其他多包存在内容差异。不是只差文档。

**修复与验收：** 统一从冻结源码重建全部包，清楚区分历史 dist 与新候选；在干净 Python 3.14.7 环境仅安装候选 wheel，确认插件/worker 的来源路径并跑最新回归、恢复和应用测试；形成同一批源码、锁文件、wheel 与报告的可追溯身份。此修复不能用单独重建 orchestrator 代替。

### F14 · P1 · L6 CI：nightly 所需真实金标与性能测试为空

**证据：已实测收集 + workflow 确认。** `.github/workflows/l6-ci-reusable.yml:52–61` 逐一要求 real_golden、perf 收集成功；本次两者合计为 0。门禁设计为失败关闭，不是已被绕过；当前测试材料不足以满足门禁。

**修复与验收：** 加入有独立预期账户/成交轨迹的受控真实数据金标和固定预算 perf 用例，正确标记并运行；校验 nightly/release 实际结果。不得只把合成测试改 marker 名称来声称真实市场验收，也不得移除空集门禁来获得绿色状态。

### F15 · P2 · 总集成：快照、版本与文档口径未统一

**证据：文件确认。** 总集成历史记录、继续开发报告与旧 engineering JSON 分别保留不同时点的计数；旧 JSON 的干净安装快照总计 126 项（其中 4 skipped），不覆盖当前 213 项源码。根项目没有统一 Git 身份。

**修复与验收：** 保留历史记录，明确标注时间/范围/源码身份；建立当前唯一验收索引和版本变更记录；统一根项目源码快照、依赖、11 包 wheel 和验收报告的对应关系。不得把历史干净安装或当前 editable 源码回归扩写为最新全包安装通过。

## 7. 对照设计验收门禁

| 门禁 | 当前审计判断 | 依据与缺口 |
|---|---|---|
| Q0 运行环境 | 部分满足 | Python/CPU 全套/CUDA SB3 可用；最新版全包干净安装未重验 |
| Q1 数据事实 | 未通过正式验收 | L2 基础功能存在；F01、真实历史事实覆盖/对照缺失 |
| Q2 原生引擎 | 部分满足 | 合成事件真实引擎通过；真实执行模型/保证金能力未闭环 |
| Q3 Gym 环境 | 部分满足 | barrier/risk/reward/replay/spawn 有证据；F02/F05/F06 |
| Q4 四算法 | 部分满足 | CPU native 各 4 步、CUDA Pendulum 各 128 步；非正式研究与 native GPU 验收 |
| Q5 Hybrid | 部分满足 | 实际 provider 合成 A4 PPO 通过；F03/F04/F09/F10 |
| Q6 研究可信度 | 未完成 | WFO/基线/统计/选择/holdout/报告尚无可执行全流程 |
| Q7 计算与性能 | 部分满足 | spawn/quota/lease 基础在；真实端到端基准/硬资源/故障矩阵未完成 |
| Q8 工件与恢复 | 部分满足 | actor/L4 底层/CPU ES 分代恢复在；完整交易 bundle/工作流恢复/全包发布缺失 |
| Q9 升级/发布 | 未完成本次验收 | CI/attestation 代码在；nightly 空集、旧 dist、统一快照问题 |
| Q10 Paper | 未验收；可选 | 本次未开展 paper 运行；不作为核心研究闭环的替代证据 |

不建议给出完成百分比：设计中的关键能力是串联依赖关系，模块/函数数量无法合理折算为完成度。

## 8. 安装工件审计

本次比较 `dist/*.whl` 中 Python 源文件与当前 `packages/*/src`。结果仅针对可读 Python 文件，不等同于二进制/元数据/依赖完整验收。

| dist 包 | 比较结果 |
|---|---|
| contracts / data | Python 源文件匹配，未发现缺失 |
| compute_runtime | benchmark.py、lanes.py、workers.py 不同；缺 validation.py |
| execution | planner.py 不同 |
| gym_bridge | __init__.py、env.py、reward.py、vector.py 不同；缺 episode.py |
| nautilus_adapter | __init__.py、bootstrap.py、session.py 不同；缺 market_data.py、submitter.py |
| research | __init__.py、ledger.py、wfo.py 不同 |
| risk | guards.py 不同 |
| rl_sb3 | __init__.py、actor.py、checkpoint.py、envs.py、errors.py、provider.py 不同 |
| es_evotorch / orchestrator | 当前审计身份读取 dist wheel 时 PermissionError；未完成内容比较 |

`.artifacts/build-r1-final` 中另有 23:19 构建的 orchestrator wheel，含新 rl_stage 实现，但单包候选不代表全部 11 包已经同步更新或重新干净安装。

**实际状态：当前源码工作区的测试是绿色；现有 dist 不应作为当前源码的完整交付包。** 两个文件的读取权限限制只影响该部分工件检查，未阻断其源码及当前环境测试。

## 9. 后续开发顺序与验收要求

### 第一阶段：修复数据/策略正确性与装配前提

1. L2 关闭 F01，提供真实数据/历史规则/DQ 能力覆盖报告。
2. L3/L1 边界关闭 F02；L4/L5 关闭 F03，覆盖归一化迁移与蒸馏。
3. L2/L3 完成 F05/F06：完整观测、native 模型绑定及事件级账户对照。
4. 总集成完成 F04，交付可安装生产 factory 和有界真实回测入口。

**阶段出口：** 一份冻结配置和真实受控数据，从正式应用入口运行独立 RL/ES，观测、订单、费用/funding、权益与模型身份可对账；错误数据/配置在执行前阻断。

### 第二阶段：补齐研究与恢复闭环

1. L6/总集成完成 F07/F08：实际 WFO、Validation、Test/Holdout、基线/统计/压力和报告。
2. L4/L5/L6 完成 F09/F10：阶段恢复、预算对账、GPU 支持组合预检和 ownership。
3. L4/L6 完成 F12：完整 L0 交易 bundle 与 L1 续训工件。

**阶段出口：** 有界真实数据 A4、必要独立算法/路由组合及恢复可复现；一次性研究预算、分割和 holdout 规则约束实际执行；干净加载工件得到一致交易行为。

### 第三阶段：资源、安装和交付验收

1. L6 完成 F11/F14：资源/故障矩阵、真实金标、CPU/GPU/Hybrid 三次性能基准、实际 CI。
2. 总集成关闭 F13/F15：统一快照/版本、全包重建、干净安装与当前报告索引。
3. 将 Q0–Q9 的证据绑定同一发布候选，逐项关闭，不用测试总数替代门禁。

## 10. 本次命令与可复核证据

以下命令均在项目根目录运行。`--no-sync` 表示审计使用现有锁定环境，不表示本轮重新干净安装。uv 缓存设置为项目内 `.uv-cache`。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --locked --no-sync pytest -q --disable-warnings --tb=short --show-capture=no
$pyfiles = rg --files packages -g '*.py'
uv run --locked --no-sync mypy $pyfiles
uv run --locked --no-sync ruff check packages tests
uv run --locked --no-sync ruff check packages ci qa tests
uv pip check --python .venv\Scripts\python.exe
uv run --locked --no-sync pytest --collect-only -q -m 'real_golden or perf'
uv run --locked --no-sync pytest --collect-only -q -m 'short_wfo or artifact_restore or dependency_resolver'
.venv-cuda\Scripts\python.exe -m pytest -q tests/test_cuda_smoke.py --disable-warnings --tb=short --show-capture=no
$env:PYTHONIOENCODING = 'utf-8'
.venv\Scripts\python.exe .tmp/audit_functional_probes_20261011.py
git -c safe.directory='D:/Program Files/NautilusTrader_RL_ES/nautilus_trader' -C nautilus_trader status --short
git -c safe.directory='D:/Program Files/NautilusTrader_RL_ES/nautilus_trader' -C nautilus_trader diff --stat
git -c safe.directory='D:/Program Files/NautilusTrader_RL_ES/nautilus_trader' -C nautilus_trader diff --cached --stat
```

定向探针结果：[audit_functional_probes_20261011.json](<D:/Program Files/NautilusTrader_RL_ES/qa/reports/audit_functional_probes_20261011.json>)。临时复现脚本：[audit_functional_probes_20261011.py](<D:/Program Files/NautilusTrader_RL_ES/.tmp/audit_functional_probes_20261011.py>)。

关键源码定位：

- [L2 DQ 判断](<D:/Program Files/NautilusTrader_RL_ES/packages/data/src/rles_data/core.py:479>)；[能力授权](<D:/Program Files/NautilusTrader_RL_ES/packages/data/src/rles_data/core.py:601>)。
- [ObservationSchema](<D:/Program Files/NautilusTrader_RL_ES/packages/gym_bridge/src/rles_gym_bridge/observation.py:13>)；[观测装配](<D:/Program Files/NautilusTrader_RL_ES/packages/gym_bridge/src/rles_gym_bridge/episode.py:207>)。
- [R1 迁移装配](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/rl_stage.py:113>)；[L4 raw 预测](<D:/Program Files/NautilusTrader_RL_ES/packages/rl_sb3/src/rles_rl_sb3/provider.py:230>)；[L4 policy 输入迁移](<D:/Program Files/NautilusTrader_RL_ES/packages/rl_sb3/src/rles_rl_sb3/provider.py:328>)。
- [CPU-only ES checkpoint](<D:/Program Files/NautilusTrader_RL_ES/packages/es_evotorch/src/rles_es_evotorch/provider.py:396>)；[WFO 计划](<D:/Program Files/NautilusTrader_RL_ES/packages/research/src/rles_research/wfo.py:72>)。
- [研究指标输入](<D:/Program Files/NautilusTrader_RL_ES/packages/research/src/rles_research/metrics.py:59>)；[actor-only 工件](<D:/Program Files/NautilusTrader_RL_ES/packages/rl_sb3/src/rles_rl_sb3/actor.py:272>)。
- [A4 原生联测](<D:/Program Files/NautilusTrader_RL_ES/tests/test_native_training_episode.py:751>)；[CUDA 冒烟](<D:/Program Files/NautilusTrader_RL_ES/tests/test_cuda_smoke.py:27>)；[nightly 必备测试检查](<D:/Program Files/NautilusTrader_RL_ES/.github/workflows/l6-ci-reusable.yml:52>)。

## 11. 最终状态判定

**已合并、可开发、可运行组件与合成原生集成测试；尚未完成完整设计方案，尚不具备正式研究/发布验收结论。**

接下来最优先的工作是修复 DQ 能力授权与归一化边界，补齐真实数据/完整观测/官方执行模型装配，并实现实际 WFO 研究闭环。发布前还须更新全包工件并在干净安装下重做对应验收。

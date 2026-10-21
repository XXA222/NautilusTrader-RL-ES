# 2026-10-17 审计整改：可连续开发目标

**Codex 活动开发目标入口**

可直接交给 Codex Goal 的完整自动执行指令和唯一续做游标见 [DEVELOPMENT_GOAL_CURRENT.md](DEVELOPMENT_GOAL_CURRENT.md)。本文件负责报告映射、任务队列、依赖与验收证据；历史执行细节见 [DEVELOPMENT_PLAN_20261017.md](DEVELOPMENT_PLAN_20261017.md)。发生摘要冲突时，以当前游标文档和仓库现状为准，不能把历史任务记录当成未完成代码缺口。

## 依据与当前基线

- 输入：用户提供的《NautilusTrader-RL-ES 源码深度审计与后续开发建议》（2026-10-17）。报告依据远程提交 `c24d1bc4a47584f91a4c4db93bf77117d4e8341c` 做静态审计。将报告建议转化为本目标中的任务、依赖、验收条件和执行顺序；Codex 后续实际执行仍须遵守下文研究数据门禁。
- 当前工作区的工程改动持续通过本地 Git 备份提交；最新提交及提交后 source manifest 以仓库状态为准。`DEVELOPMENT_PLAN_20261017.md` 记录历史实施，不能把旧测试数字或旧 HEAD 描述当作当前事实。本计划只规定**尚待完成或重新验收**的工作。
- 所有历史费用、过滤规则、流动性和清算假设须保留来源与有效期。现有回放中的 scenario-only 结果不能表述为真实可交易盈利。正式 Test/Holdout 的已有访问记录继续有效；制定计划不消耗新的正式样本。
- 初次编制目标时的只读核对确认：同时间戳归并已存在；E2 两维物理中性已有单测；`FitnessSpec` 仍采用加权惩罚，硬约束筛选未闭环；项目级访问登记已存在；DSR 仍使用当前候选数并注明范围。以前记录的 **142 passed / 52 passed** 是历史定向结果，不能作为当前全量或两平台 CI 通过证据；后续执行结果均以本目标对应的开发计划与证据为准。
- 报告的 C/L 项优先核实现状；R/H 项必须保留待验证性质。不得将“可能导致亏损”“可能有性能瓶颈”直接写成已经复现的经济或性能结论。

## 总目标与完成定义

建立从数据与原生账户事实、Actor 动作、风控与执行，到 ES/RL 目标和独立验证的可追溯闭环。完成时应能回答：一次策略动作为什么产生或没有产生订单、最终成交多少、费用与 Funding 如何进入权益、各训练阶段是否提供独立的费用后改进，以及结论使用了多少真正独立的市场样本和研究尝试。

每个任务的交付物须包括必要源码、确定性回归、运行配置/数据身份、实际执行结果、失败或未认证边界。工程任务的完成门禁是相关单元与集成测试及适用范围 Ruff、MyPy 通过；平台交付和研究晋级再检查对应 Linux/Windows CI 与经济门禁。固定回放除预期经济语义变化外保持一致；不以训练 Fitness、环境步数或计划层 `SUBMIT` 替代原生费用后经济结果。涉及性能时同时报告耗时、峰值 RSS 与结果等价性。实验按冻结协议执行、产出可复核拒绝结论，也可完成该实验任务；候选未通过不能写成通过收益门，更不能靠无限调参维持任务。

## 连续任务队列与续做游标

## 2026-10-21 杠杆与项目范围复核

按用户指示，RouteB `rles-world-model` 包及其 CLI/研究接线、配置、QA、测试和 wheel smoke 注册已移除；当前 workspace 有 **11** 个 RLES 包。杠杆风险限额现预留按执行配方估计的费用与滑点，避免成交成本降低权益后超出配置杠杆。杠杆仍只允许空仓且无订单时调整，随后重建 session 并重放。

2026-10-21 后续资金费修复及全量验证以本文末尾的当前执行记录为准：Binance REST funding `markPrice` 已贯通到原生结算并保留 tick 以外的 mark 精度；REST 首条结算不再从未来推断 interval，合法单行响应也会保留明确结算事实；未知显式费率语义拒绝进入 native interval inference；不同源 ID 的重复结算边界会在 native replay 前拒绝，CuratedStore 也会将此类数据标为 DQ-D。Windows process ancestry 现校验创建时间以区分重用 PID。最新 Windows suite 为 **812 passed、5 skipped、53 warnings**；最终 package smoke/manifest 以本轮记录为准。Hosted Linux/Windows CI 尚未运行，本机结果不替代平台 CI。

严格 instrument_rule allowlist 后的最新本机验证：Windows 全量测试 **798 passed、5 CUDA-only skipped、53 dependency warnings**；Ruff 通过；MyPy **141 files** 通过。HardRiskGuard→Planner 反转先平仓和极端费用估计必须允许强制平仓的回归均通过。历史工厂对跨有效 instrument-rule 版本的窗口本就采取 fail-closed，本轮补了回归覆盖并删除重复检查。Windows staged-file 发布对硬链接的限制也已修复并纳入该全量回归。当前 11 wheels/11 sdists 已重建，wheel manifest `all_match=True`，隔离导入及 entry-point smoke 通过（包括最新 allowlist 变更和 signed-fee 修复）。规则、transition、指标和 linear-equity reconciliation 均保留负 commission rebate 的现金流符号。冻结策略 1x/10x 配对复放记录在 `docs/audit_evidence/20261020/td3_10x_funding/summary_r4.json`；它复用了已消费的 Validation 窗口并使用 scenario-only 交易规则，不能视为独立表现或历史规则证明。账户适用的分档保证金/清算经济、真实账户现金流、Hosted CI 与 Hedge Mode 仍是未闭合门禁；Binance bracket 是签名用户数据且包含账户调整系数，Nautilus Python 只暴露固定比率 margin models，模型实现边界详见最新执行记录。

后续在 `instrument_rule` 增加严格 allowlist，拒绝任何未被静态 native rule model 消费的字段；直接回归了 `margin_tiers`、`leverageBrackets`、`brackets`、notional/maintenance 字段及一般未知字段。该变化后的 146 项定向测试、798 项全量测试、Ruff 和 MyPy 已通过；最新 11-wheel 构建、manifest 与隔离导入 smoke 通过。修复仅拒绝未支持的分档字段，不代表已实现 Binance tiered margin。档位属于签名用户数据并可能因账户调整系数而异；Nautilus Python 暴露的原生模型仍是固定比例模型。

待所有者核验的本地候选清理项：忽略的环境（`.venv*`、`.rles-python`）、生成缓存/构建结果（`.tmp`、`.artifacts`、`dist*`、`.uv-cache`、`.mypy_cache`、`.pytest_cache`、`.ruff_cache`）、`.codex-remote-attachments`，以及独立检出的上游 `nautilus_trader/`。未删除它们，因为可能保留复现输入或当前需要的依赖检出；详见最新开发执行记录。

2026-10-21 再次从标准同步后的 workspace 环境验证：旧 `.venv` wheel 指向已移出当前源码状态的 `.artifacts` 构建目录；根开发依赖缺少 Arrow extra，造成测试数据层不完整。已将 `rles-data[arrow]` 加入根开发依赖并更新 lock。`uv sync --offline --all-packages` 后，定向杠杆/资金测试 **26 passed**，全量 **798 passed、5 CUDA-only skipped、53 warnings**，Ruff 与 MyPy 141 files 通过。分档保证金、真实账户现金流、Hedge Mode 与托管 CI 门禁仍未认证。

**历史进度（2026-10-18，已被 2026-10-21 的项目范围复核 supersede）：** M0.1 原生金额解析及 M0.2 分发覆盖/manifest 范围当时已修复。该时点的 workspace 有 **12 个 RLES 包**，包括 `rles_world_model`；Windows 默认 selector 为 **669 passed、1 CUDA-only skipped、102 deselected、46 warnings**，Ruff 和 MyPy（148 files）通过，12 wheels/12 sdists 的 manifest 为 `all_match=true`。这些是 2026-10-18 的历史结果，不描述当前包数或当前测试状态。2026-10-21 按用户指示移除了 RouteB `rles_world_model` 包及接线；当前包数、最新本机验证和未关闭门禁以本文上方“2026-10-21 杠杆与项目范围复核”和 [当前目标入口](DEVELOPMENT_GOAL_CURRENT.md) 为准。该时点记录的 M0/M1/M2/M3/M4 外部依赖状态同样只代表当时进度；后续更新见开发执行记录。

**最新工程进展：N04 supervised GPU quota 正常释放和后代保护回归已通过。** Windows Job 的 PID 清单同时含有 Python launcher parent 和实际 owner；现在释放前先确认共享 GPU 内核锁已空闲，并确认 Job 内其它活动 PID 都是 owner 的祖先，活动应用后代会令 quota release 拒绝。`tests/test_resource_recovery.py tests/test_compute_device.py` **27 passed**，Ruff 与 Windows MyPy 通过。N04 仍不构成 VRAM 上限或跨 OS-user/平台认证；下一项仍按任务表选择依赖已满足的真实代码缺口。

**M4.1/A09 合成规格闭环已完成。** BTC/ETH/SOL 的原生 Funding/equity parity 和最小数量/名义额拒单原因现有确定性覆盖；Planner 对被 dated minimum 清零的目标返回明确的 `below minimum executable quantity/notional`。low-capacity QA 从已验证数据集里的 dated rule 解析 base currency，不再依赖 ETH 默认值。相关原生集成组 **199 passed**，Ruff、Windows MyPy **148 files** 和 `git diff --check` 通过。剩余 M0/M1/M2/M3/M4.1 部分状态均对应真实现金流、clean Hosted CI、历史访问/试验记录、独立 Validation/新授权样本、兼容阶段工件、真实深度或可信标的规则；索引与执行记录已列明恢复条件。M4.2 仍为 deferred，不在双向持仓/保证金/清算/Funding 合同明确前开放。此轮余项复核未发现可以在现有授权与依赖下继续实施的工程包。

**M2.2/A12 决策级费用归因补全：** 执行轨迹此前把 native fill commission 中币种后的金额直接相加；现在按币种汇总，单币种仍提供兼容 `fee` 与 `fee_currency`，多币种只提供 `fees_by_currency` 并将无单位总额置为 `null`。费用格式、非有限值及混合币种回归通过；完整 `tests/test_historical_factory.py` **59 passed**，Ruff 与 Windows MyPy 通过。没有读取市场或账户数据；本改动修复报告单位语义，不扩展或认证多标的能力。

**M2.2/A12 决策权益归因补全：** Gym 步骤和可选逐决策 trace 现记录策略 barrier 前权益、策略执行后的 barrier 权益、准确差额与币种，并检查权益变化复算一致。终端强平变化继续单列，其期初权益必须与最后一次策略 barrier 的权益一致。完整 Gym bridge 与历史工厂套件 **79 passed、18 dependency warnings**；Ruff 与 Windows MyPy 通过。只用合成原生引擎回归，未访问市场、账户或正式评价 payload，不构成收益认证。

终止结算报告也要求 drain 前后账户币种一致，合成快照不匹配回归与 Gym bridge/rollout 组合 **22 passed**，Windows MyPy 与 Ruff 通过。该防线只校验同一账户的记账单位，没有声称支持多币种账户。

**A02 最新代码缺口已修复：** Constraint-priority v1/v2 与 native ES worker 现在把有限负终值传入破产可行性筛选，拒绝原因为 `episode_insolvency_detected`；旧 Fitness 版本仍保持既有负权益校验。Worker 还验证正权益 solvent episode 的逐步对数收益能复算首末权益。Scorer/dispatcher/worker 回归 **16 passed**，完整 compute-worker pool **30 passed**，原生训练/ES worker **29 passed**，Ruff 与 Windows MyPy 通过。该项仍需独立 Validation 才能评估目标晋级，工程回归没有打开该窗口。

**数据隔离工程游标：Stage69 的真实 Train-inner 特征和 Reset profile 均已 PASS。** v0.4.1 scoped reader 与 ledger-first `read_registered_train_window()` 已接入 `application.prepare()`，训练仅允许 `evaluate=False`，market-audit/WFO/backtest 拒绝 scoped Train-only 配置。60/360 日特征 profile 结果见 [feature 证据包](audit_evidence/20261017/stage69_feature_profile_360d_v2/)。冷/热 Reset 为 60 日 **3.1787/0.0098 秒**、360 日 **21.5383/0.0838 秒**，same-seed 起始 observation/native account fingerprint 重复一致；读取前登记的物理隔离 v0.4.0 快照共 117 个 row group、1,907,766 条事件，正式区间重叠为 0，结果见 [Reset 证据包](audit_evidence/20261017/stage69_episode_reset_360d_v1/)。规则为 scenario-only，没有训练或经济收益结论。旧 M2.4 profile 使用跨正式日期的大文件整读，不作为 Stage69 Train-only 数据证据引用。下一步转到 M3.2/A13 训练覆盖报告源码复核；Stage60 日期已有复用记录，不能当作新独立市场覆盖。

M0.1 已核对空仓、开仓 mark-to-market、手续费、成交、Funding、终止平仓和原生线性清算。NautilusTrader 2.0.0rc6 的 `AccountState` 没有标准化 deposit/withdrawal 类型；Engine 中 `reported=True` 既可和初始账户状态一起出现，也可和 Funding settlement 同时出现，因此不能解释成转账。本轮新增调用方显式提供的现金流事实接口，要求事件 ID、时间、币种、方向、金额、来源 ID 与来源 SHA-256，并将 signed flow 纳入 return 与逐屏障线性权益对账；不从 `reported` 或权益变化推断转账。SHA 只固定来源标识，不证明源文件真实性。无真实类型化来源时，未知或相互抵消的现金流仍不能从原生余额推断，M0.1 保持 `PARTIAL`。旧证据见 [90 天 Train-inner](audit_evidence/20261017/m0_1_native_equity_90d.json)、[Engine 线性强平](audit_evidence/20261017/m0_1_linear_native_liquidation.json) 与[合成清算交叉案例](audit_evidence/20261017/m0_1_synthetic_linear_liquidation.json)；本轮回归见执行计划。

M0.2 将有来源/哈希的 1 秒 BookTicker Quote 用例纳入默认 PR `micro_golden` 标记。本地 Windows 上实际执行工作流相同选择器 **478 passed、60 deselected、38 warnings，115.26 秒**；Quote 用例通过。`.github/workflows/l6-ci-reusable.yml` 的 Linux/Windows 远程 job 尚未运行，须在干净交付后验收。

M1.1 已通过实现等价验收。M1.2 默认保留旧 `FitnessSpec`；新 `constraint_priority_train_inner_v1` 先筛 episode 破产、维持保证金不足和原生强平，再按有界费用后经济分数排名。保证金/强平分类只在原生快照声明能力且事实完整时启用；新协议遇到未知事实失败关闭。HardRisk 干预频率不作为违规。拒绝原因贯穿候选聚合、新协议 checkpoint 与最终 bundle lineage；真实 A4 E2 中断恢复验证了报告持久性。三种目标已在相同 Train-inner 归档、冻结 R1 actor、共同随机数场景及相同预算下分别完成 A2 搜索和同参数原生回放；三组均无不可行候选，约束优先与零干预惩罚选中相同参数，Train-inner 净收益均为 -0.1510%，legacy 为 -0.1578%。规则仅为 scenario-only，未做 Validation，因此 M1.2 保持 `PARTIAL`，不能由此推断目标优劣或策略盈利。最近定向回归 **17 passed、13 warnings**；Ruff 和 MyPy（4 个源码文件）通过。详见 [M1.1 证据](audit_evidence/20261017/m1_1_neutral_evaluation.json)、[M1.2 实现证据](audit_evidence/20261017/m1_2_constraint_priority.json)、[三目标 Train-inner 证据](audit_evidence/20261017/m1_2_three_objective_train_inner.json)。

状态定义：`PARTIAL` 表示已有实现但验收不完整；`READY` 表示当前可开展工程工作；`PENDING` 表示依赖未满足；`CONDITIONAL` 表示依赖数据、独立样本或平台证据；`DEFERRED` 表示暂缓。表中记录当前已知状态；每次实际执行后按证据更新状态、游标和执行记录。

| 顺序 / ID | 对应审计项 | 当前状态 | 依赖 / 完成证据 |
|---|---|---|---|
| 01 / M0.1 | A08 | PARTIAL，等待已认证且覆盖可证明的真实现金流采集 | 显式外部现金流输入、身份/时间/来源哈希校验、Binance USDⓈ-M `TRANSFER` JSON 导入及平仓/持仓权益核对已有回归；分页归档入口校验 inclusive 页段连续覆盖、越界事件与潜在截断，manifest 绑定每页原始 SHA；乱序 native snapshots 按时间统一排列 fill/Funding/清算生命周期；清算订单累计 `filled_qty` 拒绝回退/超量；Transfer JSON 单页/分页解码拒绝重复对象键；核心金额边界现在不再删除逗号，畸形数值失败关闭。`native_report()` 默认拒绝 off-barrier 近似收益；盘中 TWR 要求来源身份/哈希绑定的转账后权益和毫秒时间精度，按固定采样周期分段链式计算；同刻多笔资金合并净额，权益点只应用一次；缺点、错位点、首次后续屏障错误或同毫秒 native fill/Funding 无序事件失败关闭。最新现金流/指标回归 **109 passed**。真实账户来源、查询完整性和独立账本核验仍待授权证据；来源 SHA 不认证账户真实性或查询完整性 |
| 02 / M0.2 | A10 | PARTIAL，本地 Windows PR/静态/当前源码分发验收通过；Hosted clean Linux/Windows CI 待运行 | 工作流按 `packages/*/pyproject.toml` 数量核对 wheels；wheel smoke 导入包根与全部已声明 entry point 并核对 wheel provenance。最新全量 Windows 套件 **819 passed、5 CUDA-only skipped、53 warnings**；全仓 Ruff 和 `mypy packages`（105 files）通过。当前源码构建 11 wheels/11 sdists，wheel manifest `all_match=True`，隔离 smoke 验证 11 包/5 entry point；Hosted Linux/Windows CI 尚未运行，本机结果不替代平台 CI |
| 03 / M0.3 | A04 | PARTIAL，项目权限与 Train-only 所有应用/worker 入口 scoped-reader 回归通过 | 权限回归覆盖身份、半开区间、并发、跨 owner/output 同 query 重开拒绝、崩溃/重试、历史导入、同 owner Test 重标拒绝、WFO 同 query 同 split/mode 组件例外及访问行身份/状态/删除保护；`validate`、Train-only `train`、`market-audit`、`diagnose-train`、R1 多环境 worker 和 E1/E2 候选 worker 都经 ledger-first scoped reader。最近新增 `purpose`、身份键与开发来源 `schema_version` 畸形输入校验，访问/WFO/CLI 回归 **128 passed**。正式 Evaluation 配置被 Train-only 入口读取前拒绝；登记启用前的历史访问覆盖仍无法证明 |
| 04 / M1.1 | A01 | PASS，合成 Train-inner perpetual parity；E2 neutral and legacy restore regressions | 同一 frozen L4 actor 的基准 SoftRisk 与 `(1,1)` 逐步经济 trace、fills、Funding 和指标相同；新版 E2 中性候选的两代 checkpoint/resume 对照验证 evidence、durable ledger（5→10 requests）、预算与最终选择；pre-neutral E2 schema 3 恢复不插入额外候选（6 checkpoint tests passed） |
| 05 / M1.2 | A02 | PARTIAL | v1 保持兼容；v2 候选级 lexicographic feasibility 排序令任何不可行 episode 的选择分数为 -2，并将可行分数严格约束在 (-1,1)，lane 拒绝越界可行响应。最新修复了有限负终值被旧输入校验挡在破产筛选之前的问题；constraint-priority v1/v2 与 installed worker 均覆盖 `final_equity < 0`。Compute-worker pool **30 passed**、native training/ES worker **29 passed**，独立 Validation 证据未满足，不能晋级新目标 |
| 06 / M1.3 | A11 | PARTIAL（v2 三阶段分析器工程通过；完整原生经济报告缺输入） | v2 分析器验证 E1 genome/spec/hash/lane evidence 与 bundle lineage，要求 E1/R1/E2 报告绑定同一预注册及 market/rules/window/seed/environment/execution/hard-risk/scaler 身份，复算共同 native timestamps、成本对账和三个阶段差值；11 项合成回归通过。既有 R1-neutral→E2 配对保留，仓库暂无兼容 E1 bundle/三阶段报告，不重训或重放历史窗口制造输入 |
| 07 / M2.1 | A03 | PASS（T0/T1 全决策覆盖诊断） | 冻结的 90 天 Train-inner T0/T1 固定策略配对；90 个连续 rank-exclusive episode lane 完整覆盖 T0 的 129,597 个决策步（此前实现漏 87 步，现已修正）；T2 未开发，不属于当前验收范围；收益与成本仍是 scenario-only，T1 独立起始余额求和不构成复利收益或纯强制平仓因果效应 |
| 08 / M2.2 | A12 | PASS（工程验收） | 可选逐决策 trace 贯通 Actor normalized raw action、SoftRisk/HardRisk target、计划、订单、fill、币种费用与权益变化；`CANCEL/WAIT` 行补充相关既有订单的最终状态、累计 fill 数量与事件 ID、费用和残量原因，submitted fill 与相关订单事实分开。原生订单身份拒绝空白及首尾空格、拒绝冲突重复事实并仅折叠完全相同快照；`FILLED`/`PARTIALLY_FILLED` 状态必须与累计成交量一致；当前 NautilusTrader 的 15 个 `OrderStatus` 变体均得到显式 trace 结果。无效拒单原因退回稳定状态标签，完整历史工厂回归 **79 passed**，Ruff/Windows MyPy 通过；最新默认 Windows selector 与当前 wheel 证据见执行计划 |
| 09 / M2.3 | A06 | PASS（Bar Base/Stress 与合成部分成交压力工程回归）/ CONDITIONAL（真实 Quote/Depth） | Base 保留足量合成深度；Stress 的 `simulated_depth_fraction` 绑定配置身份并由原生 Nautilus 引擎实际部分撮合；同窗历史深度仍缺，因此不认证真实流动性 |
| 10 / M2.4 | A07 | PASS（60/360 Train-inner 性能与经济/Replay 等价） | 工厂范围缓存可见批次与不可变 BarrierFrame；报告冷/热 Reset、吞吐、峰值 RSS 和缓存计数；72 项工厂/训练 Episode 回归通过；后续完整 `test_historical_factory.py` 通过 47 项 |
| 11 / M3.1 | A05 | PARTIAL | WFO 最终 DSR 报告现同时包含本 protocol 的候选证据和共享登记库跨 protocol 的 standalone attempt/recipe/feature/seed/lifecycle 汇总；注册 recipe/detail 将非有限或非 JSON 输入规范为 `ValueError`，seed 限定为 `[0, 2**32)`，防止超范围种子进入 SQLite。新增 Python 与 SQLite 双层生命周期迁移校验，允许 WFO 选择后的阶段恢复和 standalone failed/interrupted 恢复；直连数据库的迁移触发器回归也通过；研究访问/WFO/CLI 组合 **144 passed**，Ruff、Windows MyPy 通过。未登记的历史人工选择和失败尝试仍不可恢复，历史有效试验数保持 `not_estimated` |
| 12 / M3.2 | A13 | PARTIAL（代码诊断通过，研究覆盖待授权） | 训练 diagnostics 分列 interaction samples、唯一决策 timestamp/revisit fraction、Episode overlap pair 和固定 volatility/trend/Funding regime bins；本轮 overlap pair 统计改为 O(n log n) sweep/heap，57 项 RL/native Episode 回归通过。Stage60 v4 使用复用 Train 日期；历史 `recorded` 标签不构成新读授权，独立市场覆盖仍缺合法的新授权协议 |
| 13 / M3.3 | A14 | PASS（Train-inner 诊断）/ 未晋级 | 36 份冻结 Base/Stress 报告完成 51,732 步逐决策归因、固定状态分组和区块 bootstrap；权益对账通过，零 Validation/Test/Holdout 查询；冻结 RL 全部落入 deadband、无实际敞口/成交，未通过跨窗口基线优势门槛 |
| 14 / M3.4 | A11/A14 | CONDITIONAL | 上述研究门禁及合格独立窗口；先小预算筛查，再决定路线扩张 |
| 15 / M4.1 | A09 | PARTIAL（合约单位实现与合成 BTC/ETH/SOL 回归通过；真实规则/费用认证待外部证据） | 规则事件可携带哈希绑定的显式 `contract_spec`；非 legacy 标的缺规格即拒绝。历史工厂、特征单位、观测账户币种、数量规划及原生线性永续消费该规格；手续费、Funding、成交额、滑点和权益对账统一使用冻结乘数。BTC/ETH/SOL 同构小样本覆盖不同乘数、数量精度、最小数量/名义额、Funding 费用币种及平仓/持仓权益单位；ETH golden 保持回归通过。该合成工程范围完成，真实交易所规则/费用与经济认证仍受 M0.1/M2.3 等门禁约束 |
| 16 / M4.2 | A09 | PARTIAL / Hedge Mode DEFERRED | linear NETTING 固定杠杆和空仓重建已实现；Hedge Mode 及真实分档保证金/清算/Funding 合同与金标准数据待定义 |

依赖只约束依赖它的工作：远程 CI、历史规则或真实深度阻塞时，可以继续不依赖该证据的工程修复与明确标注的 Train-inner 诊断。M1 的 P0 工作不必等待真实深度或正式 Holdout。依赖未满足时不得开启正式晋级实验，也不得把可推进的工程项一起搁置。

## 里程碑 M0：经济事实与可复现基础（先决条件）

**M0.1 原生账户账本闭环（A08，P1）**。在现有同时间戳快照事件归并的基础上，定义决策快照、终止平仓快照及相同事件 ID 的次序契约；核对 fill、手续费、滑点、Funding、已实现/未实现盈亏与最终权益。加入同时间戳多成交、多 Funding、重复且相同 ID、相同 ID 冲突、终止平仓和清算的金标准样例。验收为每一经济事实恰好计入一次，冲突显式失败，原生总权益与逐事件现金流核对一致；长窗真实数据回放保留可复核差异表。现有归并实现是起点，不视为该闭环已完成。

首个工作包限定在 `packages/research/src/rles_research/native_metrics.py`、`metrics.py` 与必要的原生事实接口，优先复用 `tests/test_research_metrics.py`、`test_native_liquidation_facts.py`、`test_historical_factory.py`。先保留并核对已有 `test_native_report_merges_same_timestamp_facts_and_uses_terminal_equity`，再补真正未覆盖的情况。核对式必须说明期初权益、外部现金流、价格盈亏、手续费和带符号 Funding；滑点已经通过实际成交价进入盈亏时，只作成本归因，禁止从权益再次扣除。初始工程验收不依赖新增正式评价样本；长窗核对仅用已允许的 Train 数据，记录精度容差、规则假设、逐项差额及未解释残差。

**M0.2 CI 与样例交付（A10，P1）**。把默认基础 CI 所需的小型公开数据、来源清单和哈希纳入仓库；真实大文件测试单列数据依赖 Job，缺失时输出明确 `BLOCKED` 与所需文件，不把它统计为策略失败或成功。保留已修复的 Windows API 跨平台类型检查，并在 Linux、Windows 干净 checkout 各跑基础检查。验收为两平台基础检查独立通过；有数据的 Nightly 通过，缺数据的 Nightly 明确列出阻塞证据；Quote 样例不依赖本机被忽略目录。

**M0.3 项目级研究权限（A04，P1）**。复核现有 `ProjectResearchRegistry` 的跨输出目录作用域、半开区间重叠、数据版本变更、进程崩溃和重试幂等性。把所有正式 Test/Holdout 的读取统一接入同一个项目级登记库；记录旧运行导入的来源和无法证明的历史访问。验收为换目录、换数据哈希或换协议 ID 不能把已消费 Holdout 重新标成未用；恢复只复用已完成报告，不重复读取样本；报告分别说明“本协议未重复”与“项目级尚未使用”。

**M0 门禁**：M0.1/M0.2 的经济与可复现回归通过，M0.3 的研究访问测试通过，才能启动需要正式统计解释的实验；开发诊断可继续限定在 Train-inner。没有远程运行证据时 M0.2 的跨平台交付状态保持未验收，不用本地模拟平台类型检查代替实际 CI。

## 里程碑 M1：ES 搜索与目标一致性（P0 优先）

**M1.1 可辨识 E2 参数及精确中性候选（A01，P0）**。复核新运行默认两维空间和旧三维 Checkpoint 兼容路径；禁止新协议再同时搜索两个只靠乘积起作用的常数。把物理参数 `(1,1)` 的 R1 原样候选明确纳入候选矩阵、账本和评估预算，不能依赖有限潜变量的 `tanh` 恰好到达上界。若以后增加动态波动率控制，必须定义因果窗口、观测可用时间、TTL、上下限与缺失行为，并证明与仓位缩放参数的响应可区分。验收为中性候选的逐步订单、fill、Funding、权益完全复现 R1；各有效维度单独变化产生可解释差异；旧 Checkpoint 在冻结身份下可恢复。

**M1.2 约束优先的 ES 适应度（A02，P0）**。保留旧 `FitnessSpec` 作为版本化消融，不静默改变旧 Checkpoint。明确强平、结算失败、权益/保证金/硬风险违规等不可行条件；先拒绝不可行候选，再按预先冻结的费用后经济指标对可行候选排序。经济目标与风险权重写入配置、运行身份、报告。比较旧公式、去 HardRisk 调整惩罚、约束优先三组，预算、场景和随机种子相同。验收为盈利、空仓、亏损、违规四类确定性候选的选择及拒绝原因可解释；新目标只有在独立 Validation 的原生经济指标不恶化时才可替代旧目标，空仓始终保留基准资格。

**M1.3 阶段边际贡献（A11，P1）**。冻结相同 Actor/SoftRisk、风险预算、回放区间与成本场景，分别记录 E1 前后、R1 前后、E2 前后的原生回放。对 A0/A1/A3/A4 路线报告净收益、最大回撤、实际换手、手续费、滑点、Funding、成交量与风险干预差。验收为每个阶段有可复算的增量表；若 E2 没有稳定的独立经济改进，则保留精确中性 R1。

**M1 门禁**：M1.1 的中性完全等价、M1.2 的目标排序和配置身份、M1.3 的阶段原生增量均可复核后，才考虑扩大 ES 搜索预算。

## 里程碑 M2：训练、执行与流动性语义

**M2.1 Episode 终止时域（A03，P1）**。以相同固定策略、相同行情与执行配置配对：T0 连续账户，T1 按短 Episode 强制平仓，T2（若开发）仅切换采样边界而不断开账户。已完成五步原生合成回归和 90 天 Train-inner T0/T1 对照，报告额外平仓次数/费用、Funding、持仓时长、独立 Episode 权益和窗口覆盖差异。M2.1 当前通过范围为 T0/T1；未开发 T2，因此不作 T2 结论。报告边界明确 T1 独立起始余额不复利，情景 PnL 差同时受到强制结算、重开仓、余额重置和未覆盖决策点影响；训练与评估报告不得混用两种收益语义。

**M2.2 执行动作漏斗（A12，P1）**。在现有风险目标、数量取整和原生敞口归因上，逐决策记录 Actor 请求、Soft/Hard 调整、Planner 目标数量、NOOP/WAIT/CANCEL/SUBMIT 原因、原生订单 ID、完全/部分/未成交、最终敞口和费用后权益变化。验收为任一回放决策可沿 ID 找到结果或明确的未成交原因；计划层提交数与真实 fill 数分列；缺失覆盖时标 `unknown`。

**M2.3 流动性证据与成本压力（A06，P1）**。明确 Bar 基准是固定不利滑点加足量深度假设；可冻结 `simulated_depth_fraction` 对合成 Bar Stress 盘口限深，Base 固定为旧足量深度，深度比例参与配置/场景身份；Quote 模式拒绝合成限深并只用原生盘口。原生撮合回归已证明请求 0.500 时 0.200 成交（fill ratio 0.4），剩余订单仍可见。该结果只验证压力模型行为，不是现实流动性证据。剩余研究需对相同策略做 Base、Stress 与同窗、有来源证明的 Quote/Depth 对照；仅有 top-of-book 时限制可证明成交量，不捏造深层盘口。验收为 Fill Ratio、成交量、实际滑点、费用和净收益逐场景可比；没有真实深度证据时标“流动性未经认证”。

**M2.4 Episode Reset 性能（A07，P1，已完成）**。先测当前 `available_event_window` 切片与 Reset 重建的实际成本；只有出现显著热点时才设计工厂级只读事件索引缓存，并保持每个 Episode 新建引擎/账户。验收为固定数据与种子下观察、订单、成交、Funding、权益和 Checkpoint Replay 完全一致；在 60/360 天场景报告 Reset 延迟、总吞吐、RSS 峰值及缓存构造成本。已缓存按可用时间排序的原生市场批次、逐决策增量组及不可变 `BarrierFrame`。首个 Reset 负责冷建缓存，后续随机 Episode 只重建首帧 warmup 批次并复用其余帧。360 天 warm Reset 从 20.01 秒降到 0.070 秒，60 天从 3.62 秒降到 0.017 秒；缓存完整 rollout 的速度提升分别为 12.14% 和 15.35%，最大 RSS 增量与基线同量级（360 天约 1.22 GB）。60/360 天的 native order/fill/Funding 计数、terminal-flat 和 net-return 与未缓存基线完全相同；定向 uncached frame 与缓存 Replay 测试比较观察、边界 token 和原生费用/权益报告。收益只适用于重复使用同一冻结工厂的多 Episode 情形；一次性冷启动延迟近似原基线。

**M2 门禁**：能够区分人为终止、动作未执行、流动性/费用损失与价格信号错误，才进入更大训练实验。

## 里程碑 M3：Alpha 与统计可靠性

**M3.1 有效试验数（A05，P1）**。在项目级登记候选、特征版本、参数组合、种子、人工挑选和失败尝试；制定预先冻结的有效试验数估计政策。现有 DSR 只标明“本协议候选”，不得作为全历史校正。验收为 DSR 显示试验数来源、去重/相关性假设、历史覆盖界限与敏感性；无法估计时明确标 `not_estimated`，不输出全历史显著性结论。当前 WFO 报告已写入 `trial_count_evidence`，包括冻结候选 ID 与 SHA-256、账本观察覆盖、候选 ID 去重范围、跨折处理及独立试验 Sharpe 假设。共用项目 SQLite 现在追加记录候选配方及哈希、特征哈希、seed 和 started/selected/not_selected/failed/interrupted 生命周期事件；协议恢复按身份和事件键幂等。只读旧访问清单包含 10 个 WFO 协议库、21 条 Validation 与 1 条 Test 查询、18 个协议/候选组合、10 个候选 ID、22 条 standalone 报告索引和 67 条访问登记，范围限于两个已知根；离线盘点脚本已把 11 个匹配配置中的 10 种参数/seed/feature 配方指纹与 18 个协议/候选组合分开记录；配方指纹只是精确配置去重，不是有效独立试验数。人工筛选及未入账失败研究仍不可还原。当前协议报告已增加 effective trials=1 完全依赖端点与候选数独立端点的 DSR 场景敏感性，固定观测候选 Sharpe 离散度且不估计相关系数。历史登记前研究仍未覆盖，旧候选缺少跨候选同时间切片回报，不能进行历史相关性估计或全历史去重；历史有效试验数仍为 `not_estimated`，工作包保持 `PARTIAL`。

**M3.2 独立市场覆盖（A13，P1）**。把环境交互步数与独立市场时间点分开，报告 Episode 重叠率、唯一时间点、持仓状态、波动/趋势/Funding Regime 覆盖及各状态收益贡献。验收为重复采样只计计算预算，不计独立样本；固定新增 Train-inner 数据能否改善 Validation 的检验先于扩大训练步数。

**M3.3 费用后动作价值（A14，P1，Train-inner 诊断已完成）**。冻结的 36 份 Base/Stress 原生回放已配对比较冻结 Actor、空仓、持有及趋势/反转基线，完成固定波动/趋势/Funding 状态、执行可行性、费用后账户变化及 60 分钟区块 bootstrap 归因；51,732 个策略决策时点均完成原生权益核对，分析不重复扣除已进入权益的滑点。所有六个冻结 RL 回放的请求仓位都未越过执行 deadband，实际敞口和成交为零；结果只支持“执行可行性阻断”的机制结论，不能诊断方向信号是否有效。四个窗口为复用 Train 日期，且未通过跨窗口基线优势门槛，因此 Train-inner 诊断任务完成，但独立收益/晋级门槛未通过。最终分析与冻结修订分别见 `docs/audit_evidence/20261017/m33_action_value/analysis_final.json` 和 `preregistration_revision5.json`；不得重训、选模或把结果写成独立策略证据。

**M3.4 训练路线对照（A11/A14）**。仅在 M0–M3.3 门禁通过且有未消费的独立窗口时，按相同成本、风险预算、种子和 Validation 访问预算比较不交易/持有、简单趋势/反转、PPO、SAC、TD3、DDPG、E1、R1→E2、E1→R1→E2。主要指标预先冻结为费用后净收益及风险、费用、成交与置信区间。当前 gamma 三 seed/三窗对照未通过优势门禁，不追加同一机制训练预算。零成本都不能超过简单基线时，先回查特征、动作和环境语义。

**M3 门禁**：没有跨独立窗口、费用后且可执行的增量证据，不开启新的正式 Holdout、不宣称盈利、不扩大模型搜索。

## 里程碑 M4：能力扩展（最后实施）

**M4.1 标的元数据（A09，P2）**。将 ETH 数量单位等硬编码替换为从已验证 InstrumentRule/合约规格导出的元数据，先保持单资产 NETTING 行为；BTC/ETH/SOL 用同构小样本验证数量精度、最小名义额、Funding 与权益单位。`instrument_rule.contract_spec` 固定原始符号、base/quote/settlement currency、inverse 标记和 multiplier；非 legacy 标的缺规格时失败关闭。历史工厂、特征/账户单位、native `CryptoPerpetual`、Planner 数量和名义敞口、native turnover/slippage/cash-flow/equity reconciliation 均消费该规格。BTC/ETH/SOL 的合成 Funding/equity parity、非单位 multiplier 原生金标准，以及三币种最小数量/名义额拒绝用例已通过；低于规则的目标现返回准确 NOOP 原因。QA 特征分析也从该窗口的 dated rule 解析 base currency。工程验收通过；真实规则、费用/Funding 与多标的经济认证仍待可信来源。Inverse adapter 与 Hedge Mode 未认证，M4.2 保持延后。

**M4.2 Hedge Mode 与杠杆认证（A09，P2）**。linear NETTING 回放支持原生 `>1x` 杠杆，准入限 `DQ-A/B`；缺少证据时仍封顶 `1x`。杠杆固定在配置/episode 中，活动 episode 只能在空仓、无挂单的决策屏障调用 `env.set_leverage()`，该操作以当前权益作为新 session 起始余额，在相同市场时点重建引擎并更新风险上限。Nautilus 内置 leveraged 模型会同时缩放初始与维持保证金，造成高倍数边界与真实 dated 维持率失真；现改用 `StandardMarginModel`，将初始率设为 `dated_initial_rate / leverage`、维持率保持 dated rule 原值，并拒绝初始率低于维持率的组合。因此可支持 2x 以上，但上限由规则两项保证金率共同决定。3x session rebuild、实际线性持仓维持保证金、replay v3 与 checkpoint 恢复回归通过。Nautilus 2.0.0rc6 每进程只允许一个活动引擎，因此切换时先关闭旧引擎；若新 session 创建失败，episode 关闭并需 reset。此实现不认证真实交易所分档保证金经济。仍需独立设计双向持仓账本及真实保证金、维持保证金、Mark 强平和 Funding/撮合金标准；Hedge Mode 保持未认证。

## 跨轮次续做规则

1. 每次开始先读本文、当前目标入口和最近的执行记录，检查 `git status` 与相关源码差异。当前 HEAD 为审计基线，但大量源码修改尚未提交；保留这些修改，不 reset、不覆盖他人工作，不把 HEAD 当作当前实现的完整身份。
2. 从续做游标挑选首个依赖满足的任务；先核对已完成部分，再实施最小必要修改与有意义的回归。已通过的模块只做本次变更需要的检查，不重开全仓审计、重复造哈希体系或做无关格式化。
3. 工程修复、确定性原生样例和正式策略研究分开记账。涉及历史行情的实验先冻结代码差异、数据/规则身份、窗口、成本、种子、预算、主指标和停止线；只有项目登记允许的窗口可读。已消费数据继续保留其使用状态，换目录、协议或快照不得恢复新鲜性。
4. 必要检查通过后自动推进下一项；只有实际需要新增授权、缺少无法推断的信息或前置条件未满足时才停下相关分支。Test/Holdout 仍受既有一次性访问策略和冻结协议约束，不能因“连续开发”越过数据门禁。
5. 对外部证据不足、执行失败和候选被拒分别记录 `BLOCKED`、`FAILED`、`REJECTED`，写清依赖和可继续项。相同修复连续两轮无效时保存复现与原因，换可推进的工作包，不扩大同一失败候选预算。
6. 每完成工作包或结束一轮，在 [执行记录](DEVELOPMENT_PLAN_20261017.md) 追加结果；同步本文队列状态与 [当前入口](DEVELOPMENT_GOAL_CURRENT.md) 的游标。未完成就列出剩余动作和精确续做位置，不把计划中的命令写作已执行。

每轮记录最少包含：`任务 ID / 状态`、`修改文件与行为`、`命令与实际结果`、`证据路径`、`行情窗口及访问用途（无则写无）`、`阻塞与剩余边界`、`下一任务和起点`。任务可拆为“工程已完成 / 研究未通过 / 平台待验收”，避免一个总状态掩盖差异。大工件已有持久清单即可；新的关键小型摘要和运行器纳入 `docs/audit_evidence/`、`configs/`、`qa/`，不让唯一证据只留在 `.tmp/`。

## Codex 连续执行协议

收到继续开发指令后按以下顺序工作：

1. 阅读本文的续做游标、对应任务规格和 `DEVELOPMENT_PLAN_20261017.md` 最近记录；检查 `git status`、目标源码、相关测试和已有证据，辨别已完成修改与真正缺口。
2. 从续做游标开始，选择第一个依赖满足且仍有工程动作的任务。不得因某项策略研究、远程 CI、历史规则或真实深度阻塞而停下不依赖它的源码工作。
3. 直接实施最小必要改动，补有意义的回归；运行受影响测试及适用 Ruff/MyPy。涉及运行数据时先确认登记、冻结配置、数据/规则/模型哈希、窗口、用途和资源停止线。
4. 验收通过后，把任务状态推进并自动选择下一项；验收失败则保存真实错误和复现信息，标记 `FAILED` 或 `BLOCKED`，然后继续不依赖该项的工作。实验结果未过门仍可作为已完成的诊断结论，但不得写成收益改进。
5. 每轮在开发计划追加真实命令、实际结果、证据位置、数据访问用途、边界与下一游标；同步当前入口。不得只生成建议、重复已通过的全部检查、覆盖旧证据、虚报测试或将 Train-inner / scenario-only 结果说成正式验证。

此协议是 Codex 活动目标的执行说明。用户要求“按目标继续开发”时，应启动续做游标中的工程工作并持续推进；单纯要求编辑目标文档时，只修改目标本身。任何研究执行仍须满足已登记的数据用途与访问门禁。

## 检查命令与交付

以下是各工作包可用的检查命令参考，不代表已执行。先核实项目锁定依赖和当前源码环境；依赖未装好时单列环境问题，不以零收集测试或跳过代替通过。M0.1 优先跑前两个指标/清算测试文件，涉及历史工厂再扩展第三个；每项实际命令与结果追加至开发计划。

```powershell
uv run --locked --no-sync python -m pytest -q tests/test_research_metrics.py tests/test_native_liquidation_facts.py tests/test_historical_factory.py
uv run --locked --no-sync python -m pytest -q tests/test_es_dimension.py tests/test_es_fitness.py tests/test_es_checkpoint_resume.py
uv run --locked --no-sync python -m pytest -q tests/test_research_access.py tests/test_research_wfo.py
uv run --locked --no-sync python -m ruff check packages tests qa ci
uv run --locked --no-sync python -m mypy packages qa
git diff --check
```

按影响选择定向集成与全量测试，避免每个小修改都反复跑全部检查。平台交付对照 `.github/workflows/l6-ci-reusable.yml` 的实际 Linux/Windows Jobs；CI 尚未运行或失败就如实保留状态。收到继续开发指令后，按本文件协议实施对应源码任务并把实际结果追加到开发计划；不能以维护文档代替源码工作。正式 Test/Holdout、发布或其它未获授权的动作仍须服从各自门禁。

## 保留的交付与能力目标

原目标中的协调器/learner 资源边界、硬退出恢复、必要 GPU 路线、干净环境复现与发布候选验收继续保留，细节沿用 [资源恢复目标](DEVELOPMENT_RESOURCE_RECOVERY_20261013.md) 和 [2026-10-16 开发计划](DEVELOPMENT_PLAN_20261016.md)。R1-learning v5 现按训练块记录 CPU RSS 与 CUDA allocator 的 allocated/reserved 峰值，实际 CUDA smoke 已验证数值关系；这不构成整机 VRAM 使用测量或硬件级显存上限。N04 的 Windows Job 仍只硬限 CPU 提交内存，其他平台及 VRAM 强隔离继续保留未认证边界。

最终交付须能从干净源码与可分发小样本复现基础链路，给出当前实现、平台、研究与能力边界的独立状态。正式经济验收若因历史规则或独立样本阻塞，允许交付明确边界的工程成果，仍保持收益认证未完成。未经经济与语义门禁，不扩大模型/ES 搜索、不开放新 Holdout、不开展多币种或高杠杆主线。

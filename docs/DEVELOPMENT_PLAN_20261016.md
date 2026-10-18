# 2026-10-16 审计后的自动开发计划

依据：[审计报告](AUDIT_REPORT_20261016.md)。此计划替代旧目标文档的优先级；历史结果保留。执行中的修改叠加在已有未提交工作上，不把当前 HEAD 当作全部源码身份。

## 目标与执行顺序

| 顺序 | 目标 | 自动执行和验收 | 后继门槛 |
|---|---|---|---|
| A1 / F1–F2 | 可信数据访问 | 普通 CLI/API 未授权 Test/Holdout 在构建环境前拒绝；WFO 通过共享授权服务；项目 SQLite 按市场、数据族和半开时间区间查重，不依赖 output、seed、protocol 或快照 SHA；完成报告绑定收据；中断访问不退还 | 跨目录、扩展数据、部分重叠、恢复、压力模式回归通过 |
| A2 / F3 | 冻结预处理 | Scaler 构造时保存统计快照与 SHA，运行中做廉价结构比较；覆盖首次读 hash 前后、恢复和 transform 内部变更 | 不按分钟重新序列化全统计资料，旧未变 Scaler 的 SHA 不变 |
| A3 / F6 | 终端价值 | 所有强制清仓结算的 episode 视为有限期终止，终态价值为零；配置/transition/checkpoint 身份升级；解析 SB3 target 与单/并行环境回归 | 全量 CPU、Ruff、MyPy 通过后进入 B |
| B / F4–F5 | Train 内低成本信号基准 | 在已允许 Train 固定多个时间窗，比较空仓、固定多空、因果趋势/反转和当前冻结 RL；相同风险预算、Base/Stress 及费用，分别归因方向、择时、Funding | 没有跨窗基准优势则停止预算扩张；内部窗只称开发数据 |
| C / F4–F6 | 有限配对机制试验 | 最多 10 个配方：连续目标对照、显式零目标退出、5/15m 决策、1h/6h 折扣半衰期、特征消融；每次只改一个机制，3 seeds，晋级后至少 5 seeds | 不给交易次数奖励，不移除成本，不以训练 reward 或活跃度替代净收益 |
| D / F5–F7 | 覆盖、吞吐与恢复 | 仅晋级候选执行 100k → 300k → 覆盖充分的预算阶梯；先测 1/2/4 worker 与空仓/长持仓/调仓增长曲线；保留完整轨迹对照及检查点 | 下个 worker 预计突破约 1 GiB 可用物理内存底线前停止扩容；不杀其他进程 |
| E / F1–F2、F8 | 最终独立验收 | 冻结代码/模型/特征/动作/奖励/种子/选择准则/行情/规则有效期/费用/成交场景，再执行顺序 WFO 与一次性未消费 Holdout | 多窗/seed 净收益、回撤、风险和压力成本过预登记门；失败保留记录 |
| F / F9 | 可复现交付 | 配方、验证器、来源和小型摘要纳入 docs/configs/qa；大工件保存持久清单和校验；干净 checkout 复现 | 不把 .tmp 当作可交付依赖；源码备份、发布状态按实际证据记录 |

## 自动停止条件

1. 工程完整性未过 A：修复并复验，不开始策略筛选。
2. B/C 未见稳定优势：记录失败并转向数据、特征、周期机制，不执行 D/E，不无限调参。
3. 没有足够独立窗口或历史规则有效期证据：继续明示场景假设的 Train 研究；最终历史经济验收保持未完成。
4. 中断后缺少完成报告/访问收据：拒绝重放已启动 Test/Holdout，不能用新目录绕过。

## 研究边界

项目登记文件为 `.artifacts/research-access.sqlite`，与运行内 `research.sqlite` 并存。应备份项目登记；删除数据库、另建仓库或人为重命名市场不属于本地治理可防御的安全边界。不同快照版本不影响重叠检查。市场标识去除模拟 venue 和 `-PERP` 后映射；新增不同别名/数据族须先核实同一底层市场关系。

旧运行库的使用区间须先只读盘点并导入项目登记。缺少明确边界的旧访问保守隔离，不能当成未消费数据。已有模型可做明确标注的原 Train 诊断，不能因工程修复把旧 Holdout 重新视为新鲜验证。

## 本次执行结果

### 已实施的工程目标

- **F1/F2**：新增共享 `EvaluationAccess` 与项目时间区间登记。普通 API/CLI 的 Test/Holdout 在构建评价环境前拒绝无授权请求；明确声明 Validation 的独立回放标记为探索性，并登记开发使用。WFO 的策略与预登记固定基准分别消费一次性访问，完成报告 SHA 绑定项目收据；已开始却缺少报告的访问不退款、不重放。更换输出/协议/seed/快照仍检查区间重叠；已用开发区间不能跨实验改称新 Holdout。
- 已只读盘点 **10 个旧 WFO 数据库与 22 份已有独立报告**，生成 **67 条**保守访问记录，导入 `.artifacts/research-access.sqlite`。清单及 SHA 为 `audit_evidence/20261016/legacy_research_access_v2.json`。盘点只覆盖列明的已知运行根；没有账本/报告的历史访问无法从文件反推出，不声称历史记录穷尽。旧数据库未修改，未执行新的受保护评价。
- **F3**：Scaler 保存构造时统计快照与既有 SHA；读取 hash 和 transform 前做结构比较，L2 adapter 继续前后检查。缓存前后低层变更、pickle 恢复后变更、transform 内部变更均被回归覆盖。没有按分钟重复序列化所有统计量。Event/FeatureSpec/FeatureRow 的缓存保留：它们没有 Scaler 的可调用运行态变换接口，正常公开构造/赋值仍受冻结约束；此项不声称防御任意低层篡改。
- **F6**：清仓并结算费用的历史窗结束及 max_steps 结束返回 terminated=True、truncated=False；终态不再进行 TimeLimit bootstrap。终端语义进入训练和原生 transition 身份，旧检查点不能按新语义静默恢复。真实原生账户、固定 V=7 的解析 PPO 回报目标、1/2 环境及恢复路径已验证。
- **F4 的首项机制**：新增可选 `execution.zero_target_policy="close"`，精确零目标绕过普通死区、先取消未结挂单，再发出完整 Reduce-Only 退出；保留历史最小数量/名义额和明确声明的减仓豁免。默认 `deadband` 保持原身份与行为，作为对照。已覆盖多/空、错误部分退出、最小订单限制和挂单 ACK；本轮补齐五层动作量化归因（请求、软风险、硬风险、数量步长取整目标、引擎后的实际敞口），但尚未训练该动作变体或证明策略收益改善。
- **F7 的工程优化与验收**：历史工厂可复用已验证的 CuratedStore；compact Position 只读取公开标量，funding 按 `ts_last` 增量读取，指标行逐条累计；紧凑报告仍保留每分钟 snapshot、事实账本和精确收益序列。新增 1/30/90 日空仓/长仓/调仓原生增长曲线，详情见本计划后续 F7 记录。该建议剖析矩阵已完成；不代表常数内存、全轨迹恢复对照或最终 D 阶段 1/2/4 worker 扩容验收完成。

### 自动执行的 Train 信号筛查

预登记：[固定基准计划](../configs/train_signal_baselines_20261016.json)。冻结原 Stage62 seed102 候选，固定 2024-01-15、02-15、03-15 三个原 Train 日窗口；比较空仓、固定多、固定空、20m 因果趋势/反转及冻结 RL，各做 Base/Stress，共 **36 次**原生短回放。复用原 scaler，不训练或选参，不新增正式 Test/Holdout 评价。全部终态为空仓。

| Train 日窗口（UTC） | 冻结 RL Base / Stress | 固定多 Base / Stress | 固定空 Base / Stress |
|---|---:|---:|---:|
| 2024-01-15 | 0% / 0% | +0.1395% / +0.1215% | -0.1635% / -0.1816% |
| 2024-02-15 | 0% / 0% | +0.1578% / +0.1397% | -0.1819% / -0.2000% |
| 2024-03-15 | 0% / 0% | -0.4006% / -0.4181% | +0.3771% / +0.3595% |

趋势/反转在这三窗及两种成本下均为负。冻结 RL 六次均完全空仓，未严格超过空仓或各窗方向基准，自动门返回 `FAIL_NO_CROSS_WINDOW_BASELINE_ADVANTAGE`、`budget_expansion_allowed=false`。因此**停止扩大当前候选至 100k/300k、停止其正式 OOS 晋级**。这不是对所有 RL 或所有市场状态“没有信号”的证明；固定多空收益来自方向敞口，不能作为择时学习证据。各策略共享风险预算/上限，实际敞口并不相等，详情分别保留。

原始 36 份报告和失败尝试位于 `.artifacts/audit-development-20261016/`；可纳入版本管理的小型摘要为 `audit_evidence/20261016/train_signal_baselines_summary.json`。每份报告有 SHA，数值单位按账户净收益而非风险预算敞口解释。规则为明示情景假设，历史经济事实缺口见 [更新后的阻塞说明](HISTORICAL_RULES_BLOCKER_20261016.md)。

### 低容量预测诊断（B 阶段补充）

在预先登记并避开五条已消费 Test/Holdout 区间后，用三个不重叠的 180 日训练/30 日开发诊断折，比较固定 Ridge（α=0.1，5 个冻结 `legacy` 市场特征，折内训练均值/标准差）与零分、同周期动量/反转。预测目标为未来 5/15/60 分钟 log 价格变化；特征按 5 分钟采样，训练标签严格在测试窗口起点前结束。所有窗口在原 Train 边界内，并已登记为 development；数据质量均为 DQ-B，dataset hash 与冻结 bundle 相同。测试标签用于开发诊断，不是独立 OOS。

| 开发诊断折 | 5m Ridge 排序 IC | 15m Ridge 排序 IC | 60m Ridge 排序 IC | Ridge 多头 top 10% 扣情景成本后（5/15/60m） |
|---|---:|---:|---:|---:|
| 2025-02 | -0.0093 | -0.0197 | +0.0095 | -13.93 / -13.00 / -5.26 bp |
| 2025-09 | +0.0367 | +0.0003 | -0.0057 | -10.88 / -11.62 / -11.57 bp |
| 2026-04 | -0.0157 | +0.0143 | -0.0039 | -12.13 / -11.16 / -12.01 bp |

双边情景成本固定为 12bp（两侧各 5bp taker 费率和 1bp bar 滑点，均为 scenario-only）。九个 Ridge 多头 top-decile 结果、对应空头 bottom-decile 结果都未转正；IC 幅度小且不跨折稳定。反转预测排序 IC 在三折均为正，但 top-decile 成本后仍为负。没有据此调参、晋级策略或增加训练预算。更详细的 36 组模型/基准指标、每折报告及 SHA 在 `.artifacts/audit-development-20261016/low-capacity-predictor-v2/`；固定协议和小型可版本化结果见 `configs/low_capacity_predictor_20261016.json` 与 `audit_evidence/20261016/low_capacity_predictor_summary.json`。方案标签字段有一条纯文字公式勘误，未改变实现或结果，见相邻 `.errata.json`。

### F4 五层动作归因

原报表将“硬风险目标→引擎后敞口”的差额混在一起，无法单独看数量步长取整。现在历史工厂在调用相同 `TargetQuantityPlanner` 时保留其 `PlanDecision.target_qty`，作为数量取整后的目标敞口；Gym info、top-level `quantity_rounding_diagnostics` 和按 plan action/reason 分组的 `execution_attribution.quantity_rounding` 都分别报告风险调整目标、取整目标、二者差额，以及取整目标到下一决策屏障实际 native 敞口的差额。该实际敞口来自原生账户事实，已包含引擎执行结果；真实 fill 数量/费用继续由 native report 独立汇总。若任何一层缺失，coverage 标为 unknown，不将 `SUBMIT` 解释成成交。此诊断仅增加只读字段，不改变目标/计划/回放状态。

定向 adapter/rollout/historical 三组 **90 passed，28 warnings，91.66 秒**；随后全量 CPU 回归 **497 passed，46 warnings，285.76 秒**。Ruff 全仓通过，MyPy `packages qa` **117 source files** 通过，`git diff --check` 通过。真实历史 factory 回归断言 native action 数与数量归因样本数相等且各层均已知；合成多组报告回归核对了风险→取整及取整→实际敞口差额。市场/绩效数值未因本改动重算；旧回放档案不含新字段，不能用作五层分解。

### F4 显式零目标退出语义探针

按冻结配置 `configs/zero_target_exit_probe_20261016.json`，在同一 Train-inner 一日窗口、同一 bundle/行情/执行规则上，只切换 `execution.zero_target_policy`，运行 3 个配对 seed。动作序列固定为 step 0 请求 0.25、step 1–21 请求 0.05、之后请求 0；这是执行语义夹具，不是 RL 策略训练或收益实验。

三个 seed 的第 21 步实际残余敞口均为 **0.0495855**。第 22 步请求零时，`deadband` 均得到 `NOOP/deadband`，后续 60 步观察内没有主动清仓；`close` 均得到 `SUBMIT/policy_flat`，在同一第 22 步的 native 决策屏障后实际敞口为 0。三组配对均满足预登记语义条件，汇总与源码/计划 SHA 见 `audit_evidence/20261016/zero_target_exit_probe_summary.json`，完整逐例报告和 checksum 位于 `.artifacts/audit-development-20261016/zero-target-exit-v1/`。

两个臂最终都在 episode 结束时强制结算为空仓，且每个臂各有 3 笔成交；不能以终态、成交数或本探针的收益判断策略优劣。行情及经济规则属于此前已用 Train 与 scenario-only 假设，没有测试 Holdout 或执行参数选择。结论仅确认零目标退出契约差异；尚未证明 RL 能可靠地产生零动作，也没有证明加入该机制改善成本后收益。此前 F4 已实现的 `close` 仍为显式 opt-in，历史默认 `deadband` 不变。

### 验证与工件

- 最终全量回归：**493 passed、46 warnings，304.57 秒**。日志 `.artifacts/audit-development-20261016/pytest-final.log`。没有跳过项；不把这条总数单独解释成新的硬件或收益认证。
- Ruff `packages tests qa` 通过；MyPy 保留现有未标注函数体提示，源码检查通过。`git diff --check` 通过，保留既有 LF/CRLF 提示。
- 模型、行情完成标记、规则及其引用说明均已复制到持久目录；清单 `audit_evidence/20261016/persistent_artifacts_v2.json`。源码快照包含本轮及此前尚未提交修改，逐文件哈希用于定位实际代码，不以 HEAD 代替工作区。
- 本轮没有发布或推送 GitHub；F9 的 CPU 干净源快照复现已完成，但 CUDA/GPU 验收、独立新机器复现、基于当前策略的三/五 seed 学习机制试验、完整真实 fill 原因归因及最终独立盈利验收仍未完成。F7 建议的 1/30/90 日增长曲线已完成，但不等于 D 阶段 worker 扩展验收。

### F9 干净源快照复现

原 `source-snapshot.zip` 在本轮探针前生成且遗漏了仓库 `packaging/`；后续独立全量测试还发现两个被 `.gitignore` 排除的验收输入。`qa/package_development_evidence_20261016.py` 已支持版本化输出，v4 源快照扩大到完整源码/文档目录，并显式纳入真实 1 秒 quote 小样本（4.9 MB）和一个只供原生 liquidation 回归使用的 NautilusTrader 示例脚本。未修改 NautilusTrader 上游工作区文件。v4 ZIP 含 **343 项**，SHA-256 为 `38643fc3a03b2249a722a5236a2c45e5f323b04022ce4c0f67aae43f994426bb`；源码逐文件及工作树差异清单见 `audit_evidence/20261016/development_source_manifest_v4.json`。

在 `.artifacts/audit-development-20261016/clean-extract-v4/` 全新提取，`uv sync --locked --offline --all-packages --no-editable --extra application` 在独立 `.venv` 里安装 68 个锁定依赖并从提取源码构建 11 个 RLES 包；随后全仓 CPU 回归 **493 passed、4 CUDA skipped、45 warnings，279.32 秒，exit 0**。pytest 临时根位于 `%TEMP%`，避免本机对项目目录硬链接的限制。详细安装/命令、日志 SHA 和跳过项说明在 `audit_evidence/20261016/clean_source_reproduction_v1.json`。之后为把复现结果纳入快照，又生成最新 v5（347 项，SHA-256 `7ca9a9ba622d4cac3f6e1f425f3c8f7d284aead068c83163dc7134abc99d077b`）；v4→v5 的 `packages/ tests/ qa/ configs/ packaging/ ci/ .github/` 逐文件 SHA 完全一致，改动仅为 docs/evidence 记录，故全量测试证据适用于同一代码和测试源码。v5 没有另跑第二次全量测试。这证明当前 Windows/CPU、已有离线依赖缓存条件下的源码提取复现；尚非新机器联网复现，GPU 验收仍待执行。

### F8 官方经济资料边界复核

补查 Binance 官方 USDⓈ-M 文档与公告：`exchangeInfo` 是当前规则入口，没有历史时点查询参数；公开 `fundingRate` 可按日期读取历史 funding。2024-07-25 的 tick-size 公告给出一项带生效时间的具体调整，但对象是公告列明的合约，并不能建立 ETHUSDT 全时期过滤规则。`commissionRate` 是需 API key 的签名 USER_DATA 接口，文档描述当前用户费率且没有历史有效期参数。来源能力和剩余事实清单已记入 `audit_evidence/20261016/binance_economics_sources.json`，正文见 [历史经济规则阻塞说明](HISTORICAL_RULES_BLOCKER_20261016.md)。

这次核查让缺口更具体，但没有补齐完整历史过滤参数、账户费率阶梯、历史保证金/清算或延迟/成交校准；未读取账户数据、未调用需要凭据的接口、未把当前值倒填历史。F8 因而仍是最终历史经济验收阻塞，所有对应回放继续保留 scenario-only 标签。

后续复核官方 Binance 公告发现：ETHUSDT 永续合约最小名义价值从 5 USDT 调整为 20 USDT，公告发布于 2023-10-27，要求最迟于 2023-11-02 10:00 UTC 生效。它早于行情包起点 2023-12-30，说明情景文件的 20 USDT 有样本前官方调整记录可对照；但没有建立该值在整个样本期间持续有效。具体边界与来源已更新至 `HISTORICAL_RULES_BLOCKER_20261016.md`、`audit_evidence/20261016/binance_economics_sources.json`，后者 SHA-256 为 `7ad5312191e4f3bc1e7925a850ecb0939be4ce9926771091b9c18faf9f06cfbb`。没有据此改写规则事件或重算行情结果。

本次只改文档与来源证据，随后生成 `source-snapshot-v7.zip`（353 项，SHA-256 `ef265e8f32eb28c691374c5bca7a18a1124fcff164f54b51a3d33ddcd0c0bb42`）。ZIP 完整性检查通过，包含更新后的阻塞说明和证据 JSON；`packages/`、`tests/`、`qa/`、`configs/`、`packaging/`、`ci/`、`.github/` 相对 v6 逐文件 SHA 相同，故 v4 的 CPU 全量回归仍覆盖本次快照所含代码和测试。无需因纯文档更新重跑全量测试。

### 下一执行目标

当前旧 RL 候选和固定 Ridge 均未证明跨窗、扣情景成本后的优势；停止扩大当前预算，也不进入 D/E。B 阶段的低成本筛查已覆盖空仓/方向基准/动量反转/RL及低容量预测诊断。除上述零目标语义 probe 外，进一步核验了三段冻结 Train 窗在 Base/Stress 下的 6 份 RL 报告；每份 1,437 个动作的 `requested_nonzero_fraction=1.0`，且报告 SHA 与基准摘要一致。因此现有连续策略从未请求精确零，`zero_target_policy=close` 对此冻结候选的动作轨迹没有触发机会；停止将它作为当前策略的独立训练候选。语义功能仍保留为未来动作契约的 opt-in 能力。后续若继续 C，应先预登记一个能实际表达 hold/flat/rebalance 的动作契约或独立决策周期实验，且每次只改一个机制、至少 3 个相同 seed；不在同一诊断折上反复选模，也不得把固定动作语义探针冒充训练效果。决策周期试验仍须逐分钟推进 Funding、风险和权益。没有新独立窗口和带有效期的经济规则前，策略研究仅作 Train development，E 阶段保持未完成。

### 后续工程续做：紧凑 Position 快照

检查 F7 长持仓热点后发现，紧凑模式此前仍调用 Nautilus `Position.to_dict()` 完整构造每笔成交、调整、佣金和交易 ID，再筛掉这些列表。现改为只读取当前观察所需的公开 Position 标量字段；普通非紧凑模式仍保留原始 `to_dict()` 行为。真实 rc6 原生 Position 回归确认紧凑行与完整字典中全部可用标量字段相等，且事实读取能直接返回同一记录。`buy_qty` / `sell_qty` 只存在于完整历史派生字典、没有 Position 标量 getter，也不参与观测或计量，因此不再在紧凑模式构造它们。

验证：Nautilus adapter smoke、原生训练 episode、原生清算事实合计 **55 passed，16 warnings**；Ruff 针对修改文件通过，MyPy 修改源码通过。先前一次不同文件顺序的合并运行有一项输出捕获失败，但该测试单独通过，反向顺序的复跑全通过。该改动消除每分钟完整 Position 历史反复序列化；随后继续优化 funding 增量和指标中间数据，并完成端到端增长曲线（见下节）。

继续处理账户指标中间数据：新增 `AccountFactsAccumulator` 逐行校验与累计账户事实。`native_report(include_account_rows=False)` 不再把全部账户行和归一化副本同时留在内存；显式请求账户明细仍返回相同结构。精确 VaR/CVaR 仍保留收益序列，且 factory 快照与 native funding/fill 事实账本也仍占用随样本增长的空间，因此只是消除一组重复行字典，不是常数内存实现。`test_research_metrics.py`、历史工厂和 golden report 合计 **49 passed，18 warnings**；Ruff、MyPy 修改模块、`git diff --check` 通过。首次在工作区 `.tmp` 执行时 Windows 禁止 Parquet 暂存文件硬链接，历史工厂集因此出现环境目录失败；把 pytest 临时根放到系统 `%TEMP%` 后，同集通过。

继续处理 compact reader 的 funding 扫描：Nautilus Position 暴露 `ts_last`，在冻结的 rc6 原生测试中资金调整会推进该值，而每分钟静止行情不会推进。compact reader 现在按 `(position_id, ts_opened)` 记住已消费的 `ts_last`，仅变化时枚举该 Position 的 adjustments；首次读取归档快照以支持中途挂接，event ID 冲突仍拒绝，且只返回本次新增的资金调整。HistoryFactory 已按事件 ID 汇总这些增量；非 compact reader 仍输出旧的 cache 全量快照语义。真实正/负 funding long/short 用例、compact/non-compact 报告完全一致回归、未变化 Position 不重复扫描单元回归通过。该策略依赖引擎按完整时间戳批次读取，以及每次 Position 调整推进 `ts_last` 的 rc6 行为；未来更换 Nautilus 版本须重验此行为。

指标与 funding 读取模块变更后重新跑全量 CPU 回归，使用系统 `%TEMP%` 下全新目录：**497 passed，46 warnings，306.32 秒**；Ruff `packages tests qa` 通过，MyPy `packages qa` **114 files** 通过，`git diff --check` 通过。MyPy 保留未标注函数体提示。全量通过验证了指标累计器和 funding 增量在应用/WFO调用链中的兼容性；不代表常数内存、端到端增长曲线、研究收益或历史经济事实验收已经完成。

### F7 端到端长窗增长曲线

新增 `qa/profile_native_history_growth_20261016.py`，按冻结 Stage62 seed102 bundle、同一经过 hash 验证的 Train-inner CuratedStore 与 base 执行场景，跑空仓、固定 0.25 多仓、每 360 步在 0.25/0.50 间切换三种动作路径；窗口从 2024-01-15 起分别为 1/30/90 日。每窗每路径均记录 rollout 步数、快照/成交/funding 数、耗时、Windows 进程 RSS 采样峰值、空闲物理内存；无拟合、选参、策略晋级或 Test/Holdout 查询。

| 窗口 | 决策步数 | 三种路径耗时范围 | RSS 峰值相对各自起始值增量范围 | 成交事实（空仓/长仓/调仓） |
|---|---:|---:|---:|---:|
| 1 日 | 1,437 | 2.19–2.58 秒 | 33.7–55.2 MiB | 0 / 3 / 9 |
| 30 日 | 43,197 | 17.22–21.56 秒 | 36.3–161.2 MiB | 0 / 3 / 241 |
| 90 日 | 129,597 | 48.30–65.60 秒 | 241.5–361.4 MiB | 0 / 5 / 721 |

每例有 1,440 / 43,200 / 129,600 个分钟快照（含启动/收尾边界）；funding 事实分别为 0 / 4、0 / 178、0 / 538（空仓/持仓路径；调仓与持仓相同），九例均终态平仓。验证源含 **4,403,867 个事件**，dataset hash `258834aba27e77b0c97fbb4a5ca9fc6c18699f2f8d0d56c77812d1946bbc1adb`；全程可用物理内存最低约 **13.31 GiB**，高于 2 GiB 停止线。报告和每例 SHA 位于 `.artifacts/audit-development-20261016/f7-growth-v2/`。另保留首次运行的 Windows ctypes 句柄采样失败记录 `f7-growth-v1/FAILED.json`；明确句柄类型后通过单测并在全新 v2 工件目录完成全部九例。

RSS 是每 50ms 采样的当前工作集，不是隔离分配归因或严格峰值；案例起始工作集有明显波动，故跨案例增量只作近似比较。绝对进程工作集约 5.7–6.6 GiB，主要包含复用的 440 万事件对象；不能从本次用例单独推断每分钟快照的精确字节成本。结果表明 90 日三个路径均可在当前资源底线内完成，长仓与调仓回放时间高于空仓；没有据此声称常数内存、收益或历史费用有效。F7 的建议长窗对照已执行，仍保留全分钟快照/动作轨迹、精确风险收益序列及 fill/funding 账本的线性存储成本。

复现：`.venv/Scripts/python.exe qa/profile_native_history_growth_20261016.py --plan configs/train_signal_baselines_20261016.json --output .artifacts/audit-development-20261016/NEW-F7-PROFILE-DIR`（输出目录须全新；要求本地持久 bundle、行情、规则工件）。本脚本 Ruff 通过；全项目 Ruff、MyPy `packages qa` **115 source files** 通过，`git diff --check` 通过。

复现命令（在仓库根目录执行，输出目录必须全新）：

```powershell
.venv/Scripts/python.exe -m pytest -q --disable-warnings --basetemp=.tmp/NEW-UNUSED-PYTEST-DIRECTORY --tb=short
.venv/Scripts/python.exe -m ruff check packages tests qa
.venv/Scripts/python.exe -m mypy packages qa
.venv/Scripts/python.exe qa/train_signal_baselines.py --plan configs/train_signal_baselines_20261016.json --output .artifacts/NEW-TRAIN-BASELINES-OUTPUT
```

最后一条需要持久工件清单中的本地数据与模型；新机器应先恢复这些工件并调整计划的 archive 路径，不能把本机路径存在当作干净 checkout 复现已经通过。

2026-10-17 收到更新的源码审计后，整改优先级与逐项状态续记于 [DEVELOPMENT_PLAN_20261017.md](DEVELOPMENT_PLAN_20261017.md)。同日完成的六次 gamma 配对训练与 36 份原生诊断未通过跨窗口优势门禁，停止该方向预算扩张；详情和原始摘要见新计划。

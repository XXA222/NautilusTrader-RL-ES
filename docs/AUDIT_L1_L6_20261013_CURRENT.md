# Nautilus-RLES L1–L6 当前状态与全面审计报告

- 审计截止：2026-10-13（Asia/Shanghai）。
- 审计范围：当前工作区源码、CPU/CUDA 测试与静态检查、原生历史回测应用、真实归档 WFO 实跑、现有场景输出和发布 wheel。
- 需求依据：用户提供的 v0.4.0 总体设计和 L1–L6 工作包。附件中的开发指令作为设计材料核对，不视为本轮额外发布授权。
- 审计方式：源码审查、复跑旧报告探针、全量 pytest、MyPy/Ruff、依赖检查、当前 wheel 与源代码比较、检查真实市场档案的场景回测产物。
- 结论边界：当前有真实 Binance ETHUSDT 市场档案驱动的集成场景结果，但 instrument rules、费率/资金费和执行成本明确使用场景假设；不得把它们称作历史规则回测、策略有效性证明或生产验收。

## 1. 总体结论

### 后续开发状态补充（2026-10-13）

本文下方的安装、输入域、恢复和资源缺口是各次审计时点的记录。当前开发状态以 `docs/DEVELOPMENT_GOAL_CURRENT.md` 为准：独立 CPU 核心安装已完成；硬风控减仓、ES 输入尺度、Train-only 校准、Canonical→PPO 映射、安装布局 WFO 身份和 Validation 检查点恢复已修复。Windows ES worker 已实现提交内存硬限制及真实原生强杀清理，最新相关安装包集成回归 **108 passed**，并非最新全仓测试数字。协调器/learner 资源、硬退出后显式租约恢复、历史经济事实、新窗口多折策略准入及统一 GPU/发布验收仍未完成。完整项目验收结论继续为未通过。

**L1–L6 的主要源码已集成，且目前 CPU 全量测试通过；项目已具备配置驱动的训练、历史回测和 WFO 应用代码。仍未完成完整设计验收，也不能宣布正式真实历史交易研究已通过。**

### 最新审计快照 · 2026-10-13（WFO Validation/Test 阶段恢复、真实盘口回放、A4 恢复、ES 代际恢复前置保护）

本节取代本文较早段落中的“当前全量测试”和最新 wheel 数字；其余日期段落保留为历史记录。

- 新增 2024-01-06/07/08 Binance ETHUSDT 单日可移植夹具，各自保留 K 线、mark、index 与 funding 原始 ZIP、相对路径导入清单、ZIP SHA-256、curated parquet 和完成收据。与此前 Jan 1–5 夹具合计，真实行情应用 CLI 现有 **8 个独立流程**：A0 PPO/SAC/TD3/DDPG 分别用 Jan 1/6/7/8，A1 ES 用 Jan 2，A4 E1→R1→E2 用 Jan 3，A2 R1→E2 用 Jan 4，A3 E1→R1 用 Jan 5。每个用例都执行配置驱动 `train`、bundle/hash 检查和独立 `backtest` Test；覆盖五条 A0–A4 路线和四种 SB3 算法。所有八条均为 DQ-B、零清算、终态平仓，资金结算覆盖及 `fundingRate` 来源可审计。市场数据是真实 Binance 归档，规则、费用、保证金与滑点仍为 scenario-only；这只是流程证据，不是历史经济模型或收益证明。
- 真实 A4 恢复矩阵分别在 E1 第一代、R1 第 32 步和 E2 第一代的持久 checkpoint 后注入中断；三条都恢复并完成 E1→R1→E2，bundle 阶段顺序、E1 第二代、R1 第 64 步和 E2 第二代完成标记均验证通过。每次均使用 Jan 3 真实 Binance 行情执行独立 Jan 3 Test 的 Nautilus 原生回测，DQ-B、资金结算覆盖、0 清算、终态平仓。恢复与回测都不是合成行情。
- ES 恢复现于资源分配前对照最新完整 generation checkpoint 的 evaluations×episodes 与 L6 持久请求数。若崩溃发生在新 generation 的请求已预留而 checkpoint 尚未提交，系统明确拒绝在同一 attempt 重放，避免双重执行已计费评估；Native Engine/L6 回归确认没有 worker/quota lease 被启动。完整代之间的恢复仍通过并保持候选流一致。
- 新增真实 Binance `bookTicker` 驱动的 Nautilus quote fill 回归：[test_real_golden_ethusdt.py](<D:/Program Files/NautilusTrader_RL_ES/tests/test_real_golden_ethusdt.py>)。测试验证 2024-01-01 原始 quote ZIP SHA `8b0ba775…e98eb2f` 和合并行情 SHA `d4bfb692…466b68b`，用 1 秒采样的真实 bid/ask 与显示数量回放 477 个决策步；两笔原生盘口成交均逐笔断言成交价等于成交时实际 ask/bid、成交量不超过该报价显示数量。结果为 DQ-B、0 清算、账户 10,000.37901426 USDT、终态平仓。规则/费用仍为 scenario-only，top-of-book 不代表完整订单簿或队列成交证明。已有本地配置驱动真实 quote PPO、ES、A4 Hybrid 训练/工件导出和独立回测产物。新增断言在源码及更新 wheel 环境各定向通过 1 项。
- WFO 的 Validation 候选全部不满足冻结选择门时，runner 现先写入哈希校验的 `wfo-report.json`，包含折清单、聚合候选及拒绝原因、逐个 Validation 报告路径/hash/核心指标、账本证据，并明确 Test 为空、最终 retrain 未运行、Holdout 保持 LOCKED；随后以 `NoValidationCandidate` 将账本置为 ABORTED，不会为了产出报告而开启 Holdout。新增 WFO 控制流回归使用隔离 fold/训练依赖验证报告 sidecar、两条 Validation 查询、0 Test/Holdout 和完整 audit chain；该测试只证明报告与账本逻辑，不作为真实行情或策略表现证据。
- WFO 增加验证阶段边界恢复：`rles wfo --resume --config ...` 仅允许恢复仍在 DEVELOPMENT/SUSPENDED 阶段的同一协议；[wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py>) 将配置、源码、数据集、特征及折清单绑定到 `wfo-run.json`，复用完成回执与 SHA 匹配的 Validation bundle/report，不重复训练或登记预算。KeyboardInterrupt 会暂停而非终结协议；未完成但已预留的查询不重放，会明确拒绝并要求新协议尝试。恢复定向测试使用隔离 fold/训练替身，仅证明持久化、去重和安全拒绝控制流，尚未在长窗口真实 Binance WFO 上注入进程中断。ledger 状态/查询恢复位于 [ledger.py](<D:/Program Files/NautilusTrader_RL_ES/packages/research/src/rles_research/ledger.py>)；CLI 参数位于 [application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/application.py>)。
- WFO 增加 fold Test 与最终阶段持久恢复：[wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py>) 会复用哈希匹配的 Validation/Test、已完成 final-retrain bundle 和已持久化 Holdout 报告；[ledger.py](<D:/Program Files/NautilusTrader_RL_ES/packages/research/src/rles_research/ledger.py>) 现支持 RETRAINED/HOLDOUT_OPENED 边界暂停恢复，并把 Holdout report SHA 写入审计链。若 Holdout 已打开但尚无完整报告，恢复会拒绝重放并将协议失败关闭；若最终账本已 CLOSED 但总 WFO 报告尚未写出，会根据持久 Holdout 和既有 fold 产物重建总报告。故障注入验证了 final-retrain 完成后中断、Holdout 打开但无报告时拒绝重放、Holdout 报告入账后恢复关账，以及 CLOSED 后重建/复用总报告；测试使用隔离 fold 与替身训练/回测依赖，只证明应用恢复控制流，不是真实行情进程中断或策略表现证据。final-retrain 内部尚未抵达可恢复 checkpoint 的中断仍失败关闭，需要新协议尝试。
- 补上 final-retrain 内已有完整 stage checkpoint 时的 WFO 自动续跑：[wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py>) 根据 A0–A4 路线，从最新的完整 R1 step 或 E1/E2 generation checkpoint 恢复；checkpoint 完整性、冻结 run recipe 和 ledger 仍由现有应用训练器复核。A0/A3 等 R1 已达完整训练预算但 actor 导出中断时，[application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/application.py>) 允许从完整 R1 checkpoint 仅完成 actor/bundle 收尾，并对已有 actor 做 parity 校验。WFO 故障注入验证 checkpoint 路径传递及恢复后不重复计训练；应用级真实小型训练夹具验证 full-budget R1 不再训练即可 finalize。未产生完整 checkpoint 的 RL 步骤内中断、ES 代内预留多于最新 checkpoint 的中断，仍失败关闭，不重放已计费请求。
- 新增 86,400 步长窗 WFO 完整训练验证运行：[配置](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/config-wfo-real-fullpass-60-15-15-v1.json>)、[WFO 报告](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-fullpass-60-15-15-v1/run/wfo-report.json>)、[研究账本](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-fullpass-60-15-15-v1/run/research.sqlite>)。PPO 与 SAC 都完成 86,400 个训练步，归一化计数为 86,401.0001；两个候选各完成 21,598 个 Validation 收益观测，报告均为 DQ-B、0 清算、终态平仓。PPO Validation 净收益 -1.8935%、最大回撤 2.7195%、换手 0.3048；SAC 净收益 -25.5178%、最大回撤 25.5196%、换手 492.3810、手续费 2,131.65 USDT、滑点成本 431.98 USDT，并触发 1,106 次硬风控调整。两者均因违反冻结选择约束被拒绝，`fold_tests=[]`、最终 retrain 未运行、Holdout 为 `LOCKED`；账本为 `ABORTED`，仅 2 条 Validation query、无 Test/Holdout query，audit chain 有效。WFO 报告 SHA 与 sidecar 一致（`1a57fa7d…96bf3076`）。重要限制：本次 Validation 日期与此前协议已用过的 2024-12-08 至 2024-12-23 窗口重合，且候选/配置在该窗口结果已知后选择，因此属于完整训练预算的探索性复核，不是独立新鲜验证；不得据此挑选参数后再把同段收益称为客观验证。行情窗口 instrument rules、费用、资金费、保证金及执行仍是 scenario-only。
- 增补完整 Train 段的真实盘口闭环：基于 Binance ETHUSDT 2024-01-01 K 线、mark、index、funding 与秒级 `bookTicker`（报价规则仍为 scenario-only），统一按 00:05–08:05 Train、08:05–12:05 Validation、12:05–20:05 独立 Test 执行。PPO/SAC/TD3/DDPG 和 A4 的 R1 使用 480 个训练步；bundle 内 R1 checkpoint 为 step 480，VecNormalize count 为 481.0001，证明模型观察覆盖了该 Train 段约 477 个可评分分钟步。独立 ES 的 E1 使用该完整 Train episode；A4 的 E1/E2 各完成 4 次评估、1 代，R1 使用 480 步。六条 Test 均来自同一真实行情快照 `c19e2670c2c3f7950e9cb3f224b3fc03e9f959db4c5c7bb9525b5378413d6767`，DQ-B、重复/冲突均为 0、478 个收益观测、0 清算且终态平仓；原生 backtest CLI 的 COMPLETED 收据将 bundle 与 report SHA 绑定。结果如下，均为单日情景证据，不表示策略通过：

| 路线 | Test 净收益 | 最大回撤 | 换手比 | Test 收据 |
|---|---:|---:|---:|---|
| A0 PPO | -0.2957% | 0.2957% | 5.7278 | [COMPLETED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-bookticker-full-train-closure-20261013/ppo-test/COMPLETED.json>) |
| A0 SAC | -0.1334% | 0.1370% | 3.0618 | [COMPLETED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-bookticker-full-train-closure-20261013/sac-test/COMPLETED.json>) |
| A0 TD3 | -0.1867% | 0.2025% | 4.8933 | [COMPLETED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-bookticker-full-train-closure-20261013/td3-test/COMPLETED.json>) |
| A0 DDPG | -0.1768% | 0.1946% | 5.0979 | [COMPLETED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-bookticker-full-train-closure-20261013/ddpg-test/COMPLETED.json>) |
| A1 ES | -0.0195% | 0.0222% | 0.0223 | [COMPLETED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-bookticker-full-train-closure-20261013/es-test/COMPLETED.json>) |
| A4 E1→R1→E2 | -0.0081% | 0.0089% | 0.0380 | [COMPLETED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-bookticker-full-train-closure-20261013/a4-test/COMPLETED.json>) |

六条的 Validation 与 Test 净收益均为负。盘口是秒级抽样 best quote，不能代表完整深度、队列优先级或成交概率；手续费、instrument rules、保证金、延迟与其他执行成本仍是显式情景参数。

- 该 Test 暴露终态小额余仓被最低名义价值挡住。`packages/execution/src/rles_execution/planner.py`、`packages/data/src/rles_data/schemas.py` 与 `packages/gym_bridge/src/rles_gym_bridge/historical.py` 现支持每条 instrument rule 显式声明 `reduce_only_min_notional_exempt`。planner 仅当该能力为真、平仓数量仍满足 minQty 且数量精度合法时允许低于 minNotional 的终态退出；缺省仍为 false。Nautilus submitter 已将终态退出单标记为 Reduce-Only。Binance 官方 USDⓈ-M 于 2021 公布的最低名义规则注明 Reduce-Only 不受限制，2023 公告把 ETHUSDT 永续最低名义价值调整为 20 USDT：[2021 Reduce-Only 例外](https://www.binance.com/en-AE/support/announcement/detail/76719bbaeeb847bbac4daa2906fcdcc0)、[2023 ETHUSDT 调整](https://www.binance.com/en/support/announcement/detail/e4384cba297a4bd2a154be644d5d76f9)。夹具整体 instrument rules 仍标记 scenario-only，因为费用、保证金等字段没有完整历史账户依据；有官方来源的单项规则不会把这些场景假设变成历史事实。
- WFO checkpoint 续跑修改后，全量源码回归 **310 passed、4 CUDA-only skipped、45 warnings，248.94 秒**；`tests/test_research_ledger.py tests/test_research_wfo.py tests/test_historical_factory.py::test_application_finalizes_full_budget_r1_checkpoint_without_retraining` **28 passed**；全仓 Ruff 通过，MyPy `packages qa` 103 个源文件无问题。11-wheel 构建命令成功产出全部 11 个 wheel；wheel 文件继承 owner-only ACL，当前测试进程无法读取，`wheel_manifest.py` parity 检查和 wheel 重装未完成。上次已通过的 wheel 测试对应更早源码，不能替代本次源码验证；第三方依赖也尚未做到完全隔离安装。

仍未通过项目级设计验收：真实 ETHUSDT RL、ES 与 A4 Hybrid 的配置驱动训练/保存/独立 Nautilus 回测链路均有短窗真实归档产物；扩展到 86,400 步 PPO/SAC 训练后，重叠的既有 Validation 窗口仍无候选越过 0% 净收益选择门，因此没有执行 fold Test、最终 retrain 或开启 Holdout。该长窗 WFO 是探索性复核，不构成独立新鲜验证。秒级真实 top-of-book 回放不代表完整盘口深度、队列优先级或成交概率已建模。没有完整历史 instrument rules、账户费率等级、保证金/清算规则与实际执行成本核验，也没有据此证明策略盈利的结果。GPU/ES 目标设备组合、较长窗口训练/WFO 吞吐、ES 代内强杀恢复和完全隔离的第三方依赖安装仍需验收。

### 2026-10-13 本轮追加核验与修复

为继续验证修订后的 WFO 净收益选择门，本轮查找并复用了本机此前已通过 Binance 官方来源 SHA/ZIP 校验、但未进入上一份连续数据包的 2025-01-08 至 2025-01-23 官方归档。导入器现允许输出目录中仅有 `raw/` 的来源缓存暂存结构，以便中断后校验并继续导入；新增导入回归通过。K 线、mark、index 与月度 funding 共 98 个来源归档/收据文件完整，离线重验导入 69,218 个事件。与此前两段经校验的市场包合并后为 713,786 个事件，重复 0、冲突 0，市场 SHA-256 为 `0fd5cef7b15fa3bc6118373973a55f7d76f5eff1b5ca66b66fbced2026d5c0e9`；合并清单保留了 3 个来源 manifest 与行情文件哈希。该行情仍没有历史 instrument rules，执行/费用/滑点均为 scenario-only。

新开 WFO 协议 `real-ethusdt-wfo-oct-jan-2024-60-15-15-v3`，窗口 Train/Validation/Test 按 60/15/15 天并按特征回看插入 purge，最终 Holdout 固定为 2025-01-08 至 2025-01-23。先后两个起点因只预留 90 天、遗漏 purge 而在创建研究账本前被预检拒绝；第三个配置向前扩展约 16 小时后通过数据覆盖预检。冻结候选各完成一个真实 Binance 归档驱动的 Nautilus 原生 Validation：

| 候选 | Validation 净收益 | 最大回撤 | 换手比 | 清算 | 硬风控干预 | 终态平仓 |
|---|---:|---:|---:|---:|---:|---|
| PPO | -3.3852% | 3.4767% | 74.6311 | 0 | 0 | 是 |
| SAC | -3.2855% | 3.2929% | 63.3073 | 0 | 439 | 是 |

两个候选都低于冻结的 `min_median_fold_net_return=0` 选择门，故 WFO 按设计以 `WFO produced no feasible validation-selected candidate` 终止。研究账本状态 `ABORTED`，只有 2 条 Validation 查询、0 Test、0 Holdout；Holdout 未打开/未消费，audit hash chain 验证通过，`FAILED.json` 与两份 Validation 报告均在运行目录留存。该结果进一步表明在此情景成本与窗口下候选没有可入选的正收益证据；SAC 的 439 次硬风险干预还需要在下轮开发中解释及降低。此验证并非历史费率/规则意义的策略验收。

上一轮 CPU 全量回归 **282 passed、4 CUDA-only skipped、33 warnings，64.33 秒**；wheel 安装隔离环境全量 **282 passed、4 skipped、33 warnings，64.11 秒**；Ruff 全通过，MyPy 103 个源文件无错误。隔离环境复用已有第三方依赖，故只作为 RLES wheel 安装与运行证据。本轮更新见下方真实 golden 补充。

#### 2026-10-13 真实行情 golden 补充

新增 `tests/fixtures/real_ethusdt_20240101/` 可移植夹具，保留 2024-01-01 Binance Vision ETHUSDT K 线、mark、index、funding 原始 ZIP（SHA-256 均逐项校验）、导入收据、curated parquet 完成标记及显式 scenario-only 规则。`tests/test_real_golden_ethusdt.py` 实际构造 Nautilus 原生环境并回放到 08:00 UTC 资金费结算：DQ-B、477 个决策步、3 笔成交、账户末值 9,998.21338803 USDT、费用 0.49911263 USDT、滑点成本 0.10028 USDT、资金费 -0.15498934 USDT、0 清算、终态平仓。负号代表该窗口资金费支出。黄金值锁定引擎账户轨迹，不表示历史账户费率或真实交易盈利验证；规则收据明确标为 scenario-only。

`real_golden` 包含真实账户轨迹和五条真实应用路线，现已覆盖 A0–A4 各自独立 Test CLI 回放；五日归档均便携保存在测试夹具并校验原始 ZIP SHA。源码环境全量回归为 **293 passed、4 CUDA-only skipped、38 warnings，152.40 秒**；最新隔离 wheel 环境全量也为 **293 passed、4 skipped、38 warnings，152.28 秒**。wheel 环境借用已安装的第三方依赖，不是自包含离线依赖安装。pytest 默认临时目录在受限 AppData 路径报 PermissionError；指定项目内 `.tmp` basetemp 后回归通过。当前源码身份清单更新为 212 个文件，identity `cb0d73b549ec86aa76258cf68bf071a40cf812fd018bc74e15383143939b69a4`，文件 SHA-256 `9a7354e5d05824ffe0401b83e408f62ff2d5fbc1009a68afa9e5d9f96b930917`。固定预算 perf 标记仍没有测试，F14 尚未完全关闭。

随后针对审计 F09 扩展应用级恢复故障注入：A0/A2/A3/A4 的 R1、A1/A3/A4 的 E1，以及 A2/A4 的 E2 均覆盖从完成检查点恢复并完成剩余阶段/原生回测；E2 恢复还校验完整 R1 检查点和 actor hash。测试揭露并修复 A2(R1→E2) 未被 E2 resume allowlist 接受的问题。再增加 A2(R1→E2) 与 A3(E1→R1) 的真实 Binance 日包 CLI 闭环后，最新 CPU 全量为 **293 passed、4 CUDA-only skipped、38 warnings，152.40 秒**；Ruff 全目录通过，MyPy 103 个源码文件无错误。新增多阶段恢复仍是短窗 fixture 证据；A4 真实市场中断恢复及 ES generation 内中断恢复仍待验证/实现。

#### 2026-10-13 真实数据 RL/ES/Hybrid 配置闭环

使用同一真实 Binance 2024-01-01 ETHUSDT 行情包及 scenario-only 交易规则，按时间切分训练、验证和 Test，未随机拆分行情：Train 00:05–08:05、Validation 08:05–12:05、Test 12:05–20:05 UTC。所有工厂评为 DQ-B。

| 路线 | 真实执行 | Validation 净收益 / 最大回撤 | 独立 Test | 工件校验 | 终态 / 清算 |
|---|---|---:|---|---|---|
| A0 PPO | CPU R1 64 steps | -0.1685% / 0.1689% | -0.3303% / 0.3320% | bundle/report SHA 一致；独立 backtest CLI 从 bundle 加载 actor 后 report 与 sidecar SHA 一致 | 平仓 / 0 |
| A1 ES | CPU E1，4 个体 × 1 代 | -0.0109% / 0.0126% | 未跑 | bundle/report SHA 一致；actor kind 为 ES；账本记录 4 次评估 | 平仓 / 0 |
| A4 Hybrid | E1(4×1) → R1(64 steps) → E2(4×1) | -0.0031% / 0.0036% | 未跑 | bundle/report SHA 一致；阶段顺序、预算和 RL actor kind 均核对 | 平仓 / 0 |

三条路线的 `train` 命令均用真实市场数据经正式配置入口装载 L2/L3 工厂，运行 Nautilus 原生环境并自动生成可加载 bundle 与账户报告。A0 另用独立 `backtest` 命令回放时间上未重叠的 Test 区间；该单次短窗任务没有建立 WFO 的 one-shot holdout 账本，结果只用于验证工件加载与回测闭环。验证与 Test 均为负收益，证明管线执行完整，不代表策略通过。全部运行仍基于情景手续费/保证金/执行假设；结果不能声称为历史费用模型下的有效性结论。运行配置和原始输出保留在 `.tmp/real-ethusdt-next-closure-20261013/`。

复跑命令：

```powershell
.venv\Scripts\python.exe -m rles_orchestrator train --config .tmp\real-ethusdt-next-closure-20261013\config-a0-train.json
.venv\Scripts\python.exe -m rles_orchestrator backtest --config .tmp\real-ethusdt-next-closure-20261013\config-a0-test.json --bundle .tmp\real-ethusdt-next-closure-20261013\a0-ppo-run
.venv\Scripts\python.exe -m rles_orchestrator train --config .tmp\real-ethusdt-next-closure-20261013\config-a1-es.json
.venv\Scripts\python.exe -m rles_orchestrator train --config .tmp\real-ethusdt-next-closure-20261013\config-a4-hybrid.json
```

#### 2026-10-13 ES / Hybrid 独立 Test 回放

为不重复使用 Jan 1 PPO Test，从连续 Binance 官方归档包经 `rles_data.binance_archive.import_range` 的离线 raw-cache 路径重建 Jan 2、Jan 3 单日包。三个每日 ZIP 及 1 月 funding ZIP 均逐项与原导入清单 SHA 一致；两个包各 4,328 事件、包含 1,440 根 K 线/mark/index 与 4 个 funding update/settlement 事件。派生市场快照分别为 Jan 2 `7d11ddf3ce9c5cefa0170bb521598e5d29e44f76e606a3a4141c15cc02d8fb49`、Jan 3 `ba50861f5da61fdb38c20dcd9530a626b45928e490b91f529048a901f8714b3b`。

| 路线与日期 | Train / Validation / Test (UTC) | Validation 净收益 | 独立 Test 净收益 / 最大回撤 | Test 资金费 | 工件 / 终态 |
|---|---|---:|---:|---:|---|
| A1 ES · Jan 2 | 00:05–08:05 / 08:05–12:05 / 12:05–20:05 | +0.00185% | -0.02036% / 0.03288% | -0.09031006 USDT | genome bundle 与独立 backtest hash 匹配；0 清算、平仓 |
| A4 Hybrid · Jan 3 | 00:05–08:05 / 08:05–12:05 / 12:05–20:05 | +0.00913% | +0.00152% / 0.04237% | +0.00403348 USDT | E1→R1→E2 bundle 与独立 backtest hash 匹配；0 清算、平仓 |

复跑使用以下公共入口命令：

```powershell
.venv\Scripts\python.exe -m rles_orchestrator train --config .tmp\real-ethusdt-fresh-tests-20261013\config-a1-2024-01-02-train.json
.venv\Scripts\python.exe -m rles_orchestrator backtest --config .tmp\real-ethusdt-fresh-tests-20261013\config-a1-2024-01-02-test.json --bundle .tmp\real-ethusdt-fresh-tests-20261013\a1-2024-01-02-run
.venv\Scripts\python.exe -m rles_orchestrator train --config .tmp\real-ethusdt-fresh-tests-20261013\config-a4-2024-01-03-train.json
.venv\Scripts\python.exe -m rles_orchestrator backtest --config .tmp\real-ethusdt-fresh-tests-20261013\config-a4-2024-01-03-test.json --bundle .tmp\real-ethusdt-fresh-tests-20261013\a4-2024-01-03-run
```

这两段分别使用新的日行情和不同 Test 日，没有用 Test 指标调参或在二者间选优；两者仍是小预算单日流程检查，没有研究账本 one-shot Holdout 语义。Validation/Test 指标都受同一组显式 scenario-only 费率、保证金和 bar 滑点假设影响，不构成收益能力或历史执行事实证明。完整输出保留在 `.tmp/real-ethusdt-fresh-tests-20261013/`。

#### 2026-10-13 A2 / A3 真实归档路线补齐

为覆盖其余 RL/ES 组合路由，从先前验证过的 2023-12-30 至 2024-04-15 Binance 官方来源缓存派生 Jan 4、Jan 5 两个单日包。新增夹具保留真实 K 线、mark、index、monthly funding ZIP，测试逐项对照导入清单 SHA；Jan 4 有 4,326 个事件（3 组 funding update/settlement），Jan 5 有 4,328 个事件（4 组）。真实路线测试确认 DQ-B、funding 进入 Test 账户、无清算、平仓，以及 bundle 和 report 的 SHA/完成标记。

| 路线与日期 | 训练阶段 | 测试隔离 | 结果边界 |
|---|---|---|---|
| A2 R1→E2 · Jan 4 | 真实 Binance 日包；R1 后 E2 搜索 | 同日 12:05–20:05 UTC 独立 public `backtest` CLI | 流程、bundle 加载、funding、0 清算与平仓均验证；scenario-only 交易假设 |
| A3 E1→R1 · Jan 5 | 真实 Binance 日包；E1 后 R1 蒸馏/训练 | 同日 12:05–20:05 UTC 独立 public `backtest` CLI | 流程、bundle 加载、funding、0 清算与平仓均验证；scenario-only 交易假设 |

该验证补齐 A0–A4 五条编排路线的真实行情应用级 train→bundle→独立 Test CLI 覆盖，但各路线均为一天内的小预算管线验证，不构成策略收益能力、历史费率/规则或统计显著性结论。输出由 `tests/test_real_application_flow.py` 可重建；Jan 4/5 可移植数据包位于 `tests/fixtures/real_ethusdt_20240104/` 与 `tests/fixtures/real_ethusdt_20240105/`。


### 2026-10-12 本轮追加核验与修复

晚间追加复核发现 Binance 导入器会把每日 00:00 的 funding update/settlement 丢掉：kline 事件时间是首根 candle 的 close，通常为 00:01，而 funding 事件会在 00:00。`import_range` 现将 funding 下界按 UTC 日起点裁切，行情事件仍按首根 bar 裁切；边界回归通过。用本地已核验官方来源重建 2024-12-02 至 2025-01-07 区间，新增保留 2 个真实 funding 事件，总计 160,062 事件。与 2024-08-12 至 2024-12-01 官方导入合并并过滤为 2024-08-12 至 2024-12-31 后，得到 609,957 事件、0 重复、0 冲突；derived snapshot hash 为 `1dafd10b...`。Train、两折 Validation/Test 与 Holdout 的逐窗预检均为 DQ-B。继续请求 2025-01-08 后的数据时，Binance 官方来源在 TLS 握手阶段连续失败；未将不完整归档作为市场数据使用。

随后启动两折、PPO/SAC 候选的真实归档 WFO。流程在首折 PPO Validation 原生回放中运行约 19 分钟、工作集约 3.9 GiB，仍未提交候选结果；账本只有一条 Validation query，无 Test/Holdout 查询。Nautilus 仍逐批输出大量 INFO 日志，回放成本无法满足本次运行的审计时限，因此中断该次尝试。恢复时发现 Ctrl+C 未被异常清理分支捕获，协议留在 DEVELOPMENT；已将此运行明确登记 ABORTED、验证 audit chain，并写 `FAILED.json`。`run_wfo` 现对 `BaseException`（含 KeyboardInterrupt）执行终结清理，空异常详情使用固定说明；故障注入回归通过。该多折 WFO **未完成，不能用于策略/选择结论**；多折原生验证吞吐和日志成本列为待修复/验收事项。

将 `BacktestEngineConfig.logging` 显式设为 stdout ERROR、fileout OFF 后，真实两折重跑不再产生 Nautilus INFO 洪泛（日志文件 0 字节）。两折 4 个候选 Validation 均完成；每份 21,598 收益周期、终态平仓、0 清算，report/bundle sidecar SHA 均匹配。PPO fold0/fold1 净收益 **-4.711%/-5.266%**，最大回撤 4.794%/5.287%；SAC fold0/fold1 **-5.338%/-1.984%**，最大回撤 5.365%/2.001%。WFO 因无候选达到 0 净收益门槛而以 `WFO produced no feasible validation-selected candidate` 结束；ledger 为 ABORTED，4 条 Validation、0 条 Test、Holdout 未打开，audit chain 有效。全程约 **93 分钟**、工作集约 **3.8–4.0 GiB**。因此日志问题已压住，按分钟推进 native engine 的计算成本仍高；本次没有 WFO/Test/Holdout 策略结论。

最新源码完整 CPU 回归（含新增 KeyboardInterrupt 恢复测试）为 **280 passed、4 CUDA-only skipped、33 warnings**；另 `test_real_rc6_engine_streams_bars_and_reads_native_account_facts` 单测通过并确认没有 INFO 输出；Ruff 全包通过，MyPy 97 个源文件无错误。首次使用 pytest 默认临时路径的尝试因该位置不可写而报错；指定工作区 `.tmp` basetemp 后全量通过。

已成功导入 Binance 官方 ETHUSDT USDⓈ-M 归档的 2024-04-15 至 2024-07-31 数据：324 个日 K 线/标记价格/指数价格归档，加 4 个按月资金费归档，共 328 个 ZIP。328 个本地来源收据的 SHA-256 均与 ZIP 内容及导入清单相符；导入市场事件数为 467,206。导入器新增按归档原子写入的来源收据，可在网络中断后复用已校验文件，并增加瞬时 TLS/HTTP 错误重试；损坏文件会重新校验，不会仅凭收据跳过。

以该真实行情窗口运行的新单折 60/15/15 WFO 全流程 exit 0，数据集 hash 为 `1362fa92f466fa6dcfc16381b04d57ea63cfd1222b109a2301f3b513b7c53143`。PPO 在 Validation 被选中，Test 净收益 **-1.76%**、最大回撤 2.14%；B0 no-trade Test 收益为 0%，B1 always-long 为 -0.06%。一次性 Holdout 净收益 **-2.04%**、最大回撤 2.05%、换手率 30.79、0 次清算、结束平仓。报告 sidecar 与 WFO/Holdout 文件哈希匹配；账本状态 CLOSED，Validation 查询 2 次、Test 查询 1 次，Holdout OPEN/CLOSE 各 1 次。公开 `backtest` CLI 使用最终重训 bundle 重新回放同一 Holdout，exit 0，得到完全相同的净收益、最大回撤、换手率、清算数及终态。

该历史 WFO 的 `conclusion: PASS` 只通过了当时的回撤/换手/清算/风险配置门槛；它在 Test 与 Holdout 都亏损，并且 Test 逊于 B0。源码审查发现旧 WFO 选择器没有净收益下限，因此允许负收益候选被选择。现已修复：`SelectionSpec.min_median_fold_net_return` 默认 0，候选必须满足该 Validation 门槛；Holdout 的最终 PASS 还必须满足同一净收益下限，缺失/非有限收益失败关闭。**本次实际 WFO 使用的是修复前代码，且 Holdout 已消费；不能将修复后的门槛算作这次 WFO 已验证，也不会复用该 Holdout 重跑。**

净收益门槛随后在一段全新 Binance 官方归档窗口上做了端到端 Validation 验证。导入 2024-08-12 至 2024-12-01 的 341 个日/月 ZIP，共 484,506 个事件，341 个来源收据哈希与官方导入清单一致。8 月 12 日官方 mark/index ZIP 有 10:02、10:03 两根分钟缺失，严格 61 秒策略因此拒绝从 8 月 1 日开始的配置；将研究起点改为 8 月 13 日后，Train/Validation/Test/Holdout 覆盖均为 DQ-B，只有 conservative availability 标记，时间覆盖通过。

用明确标记 `scenario-only:not-historical-rule-evidence` 的 5 USDT 最小名义价值敏感性假设运行 WFO，PPO Validation 净收益 -3.04%，SAC 为 -5.14%，两者均结束平仓、0 清算。新选择器在 Test 前返回 `NO_FEASIBLE_MODEL`；账本只有 2 次 Validation 查询、无 Test 查询、未打开 Holdout，且现为 ABORTED、哈希审计链有效。该失败运行最初暴露 ledger 异常路径遗留 DEVELOPMENT 的问题；现已增加 `ResearchLedger.abort()` 和 WFO 自动 ABORT 记录。另一个明确使用原 20 USDT 场景门槛的运行在候选评估时因终端仓位低于最小名义价值而安全失败，没有据此宣称可平仓。5 USDT 只是敏感性假设，不能当作真实历史规则。

应用入口现将公开 `backtest` CLI 的结果持久化到配置的 output 目录：写入 `report.json`、SHA-256 sidecar 和绑定 bundle/report hash 的 `COMPLETED.json`；stdout 只返回路径、身份和摘要指标；已有 output 会被拒绝以免覆盖。已用 WFO 最终 PPO bundle 在真实 Binance Holdout 回放验证，CLI exit 0，净收益、回撤、换手、清算数和终态与 WFO 保存的 Holdout 报告逐项一致，报告 SHA 与完成标记相符。该回放仅用于验证 CLI 工件写入，不作为新的独立 Holdout 或策略证据；原始一次性研究结论仍是负收益。完整回测报告 10.5 MB，native 日志 55.0 MB。

本节早先记录的全量检查为 **277 passed、4 skipped、28 warnings**；本报告晚间异常恢复修复后的最新全量结果为 **280 passed、4 skipped、33 warnings**。CUDA 专用环境的 SB3 冒烟为 4 passed。设备解析/入口单测通过，且新增公开应用级 `rl_device` 与资源租约后，在真实 ETHUSDT 归档上完成四算法 A0 CUDA 与 A4 CUDA-R1 Hybrid 训练及原生回放。

新建公开应用 A0 CUDA 路径：`rl_device` 被解析进冻结 RL 配置，CUDA 不可用或没有 R1 的路由会在建环境/创建 run 目录前拒绝；R1 执行阶段使用 L6 quota 和独占 GPU owner lease，lease 在成功后释放。PPO/SAC/TD3/DDPG 四种算法均使用真实 Binance 2024-01-01 行情（含 1 秒抽样真实 bookTicker）完成 CUDA 训练、bundle 导出和 Nautilus 原生验证回放，四个配置驱动命令均 exit 0。报告各含 719 个收益观测、结束平仓、0 清算；净收益分别为 **-0.2488% / -0.0937% / -0.1817% / -0.1536%**。每个 bundle/report SHA 都与 sidecar 和 `COMPLETED.json` 匹配；所有运行结束后 SQLite lease 为空且 GPU owner 文件不存在。market provenance 是 Binance 官方归档，rule provenance 明确为 `scenario-only:not-historical-rule-evidence`，这些结果证明算法运行链路，不证明策略有效或历史执行规则正确。

进一步以相同真实 Binance 数据验证 A4 CUDA-R1 Hybrid：公开训练命令 exit 0，E1→R1→E2 完整 lineage；E1/E2 为 CPU ES，R1 为 CUDA PPO；3000 epoch 蒸馏在 parity tolerance 0.001 下通过并导出模型。bundle SHA `c1b8d5db...`，数据 hash仍为 `c19e2670...`；原生报告 719 个收益观测，净收益 **+0.00388%**、最大回撤 0.00219%、换手率 0.00419、0 清算、结束平仓。fee/margin/slippage/rule provenance 仍是场景假设，评估窗口不是独立 WFO Holdout，微小正收益不构成策略有效性证据。quota 记录及 GPU lock 均在成功后释放。

截至本次审计，先前报告中的主要接口问题已关闭：L2 经济事实覆盖不足会错误授权能力、L3 ObservationSchema 可变、L3/L6 原生账户事实无法生成研究指标、L4/L5 normalized transfer 的正式工作流输入域问题，以及 L2/L3 账户执行观察字段缺失。一次低层 provider 探针曾将 raw observations 直接传给只接受 policy-space observations 的 transfer API，不能代表 RLStageExecutor 工作流；使用非单位 VecNormalize 统计量的 A4 集成回归现已通过。本轮还修正了风险计数口径：软风险缩放与硬守卫调整现在分开记录，WFO/ES 约束仅计硬守卫调整。

项目现已实现 `rles` 命令入口和完整的 Validation→Test→选择→最终重训→一次性 Holdout WFO runner。单 fold 60/15/15 scenario run 已在真实 Binance 归档行情上完成 Validation、PPO 选择、Test、B0–B3 baseline、统计摘要、最终重训及一次性 Holdout；research ledger 进入 CLOSED，Test 查询恰为 1，Holdout OPEN/CLOSE 各一次。此前报告的 `PASS` 仅表示当时实现的流程和配置门禁通过，不表示策略有正收益或交易规则是历史真实规则；当前源码已增加净收益门槛。此前 v1–v4 的资源问题由历史 open Position DTO 中每分钟重复保存累计 fills/events 导致；v5 启用紧凑快照并使用 Nautilus 公共 cache purge API 后，在约 2.4–3.1 GiB 工作集内完成。

场景 WFO 的经济结果为负：所选 PPO 在 Test 净收益 -5.21%、最大回撤 5.22%，在 Holdout 净收益 -4.32%、最大回撤 4.35%；两段均 0 次清算。v5 报告由风险分类修复前的源码生成，其 21,597 次“非零风险调整”混合软缩放与硬限制，不能作为硬风险违规计数。随后同数据配方的 12 小时实跑显示，PPO 719 个收益周期中 718 次为软缩放、硬守卫调整为 0，证实旧计数口径会夸大硬风险触发。当前源码已区分两类，但由于该 Test/Holdout 已消费一次，本轮不重复使用它们；需要新的未使用数据窗口来验证修订后的完整 WFO。PBO 因单 fold 不适用。所有这些运行仍标记 `scenario-only:not-historical-rule-evidence`。WFO 总报告约 84 MiB，因把账户行、成交和基线明细嵌入总报告，仍有产物体积优化空间。

当前状态归纳：

| 方面 | 当前结论 |
|---|---|
| L1–L6 源码 | 已集成；根目录无 Git 仓库，但有可复算源码身份清单并记录 Nautilus 上游 commit |
| 回归 | 最新源码全量 **310 passed、4 CUDA-only skipped、45 warnings**；11-wheel 已构建但当前进程无法读取生成文件，因此 parity 和安装回归未验证；Ruff/MyPy 通过；真实归档 CLI 覆盖 A0–A4 五条路线、四种 RL 算法和八个独立日期 |
| 类型/静态检查 | MyPy packages+qa 共 103 源文件通过；Ruff packages/tests/qa 通过 |
| 应用入口 | `python -m rles_orchestrator` 已提供 market-audit、validate、train、backtest、wfo 命令 |
| 六种研究路线 | A0 PPO/SAC/TD3/DDPG、A1 ES、A2 R1→E2、A3 E1→R1、A4 E1→R1→E2 均有真实行情档案驱动的 scenario-only 产物；A0–A4 五路线覆盖八个真实日包的 train→bundle→独立 Test CLI 回放 |
| WFO | 旧版单 fold 的 Test/Holdout 亏损；2024-08 至 12 两折及 2024-10 至 2025-01 新窗口的 Validation 候选均未满足 0% 净收益门，均在 Test 前安全 ABORT；Holdout gate 尚无合格候选验证，公开 backtest 工件写入已复核 |
| 正式历史交易证据 | 已有真实行情/情景账户轨迹 golden；仍缺历史 instrument rule、账户历史费率及经证实的执行模型，不能作正式历史交易验收 |
| 交付 wheel | 11 个 wheel 构建命令成功，但 wheel 文件 ACL 阻断内容读取；当前源码与 wheel 文件 parity、wheel 安装回归、完整外部依赖离线安装均未验收；此前 wheel 测试环境借用 `.venv` 第三方依赖 |
| 发布判断 | 不通过完整项目设计验收；可继续研发和受限场景集成 |

## 2. 本轮复核证据

### 2.1 全量测试和静态检查

| 检查 | 命令/范围 | 实际结果 | 解释 |
|---|---|---|---|
| 全量 CPU 测试 | `.venv\Scripts\python.exe -m pytest -q --basetemp .tmp\pytest-full-3routes-test-20261013 --tb=short --show-capture=no -p no:cacheprovider` | **286 passed, 4 skipped, 33 warnings，105.31 秒** | 全部 RLES 包从 workspace editable 安装加载；4 项为 CUDA-only smoke 跳过 |
| 隔离 wheel CPU 测试 | `.tmp\wheelhouse-clean-venv\Scripts\python.exe -m pytest -q --basetemp .tmp\pytest-wheel-3routes-test-20261013 --tb=short --show-capture=no -p no:cacheprovider` | **286 passed, 4 skipped, 33 warnings，105.02 秒** | 11 个 RLES wheel 从隔离 venv 加载；第三方依赖由 `.pth` 借用现有 `.venv`，不代表自包含离线安装 |
| MyPy | `.venv\Scripts\mypy.exe packages qa` | **103 source files，0 errors** | 未启用 `--check-untyped-defs`，未标注函数体检查范围较弱 |
| Ruff | `.venv\Scripts\ruff.exe check packages tests qa` | **All checks passed** | 项目 lint 规则只启用 E4、E9、F |
| CUDA RL 专项 | `.venv-cuda\Scripts\pytest.exe -q tests/test_cuda_smoke.py --basetemp=.tmp\pytest-cuda-smoke-final --tb=short --show-capture=no -p no:cacheprovider` | **4 passed，1 SB3 MLP 性能提示** | PPO/SAC/TD3/DDPG CUDA 训练、保存、加载、推理；真实应用级 CUDA 另见本轮 A0 运行 |
| 真实行情应用级 CUDA | `.venv-cuda\Scripts\python.exe -m rles_orchestrator train --config .tmp/real-ethusdt-scenario/config-cuda-a0-20261012.json` | **exit 0** | 真实 Binance 2024-01-01 行情，A0 PPO CUDA 训练与 Nautilus 原生验证回放；场景规则，非历史规则验收 |
| 真实行情 SAC CUDA | `.venv-cuda\Scripts\python.exe -m rles_orchestrator train --config .tmp/real-ethusdt-scenario/config-cuda-sac-20261012.json` | **exit 0** | 真实市场归档、CUDA learner、模型工件和 Nautilus 原生回放 |
| 真实行情 TD3 CUDA | `.venv-cuda\Scripts\python.exe -m rles_orchestrator train --config .tmp/real-ethusdt-scenario/config-cuda-td3-20261012.json` | **exit 0** | 同上 |
| 真实行情 DDPG CUDA | `.venv-cuda\Scripts\python.exe -m rles_orchestrator train --config .tmp/real-ethusdt-scenario/config-cuda-ddpg-20261012.json` | **exit 0** | 同上 |
| 真实行情 CUDA Hybrid | `.venv-cuda\Scripts\python.exe -m rles_orchestrator train --config .tmp/real-ethusdt-scenario/config-cuda-a4-20261012.json` | **exit 0** | A4 E1 CPU ES → R1 CUDA PPO → E2 CPU ES；parity 0.001，Nautilus 原生回放，场景规则 |
| CUDA 不可用拒绝 | CPU `.venv` 调用 `train({route:A0, rl_device:cuda})` | **DataError，output 未创建** | 证明应用级 preflight 在数据工厂和 run allocation 前失败关闭 |
| 关键集成定向回归 | metrics、historical factory、layer invariants、L2 bridge | **49 passed** | 覆盖本轮性能优化、native report 和应用工厂路径 |
| 定向风险归因测试 | historical factory + WFO selector + ES/RL/Hybrid native integration | **51 passed**；另有 2 项新语义断言通过 | 确认软 funding 缩放不会记为硬风险违规；未知分类在 WFO 选择中失败关闭 |
| 环境依赖 | `uv pip check --python .venv\Scripts\python.exe`，项目内 `UV_CACHE_DIR` | **68 packages compatible** | 是现有环境依赖一致性，不是从 wheelhouse 的干净安装验收 |
| 应用帮助 | `.venv\Scripts\python.exe -m rles_orchestrator --help` | 0 退出，列出 5 个命令 | CLI 可被当前 editable 环境调用；`Scripts` 下未发现 `rles.exe`，wheel/console-script 安装路径仍应在干净安装验证 |

### 2.2 三项旧缺陷探针

复跑 `.tmp\audit_functional_probes_20261011.py`：

| 探针 | 结果 | 当前状态 |
|---|---|---|
| DQ 经济事实覆盖 | 1,440 根分钟线、仅 1 个 mark 和 settlement、instrument rule 第 2 分钟失效；输出 DQ-C，`economic_coverage_unverified`，liquidation 授权为 false | 旧问题已修复；能力 gate 失败关闭 |
| ObservationSchema 不可变性 | mutable list 不被接受；外部修改不能改变 schema hash | 旧问题已修复 |
| normalize=True 的 RL→R1 transfer | 正式 A4 工作流使用非单位 VecNormalize 统计量完成 PPO/ES/RL-ES 与 L6 ledger 集成验证 | 先前 raw→policy-space API 误用探针不代表应用缺陷；见已关闭 F03 |

### 2.3 真实市场档案场景输出

输入是本地 Binance ETHUSDT 归档行情（2023-12-30 至 2024-04-15 档案，实际运行窗口为 2024-01-01 附近）。盘口不是 OHLC 合成：Binance Vision `bookTicker` 日档原始文件共 7,376,016 行，sha256 `8b0ba77526b089682e541fd4d80dcf7d228f50227ec8e5c81ca7431e0e98eb2f`；按 event-time 秒保留最后报价后为 86,394 条，含真实 best bid/ask、显示数量及 update ID。抽样盘口 ZIP 已从临时探针目录归档到 `data/binance_ethusdt_20240101_quotes_1s/raw/`，导入清单与 source audit 均指向归档文件，hash 复核一致。其余规则/费用/延迟/滑点及风险设置仍注明为 scenario-only，不是已验证的 2024 历史参数；quote stream 是一秒抽样 best quote，不含更深档位或完整撮合队列证据。各报告完整性哈希已核验，市场档案 provenance 包含 5 个 Binance archive 来源 URL，DQ-B、重复/冲突计数为 0，结束时账户平仓。

| 路线/算法 | 净收益 | 最大回撤 | 成交数 | 状态 |
|---|---:|---:|---:|---|
| A0 PPO | -0.2182% | 0.2254% | 445 | 完成，情景集成证据 |
| A0 SAC | -0.0386% | 0.0475% | 339 | 完成，情景集成证据 |
| A0 TD3 | -0.1724% | 0.1759% | 517 | 完成，情景集成证据 |
| A0 DDPG | -0.0866% | 0.0926% | 434 | 完成，情景集成证据 |
| A1 ES | +0.0157% | 0.0090% | 2 | 完成，情景集成证据 |
| A4 Hybrid | -0.0151% | 0.0170% | 9 | 完成，情景集成证据 |

上表为非常短的单窗口工程集成结果，不可比较或解释为策略收益能力。各 JSON 产物位于 `.tmp\real-ethusdt-scenario\{ppo-output-cli,sac-output,td3-output,ddpg-output,a1-output-v3,a4-output-v2}\COMPLETED.json` 及相邻 report 文件。

本轮风险口径修改后，再用同一真实 Binance 归档行情执行 12 小时训练/12 小时验证，四种 RL 算法与 ES、Hybrid 都走配置驱动 `train`、写入 bundle 后加载 actor 做原生 Nautilus 评估；六条 public `backtest` 命令也均为 exit 0。

| 路线/算法 | 净收益 | 最大回撤 | fills | bundle/终态 |
|---|---:|---:|---:|---|
| A0 PPO | -0.4198% | 0.4252% | 565 | 重载成功、平仓 |
| A0 SAC | -0.1248% | 0.1254% | 518 | 重载成功、平仓 |
| A0 TD3 | -0.1819% | 0.1819% | 570 | 重载成功、平仓 |
| A0 DDPG | -0.1536% | 0.1547% | 539 | 重载成功、平仓 |
| A1 ES | +0.0157% | 0.0090% | 2 | 重载成功、平仓 |
| A4 Hybrid | -0.0109% | 0.0124% | 4 | E1/R1/E2 bundle 重载成功、平仓 |

每次评估有 719 个收益观测和 720 行账户轨迹；当前口径记录 718 个软调整周期、0 次硬守卫调整。bundle、report 的 SHA-256 校验及 COMPLETED 标记均匹配。规则来源仍是场景假设，因此这些短窗结果证明当前源码能保存/加载各路线工件并运行真实归档行情评估，不代表收益能力。

本轮再用**最新源码**对同一真实 Binance 1 秒抽样 best-quote 归档（12 小时训练、12 小时 Test）执行 PPO A0 与 A4 Hybrid 配置驱动训练，A4 完成 E1→R1→E2，分别导出 bundle/report，并各自通过独立 public `backtest` CLI 从工件加载后运行原生 Nautilus。两条 CLI 均 exit 0，两个 bundle/report 的 SHA-256 与 `COMPLETED.json` 和 sidecar 一致，终态账户均平仓；各有 719 收益观测、43,200 best-quote 行、DQ-B、重复/冲突数为 0。A0 PPO 净收益 -0.1499%、最大回撤 0.1499%、482 fills；A4 净收益 -0.1987%、最大回撤 0.1993%、507 fills。二者 `rule_provenance` 仍是 `scenario-only:not-historical-rule-evidence`，结果仅证明本轮源码的训练/保存/加载/回测链路。新产物位于 `.tmp/real-ethusdt-scenario/current-src-ppo-a0/` 和 `.tmp/real-ethusdt-scenario/current-src-a4-v4/`；A4 的 L4 v4 R1 检查点 step 32 与 step 64 均有 replay JSON 和完成标记。

### 2.4 WFO 实跑

- 计划：60/15/15 单 fold，真实 Binance 行情，显式 `scenario-only:not-historical-rule-evidence` 规则；Train、Validation、Test 和 15 日 Holdout 时间段分离，WFO 专用输出目录与 SQLite research ledger 已创建。
- 前置验证：market coverage/DQ-B、缺失/重复/冲突事件检查、settlement schedule 和完整窗口检查通过；资金费事件与 8 小时计划边界的差异在归档实测最高约 15ms，WFO 规则单独处理，不改写原始事件时间戳。
- 实跑配置：`real-ethusdt-scenario-wfo-linear-v5-20261012`，ETHUSDT 归档行情，60/15/15 单 fold；两名候选 PPO/SAC 验证后选 PPO，唯一 Test 查询一次，最终重训后消费一次 Holdout；SQLite ledger 为 CLOSED，holdout OPEN/CLOSE 各一次，`final_holdout` 独立查询数为 0。总报告 `conclusion=PASS`，只表示该配置的流程门禁通过。
- Test：21,598 个收益观测、21,599 行账户轨迹、13,635 fills，净收益 **-5.2127%**，最大回撤 **5.2231%**，0 清算、21,597 风险守卫干预。
- Holdout：15 日，21,598 个收益观测、21,599 行账户轨迹、11,605 fills、44 funding 事件，净收益 **-4.3171%**，最大回撤 **4.3501%**，0 清算。旧报告中的 21,597 是混合软/硬的非零调整计数，不能解释为硬守卫违规；报告成本为 fee 331.21、funding -0.55、slippage 67.30 USDT；规则 provenance 仍为 `scenario-only:not-historical-rule-evidence`。
- 统计与基线：bootstrap 95% 区间约 [-2.761e-6,-2.154e-6] 每期收益，DSR 概率约 6.69e-68；PBO 为 `null`（仅一个 fold）。Test 的 B0 no-trade 净收益 0，B1 always-long -0.3855%，B2 momentum -16.38%，B3 random -29.19%。所选 PPO 同期 -5.21%，输给 B0/B1。结果不支持该场景策略优于简单基线。
- 工件：总报告和 Holdout 报告均生成 SHA-256 sidecar；research ledger 有 3 次 query（2 validation、1 test），Holdout 由单独 OPEN/CLOSE 记录和最终 report hash 绑定。总 WFO JSON 为 88,896,384 bytes（约 84.8 MiB），因内嵌逐行账户、成交和基线轨迹，仍有产物体积优化空间。
- 版本边界：v5 ledger `code_sha=438fce3b...`，与当前风险分类修复后的源码树 hash `cc369a69...` 不同；v5 WFO 的净收益、费用和原生账本结果保留为已消费 Holdout 证据，但候选风险违规字段由旧口径生成。未对同一 Holdout 重跑；需要新鲜时间窗验证新 selector 口径。
- 代码审查：WFO runner 支持 60/15/15、360/15/15、coverage gate、scaler 绑定、candidate validation、验证选择、fold Test、B0–B3 baseline、bootstrap/DSR/条件性 PBO、最终 pre-Holdout retrain、一次性 Holdout 和 ledger 对账。已证明单 fold scenario 闭环；多 fold 和真实历史经济模型未验收。

### 2.5 本轮性能优化与新的实测限制

本轮为长时间数据运行补充了以下有限优化：

- `packages/gym_bridge/src/rles_gym_bridge/historical.py` 将每个决策点反复扫描全部历史市场批次，改为按单调决策时间分段；
- `packages/data/src/rles_data/core.py` 与 `features.py` 为 instrument 和 event-time 窗口建立索引，避免每行反复复制/扫描全量历史特征；
- historical snapshots 改为保存账户当前状态和逐时点成交/funding 增量，最终快照再保留完整事实；`native_metrics.py` 将增量事实汇总成账户指标并检查重复身份冲突。

这些改动的 targeted tests 与全量测试通过。为避免每分钟 open Position DTO 再复制累计 fills/adjustments/commission/trade IDs，历史工厂现使用 compact NativeFactsReader：持仓快照只保留当前经济状态；fills/funding/liquidation 由独立历史事实增量记录；每次 detachment 后用 Nautilus 公共 API purge closed orders、closed positions 和过期 account events。新 parity 测试比较 compact/non-compact 的权益、持仓经济状态、open orders、费用/funding/清算、成交经济记录和终态，均一致。v5 单 fold WFO 在约 2.4–3.1 GiB 工作集完成；此前未紧凑化的 v1–v4 在 13–17 GiB 附近增长并中止。仍缺 90 天规模峰值 RSS/profile、Windows Job Object 硬上限，以及异常中止恢复证据；WFO 总报告约 84.8 MiB 也需改成明细外置、摘要引用。

### 2.6 新鲜历史窗口获取与导入恢复

为验证风险分类修复后的 WFO，已尝试通过项目 Binance USD-M 导入器取得 2024-04-15 至 2024-07-31 的官方新日档。当前网络在 Python TLS 和 Windows SChannel 均出现间歇 EOF；系统 curl 曾对官方校验文件返回 HTTP 200，但完整窗口获取中途失败。失败目录保留 75 个已通过导入器官方 SHA 校验后写盘的原始 ZIP（覆盖至 2024-05-09），但没有 `market.parquet` 或 `import.json`，因此该部分不构成可用于训练/WFO 的数据集，也没有据此启动回测。

针对瞬时网络错误，`packages/data/src/rles_data/binance_archive.py` 对 `.CHECKSUM` 和 ZIP 请求执行有限标准库重试，并在失败后使用系统 curl（默认 TLS 证书校验、有限重试、下载大小限制）；官方 SHA 校验仍在写入前执行。随后增加每归档来源收据以支持可验证续传。最初的中断目录保留 75 个已核验 ZIP；之后网络恢复，完整 328 归档导入成功，细节见本报告追加核验段。

### 2.7 F09 的 L3→L4 checkpoint 接口

为接通已有 RL stage checkpoint 与历史原生环境恢复，`EpisodeReplay` 提供严格 JSON 序列化/解析，`NautilusBarrierEnv.rles_export_replay()` 导出动作日志并验证其 transition hash、boundary token 与当前 native episode 相符；L4 checkpoint 升为 v4，将每个 Env 的 replay JSON 纳入 checkpoint 文件校验，恢复时核对重建的 L3 replay 与 checkpoint 完全一致。`RLStageExecutor` 现按 SB3 完整 rollout block 分段训练，并在每次 `learn()` 返回后的稳定边界写 R1 checkpoint；避免在 PPO callback 的优化/观测状态尚未稳定时保存。A4 应用集成用例确认按 2 步间隔生成 step 2/4 两份已完成 v4 检查点，checkpoint replay tape 的 boundary token 与清单吻合。验证：`tests/test_l3_l4_boundary.py tests/test_native_training_episode.py tests/test_l4_provider_smoke.py` **46 passed**；A4 checkpoint 集成用例 **1 passed**；`mypy packages` **97 files, no issues**。这些是短窗 fixture 验证，不构成真实市场训练证据。

应用新增 `run.json` 固定 recipe/RunSpec/run/attempt id，A0 可在同一 run 目录指定部分 R1 checkpoint，并在新进程重建 L3 replay、恢复 SB3 continuation/optimizer/replay/normalizer/RNG 状态，续训剩余步数并累计原 usage。fixture 中断测试在 step 2 后恢复到 step 4。真实 Binance 归档上的两进程试验在 step 32 中断后恢复到 step 64，A0 bundle SHA 与连续训练相同，独立 public backtest CLI exit 0，报告 market hash `c19e2670...`、best-quote 43,200 行、终态平仓，usage 为 64 步。产物位于 `.tmp/real-ethusdt-scenario/current-src-ppo-resume/`。应用级故障注入现覆盖 A0/A2/A3/A4 的 R1、A1/A3/A4 的 E1、A2/A4 的 E2：从完成检查点恢复并完成剩余阶段和原生回测。E1/E2 会复核 L6 reservations/receipt、RunSpec、actor hash 及 R1 replay checkpoint。A4 三阶段真实归档中断恢复均通过。ES 部分 generation 的额外 L6 reservation 现于 worker pool 启动前检测并安全拒绝同 attempt 重放；需要用新 run/attempt 重新研究，已预留请求不退费。

### 2.8 本轮 PPO Hybrid 蒸馏与真实市场链路验证

真实行情 A4 使用默认蒸馏预算时，旧损失因 PPO action head 饱和而无法通过 0.001 parity 门槛。修复损失域后，300 epoch 误差下降到 0.01368；在相同严格门槛下增加到 3000 epoch 后 E1/R1/E2 完成，step 32/64 v4 检查点存在。应用默认蒸馏预算现为 3000，配置仍可显式覆写。回归测试将 PPO action head 初始 bias 设为 8，覆盖旧版裁剪后零梯度场景。`tests/test_l4_provider_smoke.py tests/test_native_training_episode.py` **40 passed**；全量 CPU suite **261 passed, 4 CUDA skipped**；`mypy packages` **97 files, no issues**；全仓 Ruff **passed**。

## 3. L1–L6 状态矩阵

| 层 | 当前可确认能力 | 未闭合的核心事项 | 判断 |
|---|---|---|---|
| L1 Contracts/Runtime | DTO、配置、schema/hash、种子、序列化、provider registry 和跨包契约；97 文件静态检查通过 | 全系统源码缺统一 Git commit/tag；算法/选择语义与 run identity 的强绑定仍需核对；完整发布验收缺失 | 基础能力可用，交付身份未统一 |
| L2 DataTruth | RAW/CURATED/FEATURE/FOLD、可用时间、事件去重、历史覆盖检查、feature/scaler、fold manifest；旧 DQ 能力过宽问题已修复 | 真实历史 instrument rules/funding/mark/index 和能力覆盖需可信外部依据；WFO 特征/行情长窗口性能需可扩展证明 | 核心组件可用，数据规则真实性未验收 |
| L3 Nautilus/执行/Gym | 官方 Nautilus 引擎、动作到订单、取消/终止、native 账户/fee/funding、清算事件、replay、历史环境 factory；账户和执行状态观察已覆盖 | 真实执行/保证金/滑点模型绑定、90 天缓存峰值/硬资源上限和异常恢复未完成 | 情景交易链路已通，生产经济模型未验收 |
| L4 RL SB3 | PPO/SAC/TD3/DDPG 训练/保存/加载/推理、VecEnv、normalizer；A0/A2/A3/A4 R1 检查点恢复；CUDA smoke 四算法及四算法真实行情 CUDA app 训练/原生回放；A0/A2/A3/A4 路由有真实 CPU 闭环；A4 三阶段真实归档恢复 | CUDA WFO | 多路线 CPU 恢复矩阵与四算法 A0 CUDA 实际数据路径已跑；WFO 与长时资源验收未闭合 |
| L5 ES/Hybrid | PGPE/CMA-ES、A0–A4 编排；真实行情 CPU A1–A4 组合路线与 A4 CUDA-R1 Hybrid 已运行；A1/A3/A4 E1、A2/A4 E2 检查点恢复 | E1/E2 CUDA 与 ES 代内中断恢复、正式历史执行参数 | 独立 CPU、CUDA-R1 Hybrid 与各支持路由的合成故障恢复已跑；ES-GPU/代内恢复/正式经济验收未闭合 |
| L6 Compute/Research | spawn worker、资源 reservation/lease、研究账本、WFO runner；Validation/Test、已完成 retrain、已有完整 checkpoint 的 retrain 续跑、持久 Holdout 报告和 CLOSED 报告生成边界现可恢复；Holdout 无持久报告时失败关闭 | 尚无完整 checkpoint 的 final-retrain 中断恢复、ES 代内请求多于已完成 checkpoint 时的新 attempt 工作流、多折真实 WFO 长跑/吞吐、OS 级硬资源限制、固定预算 perf 基准、完全隔离依赖安装 | 应用阶段恢复扩展，长跑性能/发布门禁未完成 |

## 4. 问题清单与开发要求

优先级含义：P0 阻断正式研究结论或完整项目验收；P1 发布前应关闭；P2 可追溯性/维护事项。已关闭项保留以对照旧报告。

### F01 · 已关闭 · P0 原 L2 DQ 能力授权遗漏

**证据：** 本轮反例复跑得到 DQ-C、`economic_coverage_unverified`，且 `require_usable(..., "liquidation")` 不授权。应保留对应回归测试并确保每个用途要求的 mark/index/funding/rule 时间覆盖继续绑定 DQ report。

### F02 · 已关闭 · P1 原 ObservationSchema 接受可变列表

**证据：** 本轮探针显示 mutable list 被拒绝，hash 不会因调用方修改输入而变化。保留这两项契约测试。

### F03 · 已关闭 · P1 L4/L5 normalize=True 的迁移和蒸馏输入域

`RLStageExecutor` 明确先调用 `provider.policy_observations(fixed)` 再进入 policy-space transfer；蒸馏训练输入也通过对应 normalizer wrapper 转换，teacher 仍读取 raw observation。先前探针把 raw observation 直接传入签名要求 policy-space 输入的 `transfer()`，因此得出的 0.583 差异是 API 使用错误。非单位 VecNormalize 统计量的 A4 provider-backed 集成路径已通过。本轮真实行情 A4 又发现 PPO BC 的饱和梯度缺陷：PPO 确定性推理裁剪未 squash actor 输出，但旧蒸馏损失先裁剪输出，饱和时梯度为零。L4 现用未裁剪 PPO actor 输出对齐有界教师动作；SAC/TD3/DDPG 的 BC 损失显式匹配各自的 tanh 输出域。饱和 PPO 初始策略回归通过；真实 A4 在严格 0.001 parity 门槛下，旧损失误差为 0.7264，修复后默认 300 epoch 仍有 0.01368，增加到 3000 epoch 后成功完成 E1/R1/E2，没有放宽门槛。应用与 `RLStageExecutor` 默认蒸馏预算同步设为 3000 epoch，并允许配置覆写。

### F04 · P0 · 应用入口存在，正式历史数据/规则和发布级集成仍缺

旧报告“没有应用入口”的结论已过时：`rles_orchestrator` 提供 `market-audit/validate/train/backtest/wfo`，本轮真实市场档案 A0/A1/A4 情景任务已执行。但这些任务使用 scenario-only 历史 instrument rule 和成本参数，不能通过正式 historical economics 审核。

公开 `backtest` CLI 现将完整报告、SHA-256 sidecar 和 bundle 绑定完成标记写入配置 output 目录，并拒绝覆盖已有目录；真实 Holdout bundle 重载与逐项报告一致已通过。正式历史数据/规则依据和 clean wheel install 仍阻止发布级验收。

**修改任务：** 从可核验交易所/官方数据源提供历史合约规格、手续费等级、资金费率与时间、保证金和清算规则；锁定原始归档与规则来源的 hash；从 clean wheel install 启动同一 CLI，完成一条真实历史经济事实对账运行。

**本轮补充：** 2024-01-01 的真实 Binance best-quote 原始档案已封存并核验；已知官方最低名义价值和 2023-12-24 起的最低保证金档位记录在 `data/binance_ethusdt_20240101/HISTORICAL_RULES_BLOCKER.md`。仍缺历史 tick/step/min quantity、账户费率与覆盖整个回放期的保证金/清算映射，不能据此解除 historical-economics gate。

### F05 · 已关闭 · P1 L2/L3 账户/执行观察覆盖

历史观察已纳入 equity/peak/drawdown/cash/free margin/initial margin/maintenance margin/unrealized PnL/net exposure/position quantity，以及 raw/executed targets、pending count/signed/gross exposure/fill rate、guard intervention 和 risk headroom。`test_production_factory_uses_native_models_full_observations_and_terminal_costs` 与 `test_native_observation_exposes_pending_order_direction_and_size` 覆盖关键列和挂单方向/规模。

### F06 · P0 · 执行情景与真实历史经济模型仍未事件级证明

场景运行将 execution schema、latency/slippage、market fill、硬/软风控身份写入训练和 bundle 身份；但 1bp/5bp 等运行参数明确是情景假设。没有证明它们等于 2024 ETHUSDT 的历史手续费、撮合、延迟、资金费和逐时保证金/清算规则。

planner 终态 Reduce-Only 的 minNotional 豁免已按 Binance 官方 USDⓈ-M 规则实现为逐 instrument 可选能力，并加入真实归档回归；该规则只证明终态 Reduce-Only 的最低名义价值例外，不补齐历史账户费率、保证金或逐时执行事实。

**当前运行差异：** A0/A1/A4 这些已完成的短窗运行用真实归档 best-quote（每秒抽样一次、只含 top of book）和 0bp 合成滑点的场景 fill 假设；WFO 配置使用 bar 粒度 1bp base / 5bp stress 假设。两者都没有被审计证明等于历史实际手续费、延迟、funding 或保证金规则，秒级 quote 重放也没有证明队列优先级或未采样期间的成交状态。

**修改任务：** 依据历史资料映射到实际 Nautilus fee/fill/latency/margin 对象；逐事件对照订单接受、成交时间/数量/价格、commission、funding、margin 和 equity；对 bar 粒度无法支持的能力拒绝或降级声明。

### F07 · 单 fold scenario 流程已通过 · P0 WFO 统计与经济验收仍有限

2024-04-15 至 2024-07-31 窗口完成 60/15/15 的 Validation、选择、一次 Test、B0–B3、bootstrap/DSR、pre-Holdout retrain、one-shot Holdout 和 ledger CLOSE；Test 查询 1 次，Holdout OPEN/CLOSE 各一次，最终 report hash 入账。PBO 因单 fold 返回 `null`。PPO Test/Holdout 净收益为 -1.76%/-2.04%，Test 输给 B0 no-trade；旧版报告的 PASS 只代表配置风险门槛通过。新增的 Validation 净收益 floor 与 Holdout 净收益门槛已实现；2024-08-13 至 2024-11-16 窗口的真实归档运行中，PPO/SAC 均为负收益并在 Test 前被拒绝，确认门槛能失败关闭。2024-08-12 至 2024-12-31 两折窗口完成 4 份真实 Validation：PPO fold0/fold1 净收益 -4.711%/-5.266%，SAC -5.338%/-1.984%；无候选过 0 收益下限，Test/Holdout 均未打开，ledger ABORTED、审计链有效。2024-10 至 2025-01 新窗口两候选净收益为 -3.3852%/-3.2855%，同样 0 Test、Holdout 未打开，ledger ABORTED、审计链有效。Holdout gate 仍无“合格 Validation 候选”后的真实流程证据。360/15/15、多折 Test/Holdout 闭环、跨市场/参数稳健性、策略归因及报告瘦身仍待开发。

### F08 · 已关闭 · P1 原生账户事实到研究指标桥接缺失

`application.rollout()` 当前调用 `native_report()`，将 native equity snapshots 和逐期 fee/funding/turnover/slippage 增量映射到账户行；WFO 也消费这些报告。本轮 `test_research_metrics.py`、`test_historical_factory.py` 和全量测试通过。仍须在完整长 WFO/真实规则场景做权益与逐笔经济事实对账，但原“没有生产桥接”的结论可关闭。

### F09 · P1 · L4/L5 整阶段 checkpoint/resume 未闭合

底层 RL provider 与 CPU ES 分代恢复有测试；Gym/L3 replay journal 已纳入 L4 checkpoint v4。应用已固定 `run.json` recipe/RunSpec/attempt，并通过真实 Binance 归档证明 A0 partial R1 可在新进程恢复 L3 replay、SB3 continuation/optimizer/replay/normalizer/RNG 并累计 usage；连续与恢复 bundle SHA 相同。A4 的真实归档 E1 第一代、R1 第 32 步、E2 第一代中断恢复现均已验证：每次恢复后跑完 E1→R1→E2 并以独立 Test CLI 原生回测确认。合成恢复覆盖 R1 的 A0/A2/A3/A4、E1 的 A1/A3/A4、E2 的 A2/A4；恢复会复核 L6 reservations/receipt、transfer evidence、完整 R1 checkpoint 与 actor hash。A0/A4 全预算 R1 checkpoint 支持 finalize-only 导出。任一 ES 代内部发生中断仍不能恢复未完成的一代。

本轮补上 E1/E2 应用恢复入口和恢复完整性验证；E1 与 E2 故障注入用例均在完整 E1→R1→E2 和 Native 回测后通过。E2 恢复验证还确认原 R1 actor 的已校验导出哈希不变。全量 CPU suite **279 passed, 4 CUDA skipped, 33 warnings in 70.35 sec**；Ruff 通过，mypy 97 文件无错误。该恢复验证使用短窗 fixture，不构成真实市场中断恢复、策略效能或历史规则证据。

**后续验收：** 为 ES generation 内、候选批次评估中断记录可审计的完成/重派边界，或明确拒绝从未完成 generation 自动续跑；真实 A4 三个阶段边界恢复已验证。无需承诺任意 native env mid-step 逐位恢复。

### F10 · P1 · CUDA ES checkpoint 和原生 GPU 流程未闭环

公开应用现支持 `rl_device=cpu|cuda`，且使用 L6 quota 与排他 GPU owner lease；CUDA 不可用或路由没有 R1 时在建环境前失败关闭。已完成真实 ETHUSDT 归档驱动的四种 A0 RL CUDA 和 A4 CUDA-R1 Hybrid 训练及 Nautilus 原生回放，另有 SB3 CUDA smoke 四算法通过。底层 EvoTorch checkpoint/resume 仍限制 CPU，CUDA ES/E1/E2 尚未实现；CUDA WFO 和 ES-GPU 组合也未验证。公开的 `auto` 仍要求获批端到端基准决策，当前应用只接受显式 cpu/cuda。

**修改任务：** 若启用 CUDA ES，则实现 ES codec 保存/恢复并把活动、owner 和 attempt 一致的 GPU lease 绑定执行；否则维持 `es_device` 未开放且配置层明确拒绝。补 CUDA ES/E1/E2 或拒绝测试、CUDA WFO 路径、真实 OOM 与 new-attempt 行为。确认 GPU MLP 的实际吞吐优于 CPU 后再建议默认启用；SB3 已提示 PPO MLP 在 GPU 上可能慢于 CPU。

### F11 · P0 · OS 资源限制、内存硬上限和故障恢复不足

compact history 修复后单 fold 15 日 WFO 在约 2.4–3.1 GiB 工作集内完成，表明原本 13–17 GiB 增长并非该规模下不可避免。最新两折运行在首个 Validation 回放约 19 分钟时工作集约 3.9 GiB，仍未提交结果；长回放的耗时和 Nautilus INFO 输出需先优化并计入性能门禁。L6 quota 仍是软件预留，不是 Windows Job Object 硬限制；未记录稳定的进程峰值/commit/RSS profile，也没有 OOM/kill/磁盘耗尽恢复证据。90 日运行边界尚未知。

**修改任务：** 为 worker 使用 Job Object 或等价硬限制、RSS/commit 监控和安全阈值；增加 worker 强杀、宿主中止、磁盘不足、checkpoint 损坏注入；证明账本不重复计费、lease 清理、attempt 隔离与恢复边界。

### F12 · P1 · bundle 已含交易身份，仍非独立可移植发布工件

训练 bundle 已写 actor/ES genome、feature/schema/scaler、dataset、execution、hard risk、config identities、environment 和 stage lineage；backtest 校验这些身份，较旧报告有明显进展。但 bundle/environment 仍依赖路径/工作区配置，缺同一发布物的依赖锁、wheel 身份/SBOM、可携带规则数据/校验清单及干净环境加载演练。

**修改任务：** 把所需数据/规则或不可变获取声明、feature/scaler、actor、risk/execution recipe、依赖和源/wheel 身份做成自包含清单；断网 clean install 重载并逐笔重放。

### F13 · 部分关闭 · P1 当前 11-wheel 集合同步与干净安装

旧目录 `dist-current-verified-20261012` 曾有 6 个包、19 个 Python 文件与源码不一致。本轮用 `uv build --all-packages --wheel` 构建新的 11-wheel 集合到 `.tmp/wheelhouse-current-20261013/`，逐包比较 wheel 内所有 Python 文件与当前 `packages/*/src`，97/97 个包内 Python 文件均字节一致。清单 [wheelhouse_current_20261013.json](<D:/Program Files/NautilusTrader_RL_ES/qa/reports/wheelhouse_current_20261013.json>) 绑定源码身份 `cc03cf060e7d1e477f09ae0fe2dc15ae609128e4f5fef60f37098deac7a73be6`，记录 11 个 wheel 的 SHA-256；清单文件 SHA-256 为 `18a942c23d5bd190b30a1b635e531e03a7956cacffe6d0eb3d5df2298458c9b8`。

使用 Python 3.14.7 新建隔离 venv，离线安装 11 个 wheel 后，11 个模块均从该 venv 的 wheel 安装路径导入，`wheel_imports.py` provenance 检查通过，`python -m rles_orchestrator --help` 通过。外部第三方依赖 wheel 未包含在项目 wheelhouse，uv cache 也缺多项锁定依赖；在禁网模式下依赖解析正确失败。为完成本地包 smoke，本次隔离 venv 用 `.pth` 临时读取既有 `.venv` 第三方依赖，因此这不是完整的自包含断网安装验收。

2026-10-13 五路线补测后，重新构建的 11-wheel 集合位于 `.tmp/wheelhouse-five-routes-20261013/`，manifest 为 [wheelhouse-five-routes-20261013.json](<D:/Program Files/NautilusTrader_RL_ES/qa/reports/wheelhouse-five-routes-20261013.json>)。它绑定当前 212 文件 source identity `cb0d73b5…39b69a4`；11/11 wheels 覆盖当前包源码，97/97 个 Python 文件逐字节相同。11 个 wheel 已重装到 Python 3.14.7 隔离环境，导入 provenance 与 CLI smoke 通过；隔离环境全量 **293 passed、4 CUDA-only skipped、38 warnings，152.28 秒**。该环境仍通过 `.pth` 读取已有 `.venv` 第三方依赖，所以完整外部依赖断网安装仍未通过。

**剩余任务：** 将完整外部依赖及其 hash 加入可移植 wheelhouse，在无共享 site-packages 的 Python 3.14.7 环境执行全量测试、应用数据重载和 checkpoint 恢复；将 `dist-current-20261012` 旧目录标记为过期/不供发布，发布候选使用当前绑定 source identity 的 wheelhouse。
### F14 · 部分关闭 · P1 real_golden 已补，perf 发布门仍缺

本轮新增可移植 Binance 官方原始归档小样本、逐 ZIP SHA 收据与完成标记，并运行真实行情驱动的 Nautilus 原生账户轨迹 golden。用例覆盖 08:00 UTC funding settlement，锁定 477 步、3 笔成交、费用/滑点/资金费/终值、0 清算和终态平仓；真实应用 CLI 集成现覆盖 A0–A4 五条路线及 Jan 1–5 五个独立日期，并逐条执行 Test CLI。经济参数是明确标记的 scenario-only 假设，不是历史规则或盈利证据。真实 golden/应用路线均通过；源码与最新 wheel 环境全量均为 293 passed、4 CUDA-only skipped。

**剩余任务：** 增加固定预算 perf，用隔离进程监控 RSS/CPU，并分别报告数据装载、引擎构建、评估和工件 I/O；将 real_golden 与 perf 纳入 nightly/release gate。账户 golden 可继续扩展多费用模型和未平仓结算边界，但不得把情景规则标为历史事实。

### F15 · 部分关闭 · P2 根工作区源码身份

根目录没有 `.git`。新增 [源码身份生成器](<D:/Program Files/NautilusTrader_RL_ES/qa/source_manifest.py>) 与 [当前源码清单](<D:/Program Files/NautilusTrader_RL_ES/qa/reports/source_manifest_current.json>)，按明确范围记录 212 个源码、测试、配置和工作流文件的路径、大小与 SHA-256，并记录已安装分布。当前 `source_identity_sha256` 为 `cb0d73b549ec86aa76258cf68bf071a40cf812fd018bc74e15383143939b69a4`，清单文件 SHA-256 为 `9a7354e5d05824ffe0401b83e408f62ff2d5fbc1009a68afa9e5d9f96b930917`。Nautilus 上游身份为 commit `b82f3e3c801f4d7d0423f11c34fea8806c09e68a`，tracked worktree clean；唯一状态项是未跟踪 `python/nautilus_trader/nautilus_pyo3.pdb`，tracked diff SHA 是空文件 SHA。`python qa/source_manifest.py --verify` 可复算并拒绝过期清单。

**剩余任务：** 发布构建时归档清单并将其 SHA 绑定到 wheelhouse/build attestation。当前新增测试/夹具使整体源码身份更新，而上一份 11-wheel 报告仍绑定到之前的整体身份；包内源码未随本轮改变，现有逐包内容比对与隔离安装测试仍有效，但发布时需重签当前的源码/wheel身份关联。文档或运行数据如需进入身份范围，须显式更新清单范围。

### F16 · 部分关闭 · P0 历史快照二次方增长与 cache 保留

compact native history 移除每个历史时点内嵌的 open Position 累计 fills/adjustments/commissions/IDs，并在捕获事实后调用 Nautilus 公共 purge API 回收关闭订单/持仓及旧账户事件。确定性 parity 测试确认 compact 与完整历史读取的经济账本一致。2024-08 至 12 两折四候选实跑约 93 分钟完成 4 份 Validation；本轮新单折两候选约 35 分钟完成 2 份 Validation，工作集峰值约 4.4 GiB。Nautilus logger 降到 ERROR 已消除输出膨胀，但逐分钟 runner 推进成本仍高。L6 quota 仍是软件预留，不是 Windows Job Object 硬限制；未记录稳定的进程峰值/commit/RSS profile，也没有 OOM/kill/磁盘耗尽恢复证据。90 日运行边界尚未知。


**后续任务：** 对比 1/15/90 天峰值 RSS、扫描 CPU 时间和经济账本 hash；对 84.8 MiB 总报告改用明细外置与哈希引用；在保持 parity 的前提下评估增量 event cursor。任何进一步 purge 都必须保住 NETTING 往返成交、资金费和 liquidation 身份。

### F17 · 已修复核心口径 · P1 WFO 风险调整归因

旧代码将 soft funding/turnover 调整和 hard guard 调整合并成一个连续 `risk_guard_intervention`，应用层将任意非零连续值计为一次，WFO 又把它用于 `risk_violations`。现在 `RiskActionReview` 在 Gym 层保留 total/soft/hard adjustment；历史工厂明确分解两段，env 暴露组件及分类可信标志；应用报告分别给出软调整周期/幅度、硬守卫调整周期/幅度。WFO selector 和 native ES fitness 只将已分类 hard adjustment 计入约束；遇到来源不明的分类时失败关闭。测试覆盖软 funding 缩放而硬调整为零，以及未分类候选拒绝选择。真实 ETHUSDT 12 小时验证也观察到 718 个软调整周期、硬守卫为零，与修正语义一致。

v5 WFO 的候选风险字段仍是旧版，因 Holdout 已消费，不用同一窗口重跑来刷新研究结果。其后在 2024-04-15 至 2024-07-31 Binance 归档窗口完成一折 WFO，确认候选与 Holdout 已携带新版硬风险分类字段；该窗口结果为负收益，且 Test 对照 B0 为负。2024-08-13 至 2024-11-16 的未触碰窗口现已验证净收益选择门槛会在 Test 前拒绝全部负收益候选。**剩余验证：** 用一个有候选满足 Validation 门槛的未触碰窗口验证 Holdout return gate，再扩为多折，评估选择稳健性和 PBO。

### F18 · 已修复 · P1 WFO 失败退出未终结 ResearchLedger

2024-08-13 窗口的真实归档 WFO 在所有候选未达到收益门槛时正确停止，但旧异常路径让协议保持 `DEVELOPMENT`。`ResearchLedger.abort()` 现只允许非终态协议转入 `ABORTED`，并把失败类型/摘要写入哈希审计链；WFO 退出清理现捕获 `BaseException`，包括 Ctrl+C 的 `KeyboardInterrupt`，空详情也转换成非空原因。晚间多折尝试实测确认 Ctrl+C 曾遗留 DEVELOPMENT；随后已把该账本转为 `ABORTED`，写入 `FAILED.json`，Validation query 1 条、无 Test/Holdout，审计链有效。新增 KeyboardInterrupt 故障注入回归通过。普通 no-feasible-candidate 终止路径仍由前述 2024-08-13 真实运行验证。

## 5. 研究结果的正确解释

六类场景运行证明：当前代码能把归档市场事件送入 historical data/feature/Gym/策略/风险/planner/Nautilus/账户指标路径，并写出可校验产物。它们没有证明：

1. 2024 年所用合约、交易费、资金费和 margin/liquidation 假设是真实历史事实；
2. 风险分类修复后的 WFO 已在独立 Binance 行情窗口完成；新净收益选择门槛也在另一个窗口验证为失败关闭，Holdout return gate 仍待有可入选候选的未触碰窗口验证；
3. RL、ES 或 Hybrid 对市场具有正收益或统计显著性；
4. 代码已在最新 11 个 wheel 的干净安装环境运行；
5. 设计中的所有 RL observation、GPU ES、阶段恢复、资源/发布门禁已完成。

因此项目状态应表述为：**源码集成、CPU 回归、四种 RL/A1/A4 的真实行情短窗流程通过；本轮 PPO、ES、Hybrid 均完成各自独立日期的 Test CLI 回放，ES 与 Hybrid 的新 Test 结果分别为负和小幅正；这些单日情景结果不代表策略显著性。旧版全流程 WFO 亏损，新版独立窗口验证确认负收益候选会在 Test 前被拒绝；Holdout 收益门槛尚无符合 Validation 条件的候选可验证；正式历史经济规则、策略收益能力和完整发布验收未通过。**

## 6. 建议开发顺序

1. **取得可核验的历史经济规则（F04/F06）**：历史费率、资金费、保证金、清算和执行依据未齐之前，只能做 scenario-only 分析。
2. **验证修订后的 WFO（F07/F17）**：用独立、未触碰且至少有一名合格候选的窗口验证 Holdout return gate；在可用跨度允许时扩至多 fold，报告 PBO 与选择稳定性。
3. **完善长时运行与产物（F16/F11）**：对比 1/15/90 天 RSS 与 CPU，增加 Windows 硬资源上限/异常恢复，WFO 明细改为外置哈希文件，控制 84.8 MiB 总报告体积。
4. **关闭工作流恢复与 CUDA ES（F09/F10）**；完善 bundle 和 clean-wheel 安装（F12/F13）。
5. **重建并验证当前 11-wheel 集合（F13）**，在干净环境跑 CLI、真实数据重载和恢复检查。
6. **补固定预算 perf（F14）并将源码清单接入发布 attestation（F15）**，再提交 Q0–Q9 的同一发布候选审查。

## 7. 审计证据路径

- 上一轮逐层源码发现及原始探针说明：[AUDIT_L1_L6_20261011_CURRENT.md](<D:/Program Files/NautilusTrader_RL_ES/docs/AUDIT_L1_L6_20261011_CURRENT.md>)。
- 本轮探针结果：[audit_functional_probes_20261011.json](<D:/Program Files/NautilusTrader_RL_ES/qa/reports/audit_functional_probes_20261011.json>)。
- 本轮完整 WFO scenario 产物：[wfo-report.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-60-15-15-linear-v5/wfo-report.json>)、[holdout-report.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-60-15-15-linear-v5/holdout-report.json>)、[research.sqlite](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-60-15-15-linear-v5/research.sqlite>)；报告 SHA-256 sidecar 均与文件内容校验匹配。
- 本轮追加的真实行情 WFO：[wfo-report.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-apr-jul-v1-run2/wfo-report.json>)、[holdout-report.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-apr-jul-v1-run2/holdout-report.json>)、[research.sqlite](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-apr-jul-v1-run2/research.sqlite>)；导入范围清单：[import.json](<D:/Program Files/NautilusTrader_RL_ES/data/binance_ethusdt_20240415_20240731/import.json>)。此 WFO 在 SelectionSpec 收益门槛新增前运行，供负收益缺陷取证。
- 新门槛的真实归档验证：[FAILED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-aug-dec-min5/FAILED.json>)、[research.sqlite](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-aug-dec-min5/research.sqlite>)、[导入清单](<D:/Program Files/NautilusTrader_RL_ES/data/binance_ethusdt_20240812_20241201/import.json>)；该 WFO 有意在 Test 前停机，没有 Holdout 报告。
- 最新 11-wheel 构建与文件 parity：[wheelhouse manifest](<D:/Program Files/NautilusTrader_RL_ES/qa/reports/wheelhouse-five-routes-20261013.json>)；wheel files 位于 `.tmp/wheelhouse-five-routes-20261013/`，隔离 wheel import provenance、CLI smoke 与全量回归通过；完整外部依赖 offline install 尚未通过。旧集合仍记录于 `wheelhouse_current_20261013.json`，不代表当前源码 identity。
- 新窗口 Holdout 前拒绝：[PPO Validation](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-ethusdt-oct-jan-2024-60-15-15-v3/run/folds/60-15-15-0000/validation/ppo-small/report.json>)、[SAC Validation](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-ethusdt-oct-jan-2024-60-15-15-v3/run/folds/60-15-15-0000/validation/sac-small/report.json>)、[research ledger](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-ethusdt-oct-jan-2024-60-15-15-v3/run/research.sqlite>)、[FAILED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-ethusdt-oct-jan-2024-60-15-15-v3/run/FAILED.json>)；Validation 2、Test/Holdout 0，audit chain 有效。
- 86,400 步完整训练预算探索性 WFO：[配置](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/config-wfo-real-fullpass-60-15-15-v1.json>)、[WFO 报告](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-fullpass-60-15-15-v1/run/wfo-report.json>)、[PPO Validation](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-fullpass-60-15-15-v1/run/folds/60-15-15-0000/validation/ppo-fullpass/report.json>)、[SAC Validation](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-fullpass-60-15-15-v1/run/folds/60-15-15-0000/validation/sac-fullpass/report.json>)、[research ledger](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-fullpass-60-15-15-v1/run/research.sqlite>)、[FAILED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-real-fullpass-60-15-15-v1/run/FAILED.json>)；PPO/SAC 各 86,400 步，Validation 2、Test/Holdout 0，账本 ABORTED、audit chain 有效，报告 SHA 与 sidecar 匹配。Validation 日期已在较早协议使用，故结果只能作探索性证据。
- 两折四候选最新 WFO：[PPO fold 0](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-sep-dec-2024-quiet-v1/run/folds/60-15-15-0000/validation/ppo-small/report.json>)、[SAC fold 0](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-sep-dec-2024-quiet-v1/run/folds/60-15-15-0000/validation/sac-small/report.json>)、[PPO fold 1](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-sep-dec-2024-quiet-v1/run/folds/60-15-15-0001/validation/ppo-small/report.json>)、[SAC fold 1](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-sep-dec-2024-quiet-v1/run/folds/60-15-15-0001/validation/sac-small/report.json>)；[research ledger](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-sep-dec-2024-quiet-v1/run/research.sqlite>) 为 ABORTED、Validation 4、Test/Holdout 0，审计链有效；`FAILED.json` 记录因无候选达到收益门槛退出。每个 bundle/report sidecar hash 均已逐一复核匹配。
- Nautilus 长跑日志抑制配置及单步真实 engine 测试：[bootstrap.py](<D:/Program Files/NautilusTrader_RL_ES/packages/nautilus_adapter/src/rles_nautilus_adapter/bootstrap.py:45>)、[test_nautilus_adapter_smoke.py](<D:/Program Files/NautilusTrader_RL_ES/tests/test_nautilus_adapter_smoke.py:114>)；本次 WFO 参数：[config-wfo-fresh-sep-dec-2024-quiet-v1.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/config-wfo-fresh-sep-dec-2024-quiet-v1.json>)，Nautilus CLI stdout/stderr 文件为空，仅 WFO 结构化结果在 `run.log`。
- 公开 CLI 真实 Holdout 工件：[report.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-apr-jul-v1-run2/public-backtest-unused-output/report.json>)、[COMPLETED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-apr-jul-v1-run2/public-backtest-unused-output/COMPLETED.json>)。
- 当前代码的真实行情短窗重跑：[A0 PPO](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/ppo-risk-attribution-v1/report.json>)、[A0 SAC](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/sac-risk-attribution-v1/report.json>)、[A0 TD3](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/td3-risk-attribution-v1/report.json>)、[A0 DDPG](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/ddpg-risk-attribution-v1/report.json>)、[A1 ES](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/a1-risk-attribution-v1/report.json>)、[A4 Hybrid](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/a4-risk-attribution-v1/report.json>)；每项均有 bundle/report 哈希和完成标记。
- 新增 CUDA 实际行情工件：[A0 PPO](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/cuda-a0-20261012/report.json>)、[A0 SAC](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/cuda-sac-20261012/report.json>)、[A0 TD3](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/cuda-td3-20261012/report.json>)、[A0 DDPG](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/cuda-ddpg-20261012/report.json>)、[A4 CUDA-R1 Hybrid](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/cuda-a4-20261012/report.json>)；各 report/bundle SHA 均与 sidecar 和 `COMPLETED.json` 一致，且运行后资源 lease 已释放。
- 风险分类实现：[episode.py](<D:/Program Files/NautilusTrader_RL_ES/packages/gym_bridge/src/rles_gym_bridge/episode.py:58>)、[application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/application.py:104>)、[wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py:117>)。
- WFO runner：[wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py:160>)。
- WFO validation/holdout return gates：[selection.py](<D:/Program Files/NautilusTrader_RL_ES/packages/research/src/rles_research/selection.py:18>)、[wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py:160>)。
- public backtest report persistence：[application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/application.py:59>)。
- public CUDA R1 device resolution/resource allocation：[application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/application.py:216>)、[allocation.py](<D:/Program Files/NautilusTrader_RL_ES/packages/compute_runtime/src/rles_compute_runtime/allocation.py:15>)；preflight/GPU lease 回归：[test_application_cli.py](<D:/Program Files/NautilusTrader_RL_ES/tests/test_application_cli.py:17>)、[test_compute_device.py](<D:/Program Files/NautilusTrader_RL_ES/tests/test_compute_device.py:15>)。
- failed WFO protocol terminal state：[ledger.py](<D:/Program Files/NautilusTrader_RL_ES/packages/research/src/rles_research/ledger.py:154>)、[wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py:459>)。
- funding 日界裁切修复及回归：[binance_archive.py](<D:/Program Files/NautilusTrader_RL_ES/packages/data/src/rles_data/binance_archive.py:1>)、[test_binance_archive_range.py](<D:/Program Files/NautilusTrader_RL_ES/tests/test_binance_archive_range.py>)；本次衍生市场档案、来源 hash 和时间裁切清单位于 `.tmp/real-ethusdt-scenario/wfo-fresh-sep-dec-2024-v2/`。
- KeyboardInterrupt 终结修复与故障注入：[wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py:457>)、[test_research_wfo.py](<D:/Program Files/NautilusTrader_RL_ES/tests/test_research_wfo.py:176>)；被中断尝试的 [research.sqlite](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-sep-dec-2024-v2/run/research.sqlite>)、[FAILED.json](<D:/Program Files/NautilusTrader_RL_ES/.tmp/real-ethusdt-scenario/wfo-fresh-sep-dec-2024-v2/run/FAILED.json>)。
- funding 导入边界定向测试：`tests/test_binance_archive_range.py::test_import_range_fetches_each_day_and_month_once_and_clips_funding`；WFO 中断清理定向测试：`tests/test_research_wfo.py::test_wfo_marks_keyboard_interrupt_aborted_and_writes_failure_receipt`。
- Binance archive verified resume receipts：[binance_archive.py](<D:/Program Files/NautilusTrader_RL_ES/packages/data/src/rles_data/binance_archive.py:1>)。
- historical batching 与 compact snapshots：[historical.py](<D:/Program Files/NautilusTrader_RL_ES/packages/gym_bridge/src/rles_gym_bridge/historical.py:92>)。
- indexed event-time data access：[core.py](<D:/Program Files/NautilusTrader_RL_ES/packages/data/src/rles_data/core.py:505>)、[features.py](<D:/Program Files/NautilusTrader_RL_ES/packages/data/src/rles_data/features.py:32>)。
- compact native history and cache purge：[session.py](<D:/Program Files/NautilusTrader_RL_ES/packages/nautilus_adapter/src/rles_nautilus_adapter/session.py:77>)；历史配置：[perpetual.py](<D:/Program Files/NautilusTrader_RL_ES/packages/nautilus_adapter/src/rles_nautilus_adapter/perpetual.py>)、[historical.py](<D:/Program Files/NautilusTrader_RL_ES/packages/gym_bridge/src/rles_gym_bridge/historical.py>)。
- native account metrics bridge：[native_metrics.py](<D:/Program Files/NautilusTrader_RL_ES/packages/research/src/rles_research/native_metrics.py:22>)；应用调用：[application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/application.py:127>)。
- 当前 wheel 集合：`D:\Program Files\NautilusTrader_RL_ES\dist-current-verified-20261012`。


### 2026-10-13 独立新行情 86,400 步 WFO 复核

为取得此前本地样本结束日（2025-01-23）之后的独立证据，从 Binance Vision 导入 ETHUSDT USD-M 归档 2025-01-24 至 2025-05-10。导入清单记录 321 个日线来源 ZIP（K 线、mark、index）与 5 个月度 funding ZIP，共 326 个官方来源文件及 SHA-256；曲库共 462,884 个事件，覆盖 2025-01-24 00:01 UTC 至 2025-05-10 23:59 UTC。`market.parquet` SHA-256 为 `8858143724851eae85e7d052ee6bf3f189e68435fded006e3e28aae594777ddd`，导入清单 SHA-256 为 `1e0fc9288a78b18216a4766cbfb687edb338f3fcd412f2bb14fa200ca1889552`。数据不含该时期历史 instrument rules；规则、费用、保证金和滑点仍来自明确标注的 scenario-only 假设。

新配置以 2025-01-25 00:00 UTC 开始评分，执行一折 60/15/15，并按 8 小时特征回看插入 purge；Train 为 2025-01-25 至 2025-03-26，Validation 为 2025-03-26 08:01:01 至 2025-04-10 08:01:01，Test 为 2025-04-10 16:02:02 至 2025-04-25 16:02:02，独立 Holdout 为 2025-04-25 16:02:02 至 2025-05-10 16:02:02。所有窗口的行情、funding 与 scenario rule 覆盖专用预检通过。用主环境旧策略运行通用 `validate` 首次得到 DQ-D，原因是结算时间容差仍为 4 ms，低于本批归档核验采用的 15 ms；将独立校验配置同步为 15 ms 后，`validate` 通过、等级 DQ-B，funding_settlement/更新、mark、index 和 instrument_rule 均验证覆盖。WFO 专用分折门禁同样用 15 ms 模板并全部通过。

PPO 与 SAC 分别完成 86,400 步训练及原生 Nautilus Validation 回放。选择约束为最大回撤 95%、换手 1,000、风险干预 100,000、Validation 中位净收益至少 0%。两候选回撤和换手均低于上限、清算为 0，但净收益均为负，触发收益准入门而被拒绝：

| 候选 | Validation 净收益 | 最大回撤 | 换手比 | 硬风控干预 | 清算 |
|---|---:|---:|---:|---:|---:|
| PPO | -30.5067% | 30.5067% | 585.5311 | 344 | 0 |
| SAC | -11.1273% | 11.5881% | 173.5174 | 9 | 0 |

WFO 报告为 `NO_VALIDATION_CANDIDATE`，账本 `ABORTED`；2 次 Validation、0 Test、0 final retrain、0 Holdout，Holdout 保持 `LOCKED`。两份 Validation 报告和总报告的 SHA sidecar 匹配，ledger audit chain 验证通过。该窗口未被打开作为 Holdout；不能用其 Validation 表现再调参后声称独立验证。工件位于 `.tmp/real-ethusdt-scenario/binance-archive-2025-01-24-to-2025-05-10-fresh/`、`.tmp/real-ethusdt-scenario/wfo-fresh-2025-01-24-to-2025-05-10-v1/run/`，配置为 `.tmp/real-ethusdt-scenario/config-wfo-fresh-2025-01-24-to-2025-05-10-v1.json`。

结论更新：独立新行情的长预算实跑支持当前的保守准入行为，但尚未证明策略有正收益；本次因候选不合格而安全停止，并非完整项目设计通过。需要在尚未消费的新时期上重新开发/预注册候选与选择门，重点检查 PPO 的 344 次硬风控干预、SAC 的收益与动作/成本归因，并在任何独立 Test/Holdout 前冻结规则、执行假设与方案。不得下调收益门或读取 Holdout 来制造通过结果。

#### 新 WFO 成本与动作归因补记

Validation 明细分别有 21,598 个账户观测点；PPO 记录 20,792 笔成交、SAC 15,422 笔。报告标记成本已计入 native equity。交易前已报告成本收益（`gross_return_before_reported_costs`）为 PPO -0.9986%、SAC -1.2685%；按本次场景假设，PPO 手续费 2,446.85 USDT、滑点 503.76 USDT，SAC 手续费 816.92 USDT、滑点 168.52 USDT；资金费分别为 -0.20、-0.44 USDT。也就是说本轮强烈显示策略换手与假设执行成本侵蚀收益，但这不能证明实际交易所费率/滑点就等于这些设定。下一步开发应对动作稳定性、目标仓位变化、订单替换和成交成本分别做消融归因，在独立的新 Validation 区间评估降低换手的方案；当前窗口已经用于候选选择，不应用来迭代调参。


### 2026-10-13 WFO 执行死区参数优化与跨窗口验证

基于独立 2025 年新行情 Validation 显示的高换手/成本侵蚀，本轮为 WFO candidate 增加可选 `deadband` 参数，候选以 decimal string 声明。实现位于 [wfo_application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/wfo_application.py>)：只覆盖候选 `environment.execution.deadband`，通过 `ResolvedExecutionConfig.from_dict()` 校验完整执行配置；Training 的 fold window 和最终 pre-Holdout retrain window 都保留候选执行参数。候选不能分别修改费率、滑点、延迟或其他经济假设。回归位于 [test_research_wfo.py](<D:/Program Files/NautilusTrader_RL_ES/tests/test_research_wfo.py>)，覆盖候选配置隔离、非法阈值拒绝、fold window 传递、选中配置用于最终 retrain 及恢复路径。

使用同种子 SAC、相同网络/超参数、86,400 步及共同成本模型，完成两段此前没有本地归档的新 Binance Vision 数据对照。两个窗口均含 1 个 60/15/15 fold，funding 最大观测时间偏差为 16 ms，采用不改写源事件的 20 ms 容差；完整区间通用 validate 为 DQ-B，专用 WFO 覆盖门禁通过。

| 独立 Validation 窗口 | Deadband | 净收益 | 成本前收益 | 换手比 | 成交数 | 手续费 + 滑点（USDT） | 硬风控干预 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-07-11 至 2025-07-26 | 1% | -31.60% | -0.53% | 619.86 | 21,190 | 3,107.06 | 4,640 |
| 同上 | 5% | -24.97% | -0.60% | 466.16 | 17,685 | 2,438.52 | 414 |
| 2025-10-27 至 2025-11-11 | 5% | -20.49% | +1.92% | 417.64 | 15,275 | 2,242.42 | 4,204 |
| 同上 | 10% | -13.20% | +0.01% | 235.00 | 8,624 | 1,322.23 | 10,347 |

跨窗口对照显示较大的 deadband 降低了成交/换手和成本后亏损，但 5% 与 10% 均未达到 0% Validation 净收益准入门；10% 还伴随更多硬风控干预。两协议都以 `NO_VALIDATION_CANDIDATE` 结束，ledger `ABORTED`、各 2 次 Validation、0 Test、0 final retrain、Holdout `LOCKED`，两份候选报告及总报告 SHA sidecar 匹配，audit chain 有效。该参数仍需和动作参与度/风险介入一起评估；上述结果不是可接受策略表现。两个 Validation 窗口已被消费，不用于进一步调参。

新增验证命令：`pytest tests/test_research_wfo.py`（23 passed）、`pytest tests/test_economic_coverage.py tests/test_l4_provider_smoke.py`（27 passed）、`ruff check packages tests`（通过）、`mypy packages qa`（103 files 通过）。真实运行命令分别为 `rles_orchestrator --config <配置> validate`（DQ-B）及 `rles_orchestrator --config <配置> wfo`（按预期因无可行候选停止）。归档与 WFO 工件保存在 `.tmp/real-ethusdt-scenario/binance-archive-2025-05-11-to-2025-08-26-fresh/`、`.tmp/real-ethusdt-scenario/wfo-deadband-fresh-2025-05-11-to-2025-08-26-v1/run/`、`.tmp/real-ethusdt-scenario/binance-archive-2025-08-27-to-2025-12-12-fresh/`、`.tmp/real-ethusdt-scenario/wfo-deadband-005-vs-010-2025-08-27-to-2025-12-12-v1/run/`。

剩余核心项没有解除：策略仍无通过 0% 收益门的独立 Validation；费用/滑点/保证金/历史 instrument rules 仍是 scenario-only；该消融只测试 SAC，尚未验证 PPO、ES 与 Hybrid 在同等真实窗口上的选择/交易闭环；单折不能支持稳健的 PBO 估计；也未完成真实 Holdout、独立 Test、发布级干净依赖安装或跨参数验收。



### 2026-10-13 SAC 熵系数消融与新窗口 WFO

沿用“只在全新 Validation 窗口检验、收益门槛冻结为 0、未过门不读 Test/Holdout”的协议，新增 2025-12-14 至 2026-03-29 窗口，只比较 SAC `ent_coef="auto"` 与固定 `0.001`。两者使用相同 seed 62、86,400 步、网络、其余超参数、5% deadband、执行成本假设及 60/15/15 单折；候选、选择门和窗口在启动前冻结。

官方 Binance Vision 源归档 328 个（108 天 OHLCV、mark、index 各 108 个，加 4 个月 funding），导入 467,210 条事件。`market.parquet` SHA-256 为 `cc661b6dab471aa719c4424429f76191c1fb0aabd97a9ec46dbc8752b86da70f`，导入清单 SHA-256 为 `134a87572ebfc4684cea8cde2b19e46fc10a8f49e53e973b46a98130e11f9a57`。全区间 validate 为 DQ-B：0 缺分钟、0 重复、0 冲突，funding settle/update、mark、index 覆盖完整；instrument historical rule 证据仍为 0，经济执行参数仍是 scenario-only。

单一 Validation 窗口为 2026-02-12 至 2026-02-27；两个候选结果如下（费用、资金费及滑点已进入 native equity）：

| SAC 候选 | 净收益 | 成本前收益 | 最大回撤 | 换手比 | 成交数 | 手续费 | 资金费 | 滑点 | 硬风控干预 | 强平 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ent_coef=auto` | -12.347% | +0.093% | 12.368% | 220.455 | 9,185 | 1,032.93 | -0.08 | 211.06 | 15,864 | 0 |
| `ent_coef=0.001` | -0.266% | -0.249% | 0.949% | 0.200 | 21 | 1.00 | -0.48 | 0.20 | 0 | 0 |

两候选都被冻结的 `median_fold_net_return >= 0` 门槛拒绝。协议以 `NO_VALIDATION_CANDIDATE` 安全结束，ledger `ABORTED`、Validation 查询 2、Test 查询 0、最终重训 0、Holdout `LOCKED`；总报告/运行清单及候选报告哈希匹配，SQLite audit chain 校验通过。折分边界为训练 2025-12-14 至 2026-02-12、Validation 至 2026-02-27、Test 至 2026-03-14、Holdout 2026-03-14 16:02 UTC 起；后两段没有读取。

本消融表明 auto 候选在此窗口出现大量交易与风险干预；固定系数显著降低活动，但接近空仓且净收益仍为负。不得据此把 `0.001` 提升为默认参数，也不在已消费窗口继续调参。下步应优先开发可审计的动作幅度、成交频率、风险守卫触发和费用归因诊断，并在新的、预先冻结窗口上比较有因果动机的候选；不得以调宽收益门或打开 Holdout 寻找正结果。

配置与结果：`.tmp/real-ethusdt-scenario/config-wfo-sacentcoef-auto-vs-001-2025-12-14-to-2026-03-29-v1.json`、`.tmp/real-ethusdt-scenario/wfo-sacentcoef-auto-vs-001-2025-12-14-to-2026-03-29-v1/run/`。


### 回测动作参与与风控改写诊断

基于上述新 WFO 的两种行为极端（auto 高换手/高硬干预；固定熵系数近空仓），扩展 [application.py](<D:/Program Files/NautilusTrader_RL_ES/packages/orchestrator/src/rles_orchestrator/application.py>) 的 rollout 报告：并列统计原始请求目标与 terminal flatten 之前的实际执行目标，包括均值、绝对值均值/P95、非零参与率、两者绝对差均值/P95、目标被改写比例。报告现有的风险干预次数、软/硬调整金额仍保留。回归用隔离 fake environment 验证统计口径，未访问新窗口、未打开 Test 或 Holdout。

验证：`pytest -p no:cacheprovider --basetemp .tmp\pytest-basetemp-rollout-diagnostics tests\test_application_cli.py -q` 为 6 passed；Ruff 对改动文件通过；`mypy packages qa` 为 103 source files passed。已重建全部 11 个 workspace wheels，wheel/source 逐 Python 文件比较全部匹配；wheelhouse manifest 为 `qa/reports/wheelhouse-action-diagnostics-20261013.json`，SHA-256 `cc86da785dc8c0cc01b23daa9cd69a429a7b29c59ee4ee4c5b2ee9c1dba4be64`，绑定 source identity `a41856473bf7ffc6ac54738f36fa845fc233c456aac2f3d09985e34b13a544da`。`qa/source_manifest.py --verify` 验证 233 个纳入范围文件通过。旧 WFO 运行报告自然不含新增诊断字段，需在后续未消费的 Validation 执行中验证完整真实轨迹输出。


### 2026-10-13 新窗口完整 A0 SAC WFO（含 Test、基线、重训及 Holdout）

为验证动作诊断并推进真实端到端闭环，在此前未导入/未消费的 2026-03-31 至 2026-07-16 Binance Vision 官方归档上执行冻结的 A0 SAC WFO。数据含 467,210 个事件和 328 个原始归档；market parquet SHA-256 `b36bf03069ee20308cb3e6e73eaa3ba776c4070247fca83754dab3afaad3c489`，import manifest SHA-256 `ebbadf42a173c39e69d9a0c7b82a2630f588c88096d228e3302eac0470289c0c`。完整环境 validate DQ-B，缺分钟/重复/冲突均为 0；但 `instrument_rules_supplied=false`，规则与费率、滑点、保证金仍明确为 scenario-only。数据窗口覆盖训练 2026-04-01 至 05-31、Validation 至 06-15、单次 Test 至 06-30、Holdout 06-30 16:02 UTC 至 07-15 16:02 UTC。

候选在 Validation 前冻结为相同 seed 62、86,400 步、5% deadband 的 `ent_coef=auto` 与 `0.001`。auto 净收益 -13.999%、回撤 14.002%、换手 237.073、硬干预 5,088；固定 `0.001` 净收益 +0.051%、回撤 0.742%、换手 6.673、硬干预 0，按冻结 0% 收益门选择固定系数。其唯一 Test 净收益 +0.214%、回撤 0.409%、换手 5.236、803 笔成交、0 强平、终态平仓。Test 的 95% block-bootstrap 单期平均收益区间跨越 0，DSR 概率约 `1.04e-9`；单折时 PBO 为 null，不能视为稳健统计证据。

Test 同期 B0–B3 基线净收益：不交易 0.000%、始终做多 -1.572%、简单动量 -16.888%、随机动作 -29.025%。最终预 Holdout 模型完成 R1 retrain，bundle SHA-256 为 `74aaab8a5718a237cb2acba37edfd0a25b37fcf2828ece3069fdd0e8c4187ed9`；WFO 在重训工件核验后按准入协议一次性打开 Holdout。Holdout 净收益 **-1.565%**、最大回撤 1.591%、换手 11.292、1,656 笔成交、手续费 55.99 USDT、资金费 +0.91 USDT、滑点 11.48 USDT、0 强平、终态平仓；成本前收益 -0.899%，最终 return gate `FAIL`。虽略高于始终做多基线，但低于不交易基线且绝对收益门未通过，策略不合格。该 Holdout 已消费，不得重开或用于调参。

新 `action_diagnostics` 已在本次真实 Validation/Test/Holdout 轨迹写入并逐项核验。选中策略 Test 的请求目标绝对均值约为 0.350，执行绝对均值约 0.350，平均绝对差 0.0162；Holdout 请求 signed mean 为 **-0.428**、绝对均值为 **0.428**，执行绝对均值 0.42585，平均绝对差 0.01768；两个窗口请求非零率和被改写比例均为 100%，硬干预均为 0，主要是软风险/换手控制产生差异。Holdout 请求 signed mean 为负且其绝对值接近绝对均值，显示该段保持一致偏空动作；Validation/Test 表现轻微正向而 Holdout 转负，提示策略方向/状态适应性不足。此解释是对诊断数据的推断，不构成策略优化结论。2026-10-13 续开发依据最新审计纠正 Holdout signed mean 的漏写负号；本次未重放 Test/Holdout。

最终 WFO `conclusion=FAIL`，ledger `CLOSED`；2 次 Validation、1 次 Test、最终 retrain SHA、一次 Holdout SHA 均有记录；WFO 总报告 SHA-256 `73ee4589fccac83d4124e6df1490de0b53974f493e7b93a414962454f969c4f4`，Holdout 报告 SHA-256 `236e0b78c9ffd7922e68d765509f6b0812401a9e2302d932e225e7c235665db4`，所有检查的 sidecar 均匹配，ledger audit chain 有效。结果证明真实 A0/SAC 训练—保存—原生 Test—基线—最终重训—原生 Holdout 闭环已执行，但**没有证明项目设计验收通过**。

配置与工件位于 `.tmp/real-ethusdt-scenario/config-wfo-sacentcoef-auto-vs-001-2026-04-01-to-2026-07-15-v1.json` 和 `.tmp/real-ethusdt-scenario/wfo-sacentcoef-auto-vs-001-2026-04-01-to-2026-07-15-v1/run/`。命令为 `python -m rles_data.binance_archive --symbol ETHUSDT --start-day 2026-03-31 --end-day 2026-07-16 --output ... --delay-ns 1000000`、`python -m rles_orchestrator --config <配置> validate`、`python -m rles_orchestrator --config <配置> wfo`。

剩余核心工作转向：利用观测到的 Holdout 方向失效进行有因果依据的策略状态/动作改进；在未消费数据上验证 RL 各算法、独立 ES 与 A4 Hybrid；补足多折稳健验证及历史 instrument/经济规则证据。不得在已消费的 Apr–Jul 窗口重训、重选或调参。


### 2026-10-13 新鲜窗口 A1 独立 ES 与 A4 Hybrid 路由实跑

为补上此前缺少的 A1/A4 新鲜市场端到端证据，使用 Binance Vision ETHUSDT UM 官方归档，取 2026-08-01 至 2026-08-11 UTC 数据；拆分训练 08-01 00:00 至 08-06 00:00、Validation 08-06 至 08-08、独立 Test 08-08 至 08-11 00:01。完整区间 parquet snapshot SHA-256 `b8047fb7b918614f0b2bf88127692fb325bb9745f68fed79cb127a72061fbd4a`。训练/验证和 Test 均为 DQ-B，缺分钟、重复与冲突为 0；Test 有 4,321 根 OHLCV/mark/index、10 个 funding update/settlement。`instrument_rule=0`，因此历史规则、费率、滑点和保证金仍采用 scenario 参数，不能称为交易所实况模拟。

| 路由 / 阶段 | 净收益 | 最大回撤 | Sharpe | 换手比 | 手续费 / 资金费 / 滑点 (USDT) | 动作观察 |
|---|---:|---:|---:|---:|---:|---|
| A1 Validation | +0.00107% | 0.0132% | 0.675 | 0.0168 | 0.0841 / -0.0085 / 0.0176 | 请求与执行绝对均值约 0.0855 / 0.0840；请求全为正且非零 |
| A1 Test | -0.01913% | 0.0295% | -11.226 | 0.0167 | 0.0833 / -0.0295 / 0.0172 | 请求全为正且非零；平均请求/执行绝对差 0.00139 |
| A4 Validation | +0.000264% | 0.00570% | 0.363 | 0.0407 | 0.2034 / -0.0037 / 0.0416 | 平均请求/执行绝对动作 0.0971 / 0.0375，100% 记录为有调整 |
| A4 Test | -0.004999% | 0.00621% | -6.650 | 0.0881 | 0.4405 / 0 / 0.0893 | 平均请求/执行绝对动作 0.0963 / 0.00779；执行非零率仅 10.77%，平均差 0.0885 |

两条路由均以真实导入数据完成训练、验证和 bundle 加载后的独立 Test；Test 均 0 次强平、期末平仓，训练及 Test 报告、bundle 的 SHA-256 sidecar 与文件实算一致。A1 实际 lineage 为 E1 (4 次评估、1 代)， actor kind `es`。A4 lineage 为 E1 (4 次评估、1 代)、R1 (64 PPO 步)、E2 (4 次评估、1 代)，actor kind `rl`；保存的 E2 soft multipliers 为 drawdown 0.5085、exposure 0.6403、volatility 0.5728。其 Test 动作缩减明显，需查明目标仓位到订单/执行目标的各层映射及 E2 参数来源，不能仅凭净收益推断 Hybrid 策略质量。

这组短窗口主要证明导入、训练、bundle 完整性、Validation 评估、独立 Test 和安全终态的路由可以执行；超小预算（ES 每阶段只有 4 个评估、1 代；A4 PPO 64 步）不具备收敛或策略质量证明力。两个 Test 已消费，不得据其结果继续改参后再报告为独立 Test。A1 Validation/Test 请求始终为单侧正动作，A4 Test 的低有效执行率及成本后负收益是后续重点。应在未消费窗口先冻结假设，再测试动作参与度、执行缩减归因及多折泛化；不得调低当前冻结的收益准入门。具体运行配置和工件在 `.tmp/real-ethusdt-scenario/config-a1-fresh-2026-08-train-validation-v1.json`、`config-a1-fresh-2026-08-test-v1.json`、A1/A4 各自 `*-fresh-2026-08-train-validation-v1/` 输出目录内。


### A4 风控动作链分层归因补充

针对上节 A4 的请求目标与实际仓位差异，风险审阅对象现在保留 soft-risk 中间目标和 hard-risk 后目标；Gym info 暴露这两个值，rollout 报告新增 `risk_target_diagnostics`，分别汇总 request→soft、soft→hard-risk 和 hard-risk→实际执行仓位的平均绝对差。若使用旧版注入 planner 且无法给出中间目标，则报告 `components_known=false`，避免把账户实际仓位误认为风险目标。变更涉及 `packages/gym_bridge/src/rles_gym_bridge/episode.py`、`historical.py`、`env.py`、`packages/orchestrator/src/rles_orchestrator/application.py` 及对应单元测试。

验证命令 `python -m pytest -p no:cacheprovider --basetemp .tmp\pytest-basetemp-risk-targets tests\test_application_cli.py tests\test_historical_factory.py tests\test_native_training_episode.py -q`：52 passed；Ruff 改动文件检查通过；`python -m mypy packages qa`：103 source files passed。新字段尚未在历史 A4 Test 报告中出现，因为该 Test 已消费且未重跑；须在新的未消费窗口验证其数值分解。该改动只提升风险目标到执行结果的可审计性，不改变策略、风险参数或准入结论。


本次最终源码清单验证通过：233 个纳入范围文件，source identity `411ec836baaf23c6bef1ac50a0a6965932143c78f7454863ca82ff925e43a08b`。已重新构建 11 个 workspace wheel 并逐包比较源码：全部匹配；wheel manifest 为 `qa/reports/wheelhouse-risk-targets-20261013.json`，SHA-256 `b07368dca4dd0de097f6224c321e689af636a2f802b25404cfeb6c0161fe80ba`。wheel 文件在 `.tmp/wheelhouse-risk-targets-retry-20261013/`。

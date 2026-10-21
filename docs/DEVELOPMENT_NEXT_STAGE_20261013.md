# 审计后的下一阶段开发目标（2026-10-13）

**最新续开发：** N04 的 Windows CPU worker 父进程硬退出恢复已实现，目标、严格边界及真实原生故障证据见 [DEVELOPMENT_RESOURCE_RECOVERY_20261013.md](DEVELOPMENT_RESOURCE_RECOVERY_20261013.md)。协调器/learner OS 内存边界、旧/未完成启动 lease 与 GPU 恢复仍未关闭。

## RL 观测设计续开发（2026-10-13）

针对审计中 Holdout 持续偏空、而相邻 Validation/Test 偏正向的方向失效，候选 profile `causal_mtf_v2` 在 v1 的 5/15/60 分钟收益及 15/60 分钟波动特征上，追加 15/60 分钟趋势效率：区间累计对数收益除以逐分钟绝对对数收益之和，范围 `[-1, 1]`，零路径映射为 0。它区分同样累计收益下的单向趋势与反复震荡，完全由已闭合的一分钟 bar 构造，缺口失败关闭；v1 和 legacy 的特征字段保持不变，profile/scaler/schema 仍由模型身份绑定。工程回归覆盖方向持续性、震荡效率、区间界限、缺分钟拒绝及未来 bar 扰动不影响当前观测。

该改动只增加可用的因果状态输入，不是已验证的收益提升。新近 2026 数据跨度不足完成原定多折方案且缺少历史 instrument rule，旧的 2024/2025 验证窗口已被探索或消费；不得据这些窗口反复挑选 v2。应在新的、尚未查询的足够长数据上预注册 v1 对 v2 的配对多折比较、成本压力和动作方向诊断。

验证结果：两个新增特征探针通过；全仓 pytest **401 passed、4 CUDA-only skipped**；`ruff check packages ci qa tests` 通过；`mypy packages qa` 为 **107 source files** 无错误。全套回归证明兼容与工程边界，不代表 v2 的策略收益或泛化效果。

## WFO 配对执行成本压力验证（2026-10-13）

WFO 现在默认对每个已训练候选、每个折运行配对 stress Validation：同一候选 bundle、Train-only scaler 和同一 Validation 时间段，执行模式切为配置中冻结的 stress 延迟、滑点及费率倍率。历史 `instrument_rule` 费率事实保持不变；仅 stress 原生手续费模型会放大正费率，maker rebate 不会被扩大。费率倍率进入执行指纹、scenario hash 和报告，基准/训练回放始终使用原始历史费率。当前示例配置为 2x 正费率敏感性情景，属于未校准压力假设，需在经济规则数据到位后根据真实费率和账户等级修订。额外查询在回放前写入研究账本，报告/receipt 绑定 bundle SHA、折、时间段和 stress mode；续跑时已预留但缺少完整报告的 stress 查询失败关闭，不会免费重放。选择前要求每一折的 stress 净收益达到现有 SelectionSpec 收益门（默认 0）、回撤/换手上限、0 强平、硬风控门及终态平仓；未过者不能进入 Test。协议可显式设 `wfo.cost_stress_validation=false`，但该 WFO 不具备执行成本稳健性准入证据。

验证：WFO 集成回归构造“base Validation 为正、stress Validation 为负”，确认两个候选在 Test 前均被拒绝，账本计入四次 Validation 查询且审计链有效；另验证 backtest 仅在显式 stress 授权下接受与 base bundle 只差 mode 的工厂身份。后续补入压力费率倍率（默认示例 2x 正费率敏感性），验证历史规则不变、原生手续费按场景倍率应用、rebate 不扩大，且倍率/hash 出现在回测报告。执行桥接、原生适配器、历史环境、CLI、WFO 回归 **130 passed**；Ruff 通过；MyPy **101 source files** 通过。该工程阶段当时没有真实策略压力结果；后续实际配对探索结果见下一节。新观测数据仍受 DQ-C/历史规则缺失限制，未据此运行或宣称 WFO 收益结果。

## v1/v2 实际配对探索与下一轮动作/奖励工作（2026-10-13）

为检验新增趋势效率观测是否值得继续，在已验证可用于场景研究的 ETHUSDT 分钟数据上运行同一 WFO 折、相同 SAC/seed/deadband/86,400 步预算和 2x 正费率 stress，唯一候选差异是 `causal_mtf_v1` 对 `causal_mtf_v2`。该日期区间此前已有 Validation 探索，因此本次结果仅是重复使用窗口的开发诊断，不能作为独立泛化或部署证据；配对结果不得再作为新参数搜索的独立确认。

| 候选 | Base Validation 净收益 | 最大回撤 | 换手比 | 硬风控干预 | 2x 费率压力净收益 | 压力换手比 |
|---|---:|---:|---:|---:|---:|---:|
| SAC + `causal_mtf_v1` | -29.40% | 29.40% | 580.49 | 8,700 | -71.18% | 829.37 |
| SAC + `causal_mtf_v2` | -26.44% | 26.44% | 508.58 | 6,068 | -63.44% | 670.95 |

两候选均因 stress Validation 净收益低于 0 被 `NO_FEASIBLE_MODEL` 拒绝；`fold_tests=[]`、`holdout=null`、`holdout_state=LOCKED`，ledger 记录 4 次 Validation、0 次 Test、0 次 Holdout。v2 在这个已重复使用的窗口上相对损失较小、换手和干预较低，但仍大幅亏损，不能解释为可靠收益改善。动作诊断显示两版每个决策都输出非零动作；v2 请求绝对动作均值 0.629，P95 0.940，short/long 比例 58.4%/41.6%，风险调整后实际敞口均值 0.229，请求与实际间隙均值 0.401。请求到 soft-risk 目标的均值间隙 0.396，而 soft 到 hard-risk 目标仅 0.0033、risk 到成交持仓仅 0.0016；在 21,597 个步骤中出现 20,493 次 `SUBMIT`，原因主要是 rebalance (14,425) 和 hard-risk reduce (6,068)，deadband 仅 558 次。说明这一折更明显的问题是动作变化密集、软风险显著压缩和成本侵蚀；尚不能仅凭该窗口区分策略过度换手、策略预测失效或 scenario 成本偏高。下一轮应优先在新 Train 段测量风险管线可达性及持仓/方向变化，再单独消融动作映射/缩放；硬风控保持不变。最终 r3 报告为 `.tmp/real-ethusdt-scenario/wfo-feature-ablation-v1-v2-2025-05-11-to-2025-08-26-r3/run/wfo-report.json`。其收费、保证金、价格步长等仍是 scenario-only；reduce-only 最小名义金额豁免依据 Binance USDⓈ-M 合约规格的公开说明，仅应用于场景规则中的这一项，最小数量和步长约束仍保留。

两次启动问题也保留在审计工件中：首次规则场景遗漏 reduce-only 最小名义金额例外，规划器按规则正确拒绝小额硬风险减仓；r2 修正规则来源回执后，真实运行暴露 `HistoricalEnvironmentFactory` 允许列表漏掉 `causal_mtf_v2` 的实现错误。现已修复入口白名单并增加真实历史工厂 schema 回归。相关 `test_historical_factory.py` 与 `test_research_wfo.py` **63 passed**，`ruff check packages ci qa tests` 通过。r1/r2/r3 都使用独立目录和协议，失败及已消耗查询均未覆写。

下一步应针对训练机制而不是在旧验证折继续调参。当前动作是连续标量 `[-1,1]`，对应风险预算比例的有符号目标净敞口；soft risk、hard risk 和执行 planner 再变换/裁剪目标，deadband 与最小数量/名义金额决定能否下单。奖励主体为原生账户权益的逐步对数收益，交易费、资金费、滑点通过 native account 进入收益；回撤增长/风控干预 shaping 系数默认 0，破产有单独负惩罚。旧探索 Train 曾表现持续偏空和目标被软风险大幅压缩；本次配对验证还显示极高换手、硬风控干预与大幅净亏损。不得用“增加交易奖励”或直接降低风险保护掩盖这一点。

后续优先在新的未查询 Train 段做机制消融：先测 actor 输出方向/分布和环境状态相关性，定位 soft-risk、hard-risk、deadband、lot/minNotional 对有效动作的截断率；在保持净权益回报的前提下比较动作映射/尺度与执行可达性的候选，并用 Train-only 多种子诊断确定是否减少无效反复调仓。奖励惩罚 shaping 仅作为单独、预注册候选，与原始 `lambda_dd=lambda_guard=0` 基线并行，必须报告净收益、回撤、换手、干预和空仓时间，防止模型靠不交易刷风险指标。只有取得新的独立多折和原门槛下的压力 Validation，才允许一次性 Test/Holdout；如无足够新窗口和可靠历史经济规则，则只做训练机制研究，不宣称实际可交易盈利。

审计显示当前市场输入只有 1 分钟收益、5 分钟波动/成交量、基差和最近结算资金费率；Train-only A4 诊断中请求动作绝对均值约 0.236，软风险后约 0.038，动作方向持续偏空。为让策略能够表达更长时段的趋势与波动状态，历史环境新增可选 `environment.feature_profile="causal_mtf_v1"`：在原五个市场特征后添加 5/15/60 分钟闭合 K 线收益和 15/60 分钟滚动波动率。每项均由决策时可见的一分钟 OHLCV 构造，断档/未闭合窗口按既有 L2 规则拒绝； scaler 仍只在 Train 拟合，profile 由 FeatureSpec/schema/transition 指纹绑定。

默认 `legacy` 保留既有观测顺序和模型输入兼容性。新 profile 会产生不同特征、scaler 和模型身份，旧 bundle 不能冒用新 schema。当前只完成候选输入配方的工程接入，尚无学习效果或收益证据；不可把维度增加视为改进已被验证。下一步须在未消费窗口预先冻结 profile、算法/预算、候选数、成本假设和收益门，进行多折比较，并同时报告动作方向响应、软/硬风险缩减、成交参与、费用后净收益和不交易基线。奖励保持原生权益对数回报；回撤/干预 shaping 暂不改默认值，待独立 Train 消融确定是否改善而非诱导空仓后再议。

实现回归：`tests/test_historical_factory.py tests/test_research_wfo.py` **59 passed**。新增覆盖旧字段前缀稳定、profile 字段及 61 分钟预热、60 分钟收益计算和未来行情扰动不影响当前特征。该结果验证工程与因果边界，不代表市场收益验证。

### 当前数据续接：未完成月份资金费率来源

Binance Vision 的当月资金费率月归档在归档完成前可能返回 404。导入器现仅在“请求范围最后一个月份就是当前 UTC 月份”且归档返回 404 时，回退到官方 USDⓈ-M `GET /fapi/v1/fundingRate`；历史月份缺档仍失败关闭。REST 原文单独落盘，`import.json` 记录精确 URL、响应 SHA-256 与 `captured_response_sha256`，不冒称归档校验和。接口不提供历史 interval 字段，导入器以相邻结算时间估算分钟间隔，容忍毫秒时间戳在整分钟附近 1 秒内的抖动；乱序、重复和明显非整分钟间隔拒绝。

用该路径建立隔离数据包 `.tmp/rl-feature-ablation-20261013/market-with-rest-funding/`：ETHUSDT 2026-08-11 至 2026-10-11，共 89,280 根 1m OHLCV、mark 和 index，资金费率 update/settlement 各 185 次；重复、冲突、分钟缺口均为 0，snapshot `4d76bf53a6417b5b8c928ae26cf2a10125f3213eebce307bacc404c2bc3e2f58`。质量仍因缺少该区间适用的 `instrument_rule` 与经证实账户执行规则而不能作为正式经济回测，费率/滑点/保证金仍是场景参数；本轮不训练选参，不宣称其为盈利证据。正式研究开始前仍要核对数据使用账本、锁定至少三个时序折及所有预算/准入条件。

导入器、因果观测与 WFO 相关回归 **75 passed**；Ruff 和 MyPy（107 个源码文件）通过。外部 API 来源说明：[Binance USDⓈ-M Funding Rate History](https://developers.binance.info/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)。

追加实际行情特征探针：在该包 2026-08-12 至 2026-10-11 每 7 天取一个已闭合决策点，共 9 个；10 个特征均有限、`available_at_ns <= decision_ts_ns`，5 个新增 MTF 特征均成功计算。该探针验证真实数据上的因果/数值可用性，不验证预测力。由于 DQ-C，`HistoricalEnvironmentFactory` 的 `rl_es` 用途会拒绝此数据；解除这一门槛需要有来源和适用日期的规则事实，不能仅修改质量标签。严格 WFO 仍锁定 60/15/15、15 天步长；以 2026-08-11 为起始、三折加 15 天 Holdout 估算，完整区间最早需延伸至约 2026-12-24，另加实际 purge/embargo。故目前既不具备多折时间长度，也不具备 RL/ES 训练授权；后续数据到齐和经济事实补齐前只推进工程与机制研究。

依据 `AUDIT_REPORT_20261013.md`。本阶段保持 RL、ES 独立，NautilusTrader 上游源码不改动；策略收益门、一次性 Test/Holdout 和历史经济证据门保持原要求。

## 目标与验收

| 编号 | 优先级 | 开发目标 | 验收条件 / 依赖 |
|---|---|---|---|
| N01 | P0 | Train 机制诊断：方向偏置、软/硬风险缩减、deadband 和订单最低限制 | 公共命令复用 bundle 的原 Train 窗口与 Scaler，按 planner 结果分组归因；不训练、不拟合、不查询 Validation/Test/Holdout，不修改原工件 |
| N02 | P0 | 根据 Train 证据冻结下一轮 RL / ES / Hybrid 配方及多折研究 | 启动前记录因果假设、数据使用台账、候选、seeds、足够的训练预算、成本/风控和原收益门；至少三个合法时序折用于选择稳定性研究；窗口新鲜性须由既有账本确认，不能只凭目录名 |
| N03 | P0 | 历史合约与账户经济事实 | 提供有生效日期、来源及适用账户的 tick/step/min quantity、费率、保证金阶梯和清算依据；未覆盖则严格门保持关闭，报告继续标为 scenario-only |
| N04 | P1 | 父进程硬退出后的资源恢复与协调器/learner 边界 | 明确区分 SQLite 配额与 OS 限制；自有进程树故障注入、残留 lease 盘点、显式 owner/token 恢复和重复恢复拒绝；未知 GPU owner 不自动释放 |
| N05 | P1 | 90 天吞吐与 CUDA 验收 | 冻结工作负载并记录 wall time、RSS/commit、磁盘增长；CUDA 必须使用实际可用硬件，CPU 路径不能代替 GPU 通过 |
| N06 | P1 | 修复 nightly 的空 perf lane 与恢复覆盖 | 固定预算的真实编码测量用例可收集、执行、产出时间/内存证据；WFO 一次性边界、RL/ES/native 恢复进入对应 lane，结果和测量可下载 |
| N07 | P2 | 审计口径维护 | 修正 Holdout 请求 signed mean 的负号；区分本轮验证、历史验证、策略结果和未完成项 |

## 本轮实施范围

先完成 N01、N06、N07，并用既有探索 Train 检验诊断功能。N02–N05 保留为后续验收目标；本轮不把旧窗口重新命名为新鲜窗口，不开启新的 Test/Holdout，也不把短预算/零成交视为策略合格。

N01 新增 `rles diagnose-train --config <原配方、全新 output> --bundle <已保存模型目录>`。该入口强制使用 bundle 的完整原 Train 窗口，复用训练 Scaler 和模型输入语义，并保留既有 data/risk/execution 身份检查。`evaluation` 不参与该入口；原文件级数据完整性检查可能读取含其他区间的共享档案，但原生 episode 与特征拟合不会进入其他评分窗口。

报告增加 `execution_attribution`：按 `(execution_plan_action, execution_plan_reason)` 分组，分别记录请求、软目标、硬目标、实际仓位、方向比例及各段绝对差。缺失风险目标保持 unknown，部分覆盖单列分母；不把 SUBMIT 当成交，不把价格移动导致的仓位差直接归咎于 planner。回放无需语料时不再复制、堆叠整段 observations；E1→R1 所需语料收集保持开启。

N06 性能预算在运行前声明：10,000 个固定大小 mark 事件，单独测量 canonical snapshot 编码，流式临时 Python 分配 <256 KiB、<整段物化参考的 10%，编码 <20 秒且摘要完全相同。记录两条路径的耗时和峰值；这是局部内存回归门，不是整应用 RSS、速度提升或 90 天验收证据。

## 诊断前冻结的探索运行

使用已训练的 `es-input-scale-20261013/a1-train-validation` 和 `windows-job-native-20261013/a4-train-validation` bundle，原真实 Binance ETHUSDT 2026-08-01 至 08-06 UTC Train，完整 5 天、原 seeds、原成本、原 deadband 和风险参数。只读取模型、复用 Scaler、原生回放并写入全新 `.tmp/audit-next-stage-20261013/` 子目录。两者为已使用的探索训练数据，不能充当新的独立验证。报告用于下一轮假设形成，本轮不根据结果改配方重训。

## 实施结果

N01、N06、N07 已实施。代码位于 `packages/orchestrator/src/rles_orchestrator/{application,diagnostics}.py`、`tests/test_snapshot_perf.py`、相关恢复测试及 `.github/workflows/l6-ci-reusable.yml`。

真实 Train-only 机制证据（已使用的探索 Train，各 7,197 个决策）：

| 模型 | 请求绝对均值 | 软/硬风险后目标绝对均值 | 实际仓位绝对均值 | deadband / SUBMIT | 请求偏空比例 |
|---|---:|---:|---:|---:|---:|
| A1 ES | 0.107703 | 0.107652 | 0.112933 | 7,137 / 60 | 100% |
| A4 Hybrid | 0.235836 | 0.037884 | 0.046914 | 7,194 / 3 | 100% |

两模型本段的软→硬平均差均为 0，未观察到硬风险进一步缩减；A4 请求→软目标平均绝对差为 0.197951，主要缩减发生在软风险层。实际仓位会随 mark 变化，deadband 表示没有新计划，不代表空仓：A1 实际仓位非零比例 100%，A4 约 78.42%。不可从分组摘要断言降低 deadband 就能提高收益，亦不可将三次 SUBMIT 等同于三笔成交。

两次运行 exit 0、0 清算、终态平仓，未拟合 Scaler、未打开评估查询；原 bundle/完成回执/研究及配额 SQLite 文件前后完全一致。A1/A4 本机回放约 10.64 / 9.42 秒，是诊断工作负载耗时，不能与训练吞吐混用。规则继续为 scenario-only。

下一轮待冻结的研究假设：H1，持续单边输出是否能随 Train 内市场状态变化，而不只随账户量纲变化；H2，E2 的风险惩罚与乘法缩减是否使目标长期低于可执行变化范围。应先在 Train 上分段与固定参数消融，预先记录候选与预算；保留无交易基线、原硬风控、费用和收益门，不给交易次数奖励或强制开仓。待数据使用台账确认新的独立窗口后再推进 N02，本次没有启动新鲜研究或修改已冻结配方。

真实诊断证据：`.tmp/audit-next-stage-20261013/acceptance.json`、两组 `config-*.json`、各自 `report.json` / `.sha256` / `COMPLETED.json`；复现脚本 `.tmp/audit_next_stage_diagnostics_20261013.py`。脚本拒绝覆盖原输出，重跑需改用全新根目录。

针对性回归 **45 passed / 18 warnings / 48.53 秒**；Ruff 全目录通过，现有 MyPy 范围下 `packages qa` **105 source files** 通过。性能探针 10,000 个事件摘要完全一致，流式临时 Python 峰值 **8,619 字节**、物化参考 **9,476,146 字节**，耗时 **0.794 / 0.646 秒**；这只证明局部内存改善，未证明加速。

最终全量 CPU 回归：**380 passed、4 skipped、45 warnings，279.41 秒**。4 项跳过均为 `tests/test_cuda_smoke.py` 的 CUDA 不可用；第三方 warnings 与审计数量相同。本轮未取得新的 CUDA 验收证据。全量运行的性能峰值仍为 8,619 / 9,476,146 字节，耗时 0.784 / 0.640 秒。

marker 收集：`perf` **1**、`short_wfo` **10**、`artifact_restore` **30**；三者并集 **33**，nightly 完整表达式并集 **47**。marker 存在重叠，数量不能相加宣称新增测试。全量 CPU 回归包含这些用例；未远程触发 GitHub Actions，不声称 Linux/远程 CI 或发布已通过。nightly 配置新增 JUnit 与性能 JSON 的工件上传，以保存实测数据。

完整日志与 JUnit：`.tmp/audit-next-stage-full.log`、`.tmp/audit-next-stage-full.xml`。本轮在工作区 `.venv` 加载源码运行，不替换此前独立安装/发布归档，也不把旧 wheel 验收结果当成此次源码的新发布证据。现有 MyPy 的未标注函数体检查范围限制继续保留。

复现命令（临时目录应换成全新路径）：

```powershell
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp .tmp/pytest-audit-next-stage-full --junitxml .tmp/audit-next-stage-full.xml -o junit_family=xunit1 --tb=short
.venv/Scripts/python.exe -m ruff check packages ci qa tests
.venv/Scripts/python.exe -m mypy packages qa
.venv/Scripts/python.exe -m pytest --collect-only -q tests -m 'real_golden or short_wfo or dependency_resolver or artifact_restore or perf'
.venv/Scripts/python.exe .tmp/audit_next_stage_diagnostics_20261013.py
```

N02–N05 尚未全部关闭：需要确认新的独立窗口并冻结研究配方、补齐外部历史经济事实、实施协调器/learner 的 OS 边界，并在实际 CUDA 硬件和长窗口上验收。N04 的完整启动 Windows Job CPU worker 父进程硬退出恢复已在后续一轮实现，详见文首最新续开发链接；旧/未完成启动及 GPU lease 仍保留人工边界。这些轮次完成诊断与工程改进，不宣称策略收益或项目整体验收通过。

# Nautilus-RLES 项目深度审计报告

**审计日期：** 2026-10-13\
**范围：** L1–L6 契约、数据、原生执行与 Gym、独立 RL、独立 ES/Hybrid、运行资源、研究流程、测试与 CI。\
**审计方式：** 检查源码、配置、已有真实行情研究报告和门禁定义；运行当前 CPU 全量测试及静态检查。\
**原则：** RL 与 ES 是独立模块；NautilusTrader 上游源码保持原样。

## 1. 总体结论

Nautilus-RLES 已具备可运行的工程主链：真实 ETHUSDT 行情可经过数据质量检查、Gym、独立 RL / ES、Hybrid 训练、模型保存与加载、NautilusTrader 原生回测，最终生成账户及风险报告。A0–A4 路线、四种 RL 算法及多个检查点恢复路径均已有工程证据。

项目尚未达到正式策略与经济模型验收目标。最新完整 A0/SAC WFO 的一次性 Holdout 净收益为 **-1.565%**，收益门失败；A1/ES 和 A4/Hybrid 的新窗口 Test 收益也为负。历史合约规则、账户费率、保证金与清算映射仍不完整，现有运行必须标记为 **scenario-only**。多折泛化、长时资源边界和必要的 GPU 验收仍待完成。

本轮审计复验结果：**374 passed、4 skipped、45 warnings，260.88 秒**。4 个跳过用例均要求 CUDA，而当前环境安装 CPU 版 PyTorch（`torch 2.14.1+cpu`，`cuda_available=False`）。Ruff 全目录通过；MyPy 对 `packages qa` 的 104 个源码文件检查通过。

## 2. L1–L6 进度

| 层 | 已具备能力 | 主要未完成事项 | 判断 |
|---|---|---|---|
| L1 契约与运行时 | DTO/schema、配置、跨包契约、provider registry、序列化及阶段工件接口 | 统一发布身份与最终发布候选尚未完成 | 工程基础可用 |
| L2 数据 | Binance 行情导入、数据质量检查、特征/Scaler、Train/Validation/Test/Holdout 窗口边界 | 有效日期完整的交易规则、账户经济参数及历史规则覆盖 | 行情可用，经济事实不完整 |
| L3 Nautilus 与 Gym | 原生账户/订单/费用/资金费报告、风险观察、订单执行与终态清仓 | 真实历史保证金、排队成交和完整执行模型未证明 | 场景回放链路可用 |
| L4 RL | PPO、SAC、TD3、DDPG 的训练、存取、推理和部分阶段恢复；真实行情训练路线 | 收益准入、多折 WFO、最新改动后的 GPU 路径 | 算法可运行，策略未验收 |
| L5 ES / Hybrid | 独立 EvoTorch provider、A1 ES、A4 Hybrid 及阶段间迁移 | ES 代内恢复、ES GPU 路径及独立策略质量 | 工程链路可运行，策略未验收 |
| L6 计算与研究 | spawn worker、资源 lease、研究账本、WFO/Test/Holdout 与多种恢复路径 | 协调器/learner OS 级资源边界、父进程硬退出后的租约恢复、长窗口吞吐 | 有可用框架，长运行验收未闭合 |

## 3. 优先问题

### P0：策略收益与泛化不足

- 最新 A0/SAC WFO 在 Validation 选择后，Test 净收益 **+0.214%**；Test bootstrap 区间跨零，单折 PBO 为 null。最终一次性 Holdout 净收益 **-1.565%**、最大回撤 1.591%，最终收益门 `FAIL`。Holdout 已消费，不得重新打开用于调参。
- A1/ES 新窗口 Test 净收益 **-0.01913%**；A4/Hybrid Test 净收益 **-0.004999%**。两项均是短预算工程路线验证，不证明策略质量。
- A4 Test 平均请求仓位绝对值约 0.0963，而实际执行约 0.00779；执行非零率约 10.77%。应在未消费 Train 数据中拆解策略输出、软风险、硬风控、死区与订单执行的影响，再预先冻结配方开展新窗口和多折验证。
- 真实行情上有完整预算路线运行，但部分模型决策长期处于 deadband、无交易或极低参与度。训练完成和回测成功不能视为策略通过。

证据：`docs/AUDIT_L1_L6_20261013_CURRENT.md` 中“新窗口完整 A0 SAC WFO”和“A1 独立 ES 与 A4 Hybrid”章节。

### P0：历史经济规则与交易事实不足

- 当前行情源为真实 Binance 历史档案，但完整回测期仍缺有日期依据的 ETHUSDT tick size、quantity step、minimum quantity、账户实际 Maker/Taker 费率、完整保证金阶梯和清算行为依据。
- 已采集的 bookTicker 按秒抽取最优买卖报价，不能证明盘口队列优先级、未采样时刻的成交、真实延迟或历史账户费率。
- 场景费率、滑点和保证金配置不能表述为历史交易事实。严格历史经济门禁应继续失败关闭，直到获得可核验的外部依据。

证据：`data/binance_ethusdt_20240101/HISTORICAL_RULES_BLOCKER.md`。

### P1：长窗口资源和恢复边界

- Windows ES worker 已有 Job Object 进程树内存限制与强杀清理证据；这项能力不覆盖协调器和 learner。
- 多折 WFO 运行耗时仍高，90 天规模的峰值 RSS/commit、CPU 吞吐和磁盘工件增长未形成稳定验收数据。
- 受控 KeyboardInterrupt 和检查点恢复已有证据，但操作系统强杀、OOM、磁盘耗尽，以及父进程中止后 lease 的显式安全恢复流程仍需验证。
- CUDA ES/E1/E2 与 CUDA WFO 尚未闭环；当前审计环境没有 CUDA 设备，4 个 CUDA 测试因此跳过。

证据：`docs/DEVELOPMENT_GOAL_CURRENT.md` 的当前优先顺序和 Windows worker 限制章节；`packages/compute_runtime/src/rles_compute_runtime/resources.py`。

### P1：nightly CI 的 perf 标记没有用例

`.github/workflows/l6-ci-reusable.yml` 要求 nightly 收集 `real_golden` 与 `perf`。当前 `real_golden` 收集 13 项，`perf` 收集结果为 **0 项**，pytest 收集失败。因此 nightly 会在要求性能用例的步骤失败。与此同时，`short_wfo` 与 `artifact_restore` 各只有 1 项被标记，WFO/恢复的大部分测试被归入 unit，nightly 实际覆盖有限。

建议补充一个固定预算、稳定测量关键阶段耗时或峰值内存的性能用例，并将关键恢复用例标到对应 CI lane；不需要扩大哈希或发布指纹门禁。

### P2：审计报告和结论口径维护

- `docs/AUDIT_L1_L6_20261013_CURRENT.md` 包含多轮历史快照及多个测试数量，阅读时应以各章节日期和本报告本轮复验为准。
- Holdout 诊断正文漏写请求目标均值的负号：原始报告是 `requested_mean=-0.428`，绝对均值为 `0.428`；“偏空”的解释与原始数据一致，但符号应更正。
- 部分 MyPy 提示某些函数未标注类型，因此默认不检查其函数体；目前 MyPy 通过不等于所有动态路径都已严格类型验证。

## 4. 模块边界和上游状态

- RL provider 位于 `packages/rl_sb3`，ES provider 位于 `packages/es_evotorch`，通过 provider entry point 接入。
- Nautilus 调用封装在 `packages/nautilus_adapter` 等集成边界内；本轮未发现 RL 或 ES 通过改写 Nautilus 上游实现完成功能的证据。
- 本次检查的 Nautilus 上游 Git worktree 没有 tracked diff。唯一状态项是预先存在的未跟踪 `python/nautilus_trader/nautilus_pyo3.pdb`，本轮未触碰。

## 5. 建议开发次序

1. 使用未消费的 Train 数据分析 A0 Holdout 方向失效、A4 动作缩减和低交易参与度；冻结因果假设、预算和收益门后再跑新的独立窗口。
2. 在足够预算的新窗口上验证独立 RL、独立 ES 与 Hybrid，并扩展多折 Test/选择稳定性；维持现有收益门，不用已消费窗口调参。
3. 并行整理有日期依据的历史合约规则、账户费率、保证金/清算和执行事实；缺少外部依据时继续标记 scenario-only。
4. 完成协调器/learner 资源边界、OS 强杀恢复和 lease 恢复方案；再按所需硬件验证 CUDA 与长窗口吞吐。
5. 补齐一个有实际测量价值的 `perf` 用例，并修正 CI marker 覆盖，使 nightly 能运行真实的性能和恢复检查。

## 6. 本轮复验记录

```text
全量测试：374 passed, 4 skipped, 45 warnings in 260.88s
Ruff：All checks passed
MyPy：Success, no issues found in 104 source files
CUDA：torch 2.14.1+cpu；cuda_available=False；4 个 CUDA 用例跳过
perf marker：0 tests collected，pytest 收集失败
```

本报告记录当前工程状态，不构成收益承诺、真实交易经济模型认证或外部签核。

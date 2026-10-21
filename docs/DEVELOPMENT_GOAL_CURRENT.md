# Codex 连续开发目标

本文件是可直接用于 Codex Goal 的当前执行目标。审计问题、依赖、状态和验收条件以[任务索引](DEVELOPMENT_GOALS_REVIEW_20261017.md)为准；真实结果及最近续做位置以[开发执行记录](DEVELOPMENT_PLAN_20261017.md)末尾为准。每轮开始重新核实现状，不把历史计划或审计基线当作当前代码状态。

**执行方式：** Codex 每轮自动选择一个或多个当前可实施的审计整改，直接修改源码并完成必要回归；验收后接着处理下一项，不等待逐项确认。只有当所有可实施工程项均已完成，或剩余项确实需要尚未满足的外部条件时，才暂停，并记录每项的阻塞原因、恢复条件和准确续做位置。文档更新用于记录实施结果，不能替代源码实现。

## 可直接交给 Codex 的目标

```text
持续在 `D:\Program Files\NautilusTrader_RL_ES` 中落实审计报告 `C:\Users\QX\Downloads\NautilusTrader-RL-ES_源码深度审计与后续开发建议_20261017.md` 的未完成整改（A01–A14）。这是一个连续实施目标：每轮都要从当前续做位置检查并完成可做的源码/工程工作，不能把“检查完”“提出计划”或“更新文档”当作完成。审计基线 `c24d1bc4a47584f91a4c4db93bf77117d4e8341c` 仅供历史对照。

每次启动以及每个工作包验收后都继续执行以下循环，直到没有依赖已满足且可独立实施的工程任务：

1. 先检查 `git status`、相关 diff 和当前源码；阅读本文件、任务索引及执行记录末尾。保留已有未提交工作，不 reset/clean、不覆盖无关改动。对索引中已完成或已有证据的事项，只核验本次工作所需部分，不重复实现或重跑无关检查。
2. 按任务索引优先级和依赖，选择最高优先、当前仍有明确实施动作的任务；先确认缺口确实存在，避免重复已通过工作。把需求落实为具体代码入口、预期行为和验收测试。真实现金流、独立 Validation/Train 授权、历史深度、阶段工件或 Hosted CI 等外部依赖，只阻塞依赖它们的验收；记录恢复条件后立即转做其它可独立实施的工程任务。
3. 直接完成最小必要源码修改和有意义的确定性回归；运行受影响测试及适用 Ruff/MyPy，改动范围需要时再运行集成检查。记录实际命令和结果。失败时保存可复现错误，区分 `FAILED`、`BLOCKED`、`REJECTED`；不要把未执行命令、部分运行或本机结果记作通过。
4. 涉及行情或研究数据时，读取/解码前核对登记、窗口、用途和冻结身份。只在已授权范围工作；不得访问未授权 Validation/Test/Holdout，不得通过更换目录、协议或快照重用已消费样本。不得推断缺失现金流、伪造深度、降低风险/清算门槛，或把计划订单、Train-inner、scenario-only、合成盘口、奖励/Fitness 当作真实成交或独立盈利证据。
5. 每完成一个工作包，追加到 `docs/DEVELOPMENT_PLAN_20261017.md`：缺口、修改文件与行为、真实验证命令/结果、证据路径、数据用途、未认证边界和下一游标；按需更新 `docs/DEVELOPMENT_GOALS_REVIEW_20261017.md` 状态，并让本文件保持为短入口而非复制历史日志。完成一项后自动开始下一项，不逐项停下来询问。
6. 不修改 NautilusTrader 上游源码。只有任务索引中所有当前可实施的工程工作均已完成必要实现/验证，且其余任务的状态、阻塞依赖、恢复条件和准确续做位置均已记录后，才结束目标。策略盈利、独立经济认证或外部 CI 不是工程工作的替代物，也不能在证据缺失时宣称完成。
```

## 续做位置

最新执行记录还修正了 A09 的线性 NETTING 杠杆边界：2x 以上不再受 `maintenance_rate * leverage <= 1` 这一模型伪限制。固定杠杆通过 `StandardMarginModel` 的初始保证金 `dated_initial_rate / leverage` 实现，维持保证金直接使用 dated rule；配置只在有效初始保证金不低于维持保证金时接受。空仓、无订单的 session 重建、replay v3 和 checkpoint 恢复不变。3x 含实际原生持仓/维持保证金回归通过。外部交易所分档保证金经济和 Hedge Mode 仍未认证；A09 真实规则/经济认证保持 `PARTIAL`。外部依赖状态见[任务索引](DEVELOPMENT_GOALS_REVIEW_20261017.md)和[最新执行记录](DEVELOPMENT_PLAN_20261017.md)。

2026-10-21 完成费用/滑点预留后的杠杆限额修正，移除了 RouteB 包及接线，并修复 Windows staged-file 原子发布。严格字段和有符号手续费变更后的最新全量套件 **798 passed、5 CUDA-only skipped、53 warnings**；全量 Ruff 与 Windows MyPy（141 files）通过。11 wheels/11 sdists 已重建，wheel manifest `all_match=True`，11 个安装 wheel 及 entry point smoke 通过。负佣金返佣现由规则、transition 和经济报表按有符号资金流处理，杠杆成本预留仍只计正费用。杠杆仅在空仓、无订单后重建 session 才能修改。当前状态为本机工程验证通过，不等于交易所真实经济认证或跨平台 Hosted CI 放行。

返佣修复前的 allowlist 验收记录（历史运行）：Windows 原子发布使用同卷 rename，POSIX 使用 hard link；instrument_rule 严格字段 allowlist 拒绝当前静态模型无法消费的杠杆档位、notional caps 等字段。该变化后的定向测试 **146 passed**；随后全量测试 **795 passed、5 skipped**。最终 11 wheels/11 sdists、wheel manifest 与隔离 wheel/entry-point smoke 均通过。最新本地 Git 备份见本轮结束摘要。

instrument_rule 现执行严格字段 allowlist，避免 exchange leverage brackets、notional caps、maintenance tiers 被静态 margin model 静默忽略；字段回归、历史工厂与 adapter **146 passed**。变化后全量套件 **795 passed、5 skipped**；11 wheels/11 sdists 重建、manifest 匹配和隔离导入 smoke 通过。Binance 会按名义区间和生效时点变更杠杆/保证金档位，故 fail-closed 修复不等于真实档位模型认证。

外部门禁按执行记录保留：真实账户现金流来源与长窗对账、独立 Validation、兼容的 E1/R1/E2 阶段工件、真实历史深度、新授权独立市场窗口、干净交付 Hosted CI，以及 A09 真实标的规则、双向持仓和分档保证金经济认证。它们只限制各自依赖的验收，不得成为停止其它代码工作的理由。开始新一轮时必须复核这些状态是否变化。


2026-10-21 又修复了返佣事实被忽略的链路缺口：负 maker/taker fee 可通过规则校验；transition 保留负 commission；资金指标和线性权益对账按 signed fee 计算。买入与卖出返佣闭环回归通过。最新全量 **798 passed、5 skipped**，Ruff/MyPy 通过；11 wheels/11 sdists 构建、manifest `all_match=True`，隔离环境 11 包及 entry-point smoke 通过。source manifest 289 项，最终哈希记录于执行计划。真实交易所负费率是否出现在具体交易历史，仍需来源认证。


2026-10-21 保证金档位接口审计：当前静态 native model 不消费分档 margin；Binance USDⓈ-M `/fapi/v1/leverageBracket` 为签名 USER_DATA，返回 `initialLeverage/notionalCap/notionalFloor/maintMarginRatio/cum`，且仅在账户档位调整时出现 `notionalCoef`。Nautilus Python API 当前只暴露 Standard/Leveraged 两个固定比率模型，没有 Python 自定义 MarginModel。故不能把账号档位放进 instrument_rule 后继续按单一比率撮合；需 Rust/native margin model 支持及账户/时间绑定的已认证档位历史。当前 allowlist 拒绝这些字段。详见执行记录末尾。

2026-10-21 测试环境修正：发现 `.venv` 原来安装的是 `.artifacts` 中的旧 `rles_gym_bridge` wheel，且 `uv sync --all-packages` 会卸掉测试写读 Parquet 所需的可选 `pyarrow`。根 `dev` 依赖现显式加入 `rles-data[arrow]` 并更新 lock；离线 workspace sync 后从本地 `packages/data/src` 导入源码且保留 pyarrow 25.0.1。杠杆/资金费率定向测试 26 passed；sync 后全量 **798 passed、5 CUDA-only skipped、53 warnings**；Ruff 和 MyPy 141 files 通过。该修正让标准开发同步具备可复现的完整测试依赖。

随后修复 source manifest 将自身列为 dirty 状态的自引用问题，增加临时 Git 仓库端到端回归。当前生成器排除 manifest 自身的 status/diff，生成后可直接 `--verify`，并报告其余 tracked worktree 状态。最新全量 **799 passed、5 CUDA-only skipped、53 warnings**；全仓 Ruff 与 MyPy 141 files 通过。

资金费审计又发现 Binance REST 历史同时产生 `funding_settlement` 结算事实与 `last_settled_rate` 特征；若将后者的 `interval_minutes` 交给 Nautilus，会按边界推断第二次付款。已将 last-settled update 限定为特征数据，禁止其安排账户结算；新增延迟发布的重复结算/结算 mark 原生账本回归。定向相关套件 **236 passed**；全量 **801 passed、5 CUDA-only skipped、53 warnings**；Ruff 与 MyPy 141 files 通过。11 wheels/11 sdists 已重建，11 wheels 和 5 个 entry points 的隔离 smoke 通过，wheel manifest `all_match=True`。

2026-10-21 继续核对结算金额价格基准：Binance funding history 的逐笔 `markPrice` 原先被导入器丢弃，延迟的 native settlement 会依赖缓存中上一笔 mark。现已验证并保存为 `settlement_mark_price`，解码时先刷新对应 MarkPriceUpdate 再安排 settlement；EngineSession 支持并预检一个源事件解码出的多个 native events。解码器也只在所有结算字段通过校验后才修改内部 rate cache。回归覆盖结算时 mark 2,000 与此前缓存 1,000 不同、1 ETH × 0.1% 只扣 -2 USDT。最终版本的全量 **801 passed、5 skipped、53 warnings、449.24s**；定向 54 passed，Ruff 与 MyPy 105 files 通过；11 wheels/11 sdists 构建，11 个 wheel imports 与 5 个 entry points smoke 通过。续做仍受账户资金流水来源、账户专属保证金档位、Hedge Mode 账本、独立验证窗口及 Hosted CI 等外部门禁约束。

REST funding 导入再补了单行边界：只有一条结算时，接口数据不足以推断 interval，原逻辑会抛 `IndexError`。现明确失败为 `DataError`，不猜测周期。定向套件 **55 passed**，目标 Ruff 和 MyPy 通过；最终全量 **802 passed、5 skipped、53 warnings、412.59s**。最终构建 11 wheels/11 sdists，隔离 smoke 加载 11 包和 5 个入口通过；source/wheel manifests 刷新和本地备份已完成。真实资金流水、账户分档保证金、Hedge Mode、独立验证数据及 Hosted CI 门禁不变。

资金更新语义继续加固：数据 schema 和 native decoder 都拒绝未知的显式 `rate_semantics`，并校验 funding interval 与 future settlement timestamp；`last_settled_rate` 不能携带未来结算时间。现有未标语义且明确给了 future boundary 的预测更新保持兼容。定向 72 passed；最终全量 **804 passed、5 skipped、53 warnings、466.08s**；全仓 Ruff 和 MyPy 105 files 通过。最终 wheel 构建、manifest 和本地备份本轮收尾。

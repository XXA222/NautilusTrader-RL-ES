# Nautilus-RLES · AI-03 / L3 功能与接口说明

## 2026-10-13 当前集成增量

- `HardRiskGuard.apply()` 的内部结果增加 `force_reduce=False`。当前持仓已超硬上限，或权益停止/硬限制要求减仓时，该标记为真；即使策略请求本身已低于上限或为零，也不能由执行死区阻止强制减仓。
- `TargetQuantityPlanner.plan(..., force_reduce=False)` 验证目标严格减少现有同侧仓位，绕过 deadband；有挂单时先撤单、等待 ACK、重新取快照。提交原因固定为 `hard_risk_reduce`。最小数量和精度仍有效，最小金额例外只按 instrument rule 的显式 `reduce_only_min_notional_exempt` 授权；不可执行时抛 `PlanningBlocked`。零权益仅允许 mandatory reduction/terminal 路径。
- `NativePlanSubmitter` 对该原因的 SUBMIT 校验唯一持仓归属于当前 strategy，数量不得反向开仓或超过现有仓位，并使用 Nautilus 原生 `reduce_only=True`。终态仍要求精确全平。
- Gym info 增加 `risk_soft_target`、`risk_adjusted_target`、`execution_plan_action`、`execution_plan_reason`；应用报告增加 `risk_target_diagnostics`、`execution_plan_diagnostics`，区分风险缩放与 deadband 等执行原因。旧 episode 未提供风险中间目标时明确标记未知。

以下原始交付说明保留为历史基线；当前运行及验收边界以 `docs/DEVELOPMENT_GOAL_CURRENT.md` 为准。

**基线：** Nautilus-RLES v0.4.0 (2026-10-11)；L3 任务书 L3-001—L3-024。上传的 v0.3.1 仅作历史对照。\
**所有权：** `rles-nautilus-adapter`、`rles-execution`、`rles-risk`、`rles-gym-bridge` 四个包。\
**交付级别：** 待集成的生产模块候选；尚未获得 Q2/Q3 PASS，禁止据此启动正式市场 RL/ES 训练或宣称可实盘。

## 1. 四个包的职责与变更文件

| 包 | 生产代码 | 职责 |
|---|---|---|
| `rles-nautilus-adapter` | `session.py`, `market_data.py`, `bootstrap.py`, `__init__.py`, `pyproject.toml` | 官方 BacktestEngine 公开 API、L1 MarketEvent→原生 Bar/Mark/Index/Funding 解码、每折新建/释放引擎、完整时间戳流式批次、Native Account/Position/Order/Funding 事实提取、终止后的原生账本校验、engine provider |
| `rles-execution` | `spec.py`, `planner.py`, `__init__.py`, `pyproject.toml` | resolved Execution config、deadband、Decimal 数量精度、min_qty/min_notional、pending wait/fail/cancel_replace、决策幂等键、三个阶段延迟与双场景保守滑点配置 |
| `rles-risk` | `guards.py`, `__init__.py`, `pyproject.toml` | 不可进化硬风险配置、隔离的软风控参数、资金/市值/杠杆边界、缺少清算能力时的保守限额、风险只缩小 |
| `rles-gym-bridge` | `observation.py`, `reward.py`, `env.py`, `vector.py`, `__init__.py`, `pyproject.toml` | 有效时间+单位+固定顺序/哈希观测、仅本折训练拟合的只读 scaler、native equity 经济奖励、Gym 单环境、spawn 进程隔离同步 barrier |

无 NautilusTrader 复制文件、无外线源码、无 tests/、fixture、mock、golden、运行产物、训练结果和数据集。四个子包具有独立的 `pyproject.toml`；无主仓库级配置。

## 2. 公共接口与调用顺序

### Engine / Account

- `PublicBacktestBuilder(config: NativeBootConfiguration)` 通过官方 `BacktestEngine(BacktestEngineConfig(...))`、`add_venue()`、`add_instrument()`、`add_strategy()` 建立 **新的**原生引擎；要求原生 margin、fee、fill、latency model 显式传入，不接受默认的无成本环境。
- `EngineSession(build_native, submit_plan, decode_event, facts_reader, contract_snapshot_factory, certified_barrier_protocol)` 是可注入的 Native session。`open(episode)` 校验实际 wheel 必须为 `nautilus_trader==2.0.0rc6`，并锁定数据/折/引擎指纹；`bootstrap(...)` 以完整的 UTC ns 时间戳批次注入初始化数据；`advance(plan, batches, next_decision_ts_ns)` 提交订单意图，按引擎自然顺序推进到下一 barrier；`snapshot()` 仅提取 NT 账户/订单/持仓事实；`drain_and_close(...)` 只允许 `force_flatten_with_cost`，要求 pending orders 已确认清零，先引擎原生结算再检查 flat；`close()` 使用官方 `dispose()`。
- `NativeFactsReader(account_id, instrument_id, currency, liquidation_capability=False)` 使用 `engine.portfolio.build_snapshot()`、`portfolio.account()`、`engine.cache.positions_open()`、`orders_open()`、`price(..., MARK)` 获取 native 事实，带 stale/unpriced guard。对外只送 Decimal、dict、tuple 等中立值；不能把 PyO3 `Account/Position/PortfolioSnapshot` 装进 Contracts。
- `VisibleBatch(timestamp_ns, events, complete_timestamp=True)` 为适配层内部结构。L2 的原始 `MarketEvent` 只消费已可见事件；拒绝混时戳、未来可用时间、分割同 timestamp 的数据。首版要求流式批次所有事件在 `ts_init_ns` 时即可用；滞后数据映射见 ICR-L3-002。

**强制门禁：** 完成用户 HR-L3-1 公开 API 审批、L1/L2 冻结 DTO 及真实 wheel API 核验前，不得为 `certified_barrier_protocol` 发放签核。字符串 `nt2rc6_stream_complete_timestamp_v1` 只是接口协议标识，不代表本交付已获认证。

### Execution

- `ResolvedExecutionConfig.from_dict(config)`：严格字段集，`schema_version=0.4.0`，`deadband∈[0,1]`，`pending_order_policy∈{cancel_replace, wait, fail}`，首版仅支持 `MARKET/GTC/force_flatten_with_cost`，`base_latency/stress_latency`、`base_fill/stress_fill` 为不同版本化模型；`fingerprint()` 生成签名。
- `TargetQuantityPlanner(config).plan(decision_id, instrument, target_exposure, risk_budget_notional, equity, mark_price, current_qty, pending)`：目标风险预算金额 → 合约数量 → 按 `qty_step` **向零截断** → 死区/最小数量/最小金额检查；pending 采用 fail、wait 或发出 cancel 指令并要求 ACK 后重新 snapshot/replan；不会生成 “旧挂单 + 新目标同时成交” 的无限叠加场景。
- `PlanDecision` **仅为该包内部计算结果**。`to_contract_plan(decision, factory, latency_model_id=...)` 将其转化为 L1 **所有权**的正式 ExecutionPlan DTO；不应把 `PlanDecision` 当成跨包公开序列化契约。
- `LatencyProfile` 三段纳秒时延，只描述真实需要注册到 NT 的延迟模型；`FillProfile` 保守 bps metadata，不在 L3 自己修改原生成交或账户权益。费用、滑点和资金费应只由原生执行引擎反映一次。

### Risk

- `HardRiskConfig`（不可变 dataclass）字段 `max_notional`、`max_leverage`、`margin_buffer_fraction`、`min_equity`、`incomplete_data_max_leverage`、`require_mark`、`require_funding`。`signature()` 隔离硬限制配置。`HardRiskGuard.apply(...)` 最大敞口/资金可用性/安全缓冲/清算能力 downgrade；对于可能扩大风险的方向进行裁剪，跨零反转先归零；低于权益下限只允许退出。
- `SoftRiskParams` 是 **E2 唯一可搜索**的字段集：`target_scale`、`max_turnover_fraction`、`drawdown_throttle`、`funding_sensitivity`，每项都固定在 `[0,1]`。`SoftRiskTransform.apply(...)` 必须只缩小所请求风险；`from_search_dict` 拒绝所有未知键（包括任何硬风控字段）。
- 若 mark/funding/清算事实不可证明则相应 fail-closed 或严格降低允许杠杆；该限制不构成任何“强平安全性已经验证”的声明。

### Gym / Vec

- `ObservationSchema(schema_version, names, units)` 固定大小/顺序/单位/hash，并强制包含 `account.drawdown, account.peak_equity, account.equity, risk.guard_intervention, execution.prev_executed_target`，使奖励所需 DD 状态对策略可见。
- 历史生产工厂在基础账户/风险/市场观测外，还公开 `execution.pending_order_count`、按风险预算和原生 mark 归一化的 `pending_signed_exposure` 与 `pending_gross_exposure`、以及 `fill_rate`。策略可区分方向相反但总挂单量不同的未成交订单；错误 side/负数量会 fail closed。当前 production factory schema 为 23 列，新增字段属于 Gym 输入契约并参与 schema hash。
- `ObservationBuilder(...).build(decision_ts_ns, market, account, risk, execution, time)` 要求 L2 所有 market feature 有 `available_at_ns≤decision_ts_ns`，单位匹配，所有数值有限，且数组 `np.float32` 不发生溢出。若有 scaler，必须来自 L2、具已核验 hash、已冻结，**绝不 fit/partial_fit**。
- `RewardCalculator.reset(equity)` / `.calculate(native_equity, guard_intervention)`：`r_econ=log(native_equity/previous_native_equity)`；`r_shape=-λdd·max(ΔDD,0)-λguard·intervention`。没有单独 fee/funding/slippage 减项；非正 equity 采用显式破产惩罚。
- `NautilusBarrierEnv` 是 Gymnasium 单环境；episode factory 由集成人配置，必须实现 `bootstrap/observation_inputs/action_to_plan/advance/drain_and_close/close` 并持有单独 NT session。动作是连续 Box([-1,1], shape=(1,))；`action_to_plan` 的执行顺序只能是 soft → hard → execution。若 terminated 或 truncated，必须拿到原生平仓后证据，不能以平仓前权益结束评分。
- `reset` 的 info 返回初始 native equity；每次 `step` 的 info 返回未整形 `r_econ`、drawdown、requested/executed action 与独立 risk guard intervention。L5 ES 按这些 native episode facts 和冻结 FitnessSpec 计算 Train-inner fitness，不把 shaped Gym reward 当成经济收益。
- `SyncVectorBarrier` 使用 spawn 进程每环境独占 session；一次发 N 个 action，按相同 request ID 收齐 N 个响应才返回下一批。只要 worker 异常、超时或 ID 不匹配，立即终止所有 worker，并将本 attempt 标为失败；不得重用旧 action、静默复位或让单个环境偷跑。

## 3. 配置、依赖与事实边界

| 配置或依赖 | 约束 |
|---|---|
| Runtime | CPython **==3.14.7**；版本不符不视为通过 G0 |
| Native wheel | `nautilus_trader==2.0.0rc6`，**仅 adapter 包导入**；其他包禁止 import NT |
| Contract wheel | `rles-contracts` v0.4 主版的官方中立 DTO，由 L1 提供，本交付不构造替代模块 |
| Data / Feature | L2 的 DataTruth 可用时间、分折范围、dataset_hash/fold_hash、instrument rules、只读 scaler |
| Gym | Gymnasium >=1.2,<2；numpy >=2,<3；不擅自修改 SB3 算法包 |
| Execution | 应将 FillProfile 和 LatencyProfile 对应到官方 venue native model；不允许只配置 metadata 而在原生 venue 上默认为无成本 |
| Terminal | 评分窗口内/允许的结算边界下，要有足够市场事件使原生 flatten 成交；无后续成交条件时评分为 BLOCKED，而不是伪造 terminal 价格 |
| Parallel | 每个 env 固定独立 SeedTree 与 engine；`SyncVectorBarrier` 非 SB3 `VecEnv` 子类，SB3 兼容适配属于 ICR-L3-004 |

## 4. 公开 Nautilus API Inventory（HR-L3-1 候审）

| 原生公开 API | L3 使用位置 | 事项 |
|---|---|---|
| `nautilus_trader.backtest.BacktestEngine` / `.add_venue`, `.add_instrument`, `.add_strategy` | `bootstrap.py` | 新折装配、不复用 Engine |
| `nautilus_trader.config.BacktestEngineConfig` | `bootstrap.py` | `shutdown_on_error=True` |
| `BacktestEngine.add_data` / `.run(streaming=True)` / `.clear_data` / `.end` / `.dispose` | `session.py` | 一次完整时间戳一批；end 时原生 shutdown/cost 结算 |
| `BacktestEngine.portfolio.build_snapshot`, `.account` | `session.py` | snapshot stale/unpriced/基准币 equity |
| `BacktestEngine.cache.positions_open`, `.orders_open`, `.price` | `session.py` | 原生持仓、挂单和 MARK |
| 官方 `Account.events`, `.margins()`, `.to_dict()` / `Position.to_dict` / `Order.to_dict` | `session.py` | 原生历史账本事件和只读转换 |

仅使用公开 API，无 `_private`/monkey patch/vendor。`submit_plan` 的官方 Strategy 命令投递时机尚须 Nautilus 2.0.0rc6 真实 Golden 确认，参见 ICR-L3-003。API 资料：

- https://nautilustrader.io/docs/latest/concepts/backtesting/apis-and-runs/
- https://nautilustrader.io/docs/latest/concepts/backtesting/execution-flow/
- https://nautechsystems.github.io/nautilus_docs/python-api-latest/backtest.html
- https://nautechsystems.github.io/nautilus_docs/python-api-latest/portfolio.html
- https://nautechsystems.github.io/nautilus_docs/latest/concepts/portfolio/

## 5. ICR / 缺失的上游冻结契约（未经用户批准不得变更）

| ICR | 需要的正式接口或批准项 | 当前处理 |
|---|---|---|
| ICR-L3-001 | L1 `ExecutionPlan`、`TargetExposure`、`ExecutionSnapshot`、`RiskDecision`、`EngineAdvanceResult`、`TerminalEvidence`、`EpisodeSpec` 的精确字段、构造器和 protocol version；尤其是 `ExecutionPlan` 的 cancel/ACK/action/reason 字段 | L3 内部计算对象与 `factory` seam 保持待绑定，禁止伪造 L1 DTO |
| ICR-L3-002 | L2 `MarketEvent`→官方 native event 的严格映射；落后于 ts_init 的 `available_at` 如何在 NT 回放时排序；完整同时间戳序列； mark/funding update 与实际 settlement 区分；完整 instrument rules 及 scaler 的 frozen/hash 访问协议 | 已添加 `NativeMarketEventDecoder`，验证 MarketEvent 可用时间、质量、instrument 与 payload，并映射 1m OHLCV、mark/index、funding update/settlement；rc6 合成 funding golden 验证多空现金流和单次结算。滞后事件排序、真实来源/规则数据与交易所 golden 仍待验收 |
| ICR-L3-003 | L3 官方 NT Strategy 下单/撤单投递与 event callback 时序；pending ACK / partial fill 原生映射；terminal cancel/flatten plan 生效时钟；fee/funding/margin/liquidation Golden 和 HR-L3-1/2/3/4 用户批准 | Native 提交代理必须显式 DI，未签核前 Q2 BLOCKED |
| ICR-L3-004 | SB3 2.9.0 `VecEnv` 所需 `step_async/step_wait`、done 自动 reset、terminal_observation、seeds/options 映射；该适配器由 RL owner 侧持有，L3 不 import SB3 | L3 仅提供同步 N-env barrier 与 Gymnasium 单 env，不宣称通过 SB3 VecEnv parity |
| ICR-L3-005 | L1 风险能力等级（DQ-A…D）、HardRisk/SoftRisk hash 和签名绑定、端到端 risk_budget_notional 与执行预算一致性证明 | 无 L1 已冻结证明前，不能声称结构级防绕过/清算验证通过 |

## 6. 已知限制与集成阻断

本次输入未包含已冻结并已签核的 L1/L2 代码、合同 schema 和 G0 的 CPython3.14.7/Nautilus 2.0.0rc6 wheel 环境；也没有本次实际运行的真实市场 golden corpus。因此 **L3-005/006/007/008/016/019/020/021/022/023/024 的验收条件未成立**。尤其是 funding 一次结算、同时间戳原生 parity、仓位终止成交、真实 `check_env`、SB3 VecEnv 行为均不能由 Python 逻辑推断为 PASS。`EngineSession` 是待原生验证的执行候选，不是已冻结的 Q2 事实源。

进一步边界：

- `NativeFactsReader` 需要 NT 的 `Money.as_decimal`、PortfolioSnapshot 等绑定；必须实际核查 2.0.0rc6，不能假设其他 RC 保持语义。
- 回测 Bar 无订单簿队列和冲击成本的精确重建能力。保守滑点是原生 venue model **配置要求**，不是自造精确模型。
- 非法 min-notional 残余仓位、无可用终止市场数据、下单失败、pending 未取消等情况拒绝评分；不得通过改零成本或跳过最后 N 步放行。
- 用户指定不能在最终包携带 golden/scenario/test 结果，本交付因此**不包含**审查证据/日志/benchmark；Gate 记录应保存在 Chief Reviewer 管理的独立验收区。
- 尚需官方真实 wheel 与 L1/L2 进行接口签核、事件金标、terminal/funding/fee golden、single-VecEnv 同动作 parity、fault stress。不可在未确认前启动正式 RL/ES 训练。

## 7. 用户需要执行的集成顺序

先批准 HR-L3-1 并冻结 ICR-L3-001/002，再批准官方执行回放协议与 HR-L3-2/3，最后在独立验收区运行 Q2/Q3、HR-L3-4。由用户作为唯一集成人决定最终 merge/release/tag。本包仅为 L3 独立源码交付，不包含完整仓库或任何其他开发线的源码。

---

## 8. L3 第二轮深度修订（模块构建版本 `0.4.0.dev2`）

**范围不变。** 本修订仍受总设计 v0.4.0 与 AI-03 工作说明书约束；`0.4.0.dev2` 只标识四个 L3 wheel 的生产源码迭代，不提高 L1 中立契约版本、不代表 Q2/Q3 认证。

### 8.1 按文件列出的生产变更

| 文件 | 第二轮变更 | 有效性边界 |
|---|---|---|
| `execution/planner.py` | `InstrumentRules` 规范化 Decimal/校验 `min_qty % qty_step`；当前原生数量不符合历史精度时阻断；显式 `terminal=True` 只允许目标归零；带 ACK 的 `WAIT/CANCEL`；终止 dust 不允许绕过 native 最小订单约束 | 仍要求 L2 可靠历史规则、L1 ExecutionPlan 构造器和官方 native 下单语义 |
| `execution/spec.py` | 绝不允许 Base bar 零滑点；stress 延迟/滑点不得比 base 乐观；报价精度与滑点参数严格校验 | Fill/Latency 参数必须经 native venue model 显式装配并有真实证据 |
| `execution/stress.py` | 提供只读 `ExecutionScenario` / `resolve_scenario(config, mode)`；固定 base/stress profile IDs、总延迟、bps 和模型哈希 | 只表示配置选择，不自行产生虚假成交，也不代表官方 model 已被安装 |
| `execution/faults.py` | `FaultPolicy`、`ScenarioFaultGuard`：可见的价格跳变、缺数据窗显式输出 `CONTINUE/SKIP/HALT` 或抛 `MarketFault`；带持仓/挂单时绝不跳过数据空窗 | L2 必须提供缺失/价格可用时间事实；`HALT` 不是已平仓的收益终点 |
| `risk/guards.py` | 不可变硬/软风险值规范为 Decimal；允许原生负可用保证金时的减仓路径；`HardRiskGuard.apply(..., data_quality_grade)` 强制消费 `DQ-A/B/C/D`；`DQ-D` 禁止评分，`DQ-C` 强制能力降级 | 高杠杆清算精确性仍须原生金标；任何 DQ 事实由 L2/L1 权威提供 |
| `gym_bridge/observation.py` | 对 scaler 执行前后 `frozen` 和 fingerprint 断言；拒绝 shape 变化、负的可用时间、非法 `float32` 值 | 归一化统计只能由 L2 train split 提供 |
| `gym_bridge/reward.py` | 经济收益改为 `log(E_t)-log(E_{t-1})`，防止比率中间计算溢出；破产后禁止继续普通收益步骤 | 经济权益只能来自 Nautilus native 账户，不扣第二遍 fees/funding |
| `gym_bridge/env.py` | 显式校验 action/time/evidence；在原生终止处理后评分；破产也须 native terminal 可证明；输出审计字段并确保关闭；动态暴露 `rles_transition_hash`、`rles_boundary_token`、`rles_boundary_observation` | episode 必须实现 `transition_fingerprint_hash()` 与 `checkpoint_boundary_token()` 且来自 settled native barrier；fake-seam 测试不构成真实 Engine 证明 |
| `gym_bridge/vector.py` | Worker 消息关联 `(env_id, request_id, episode_id, step_id)`；超时/乱序整批中止；done 后不自动重用旧动作；协作式 close，异常强杀兜底 | 本对象不是 SB3 VecEnv 子类；SB3 自动重置和 terminal_observation 仍需接口审查 |
| `nautilus_adapter/session.py` | 在任何副作用发生前校验整个 barrier 和 native 解码时间；`NOOP/WAIT` 不提交原生订单；校验并保护动作幂等；读 `orders_open + orders_inflight`；terminal 可先 CANCEL 或 WAIT-ACK，再 native flatten/end/dispose；fold 稳定指纹含配置哈希；严格 CPython/NT 版本门禁 | 不得把 synthetic 内部检查当成 native Golden；需原生 2.0.0rc6 实测 |
| `nautilus_adapter/bootstrap.py` | 快照化 caller 传入的 venue 配置，防止外部变更已装配执行选项 | Native model/策略工厂对象本身仍须集成阶段冻结版本与归属 |
| 四个子包 `pyproject.toml` | 版本 `0.4.0.dev2`；固定 Python `==3.14.7`，NT `==2.0.0rc6`，L1 Contracts `>=0.4.0.dev0,<0.5.0` | 安装/锁文件/运行依赖与 build backend 必须在 G0 统一验证 |

### 8.2 更新后的重要公开接口

- `TargetQuantityPlanner.plan(..., pending: tuple[PendingOrder, ...], terminal: bool = False) -> PlanDecision`。`terminal=True` 目标只能是 0，绝不隐含 native min qty/notional 例外；terminal 时有 pending 首先必须 `CANCEL` 或 `WAIT`，随后新 native snapshot 才能生成真正的 `SUBMIT`。
- `HardRiskGuard.apply(..., liquidation_capability: bool, data_quality_grade: str)`。`data_quality_grade` 是本轮新增 **强制参数**，由 L2 提供 `DQ-A/B/C/D`；等级未冻结或 `DQ-D` 时不能评分。此变化要求合并人确认 L1/L2 调用链兼容性。
- `resolve_scenario(config, mode='base'|'stress') -> ExecutionScenario` 固定执行语义；集成人负责将 profile 身份、三个延迟段与 conservative fill model 映射到官方 venue，并在运行 manifest 记录其已应用证明。
- `ScenarioFaultGuard.outage(...) / price_shock(...)` **仅** 给出故障决策，不更改 Nautilus 账户或 L2 已可见事件；`SKIP` 只有在证明完全无持仓、无在途订单时才可能返回；不能用该返回值替代终止结算。
- `EngineSession.open(episode)` 除 `dataset_hash`、`fold_hash` 外新增需要 `config_hash`；只允许 `CPython==3.14.7` / `nautilus_trader==2.0.0rc6`。`EngineSession.advance` 前校验所有未来批次；不再把 `NOOP`、`WAIT` 投递为交易命令。
- `EngineSession.drain_and_close(..., cancel_plan=None, cancel_batches=None, cancel_ts_ns=None)`：native 有 pending 时，要求真实取消命令或已在等待 ACK 的计划、对应完整时间戳批次和 ACK 后 native 事实；若无法确认 flat，直接失败且**不得记分**。
- `SyncVectorBarrier` 明确为一个 **同步并行屏障**，而非满足 SB3 全部特殊语义的 `VecEnv`。自然 episode done 后不进行隐式单 Worker reset，仍须一个显式的新 cohort reset；单/向量 trace 与 SB3 adapter 的等价性必须独立验收。

### 8.3 仍然被前置审查 Gate 阻断的工作

- **HR-L3-1 / L3-001—004**：正式 API inventory 必须以固定 `nautilus_trader==2.0.0rc6` wheel 验证。当前最新在线文档与 rc6 可能存在漂移，不能视为其兼容性证明。`NativeFactsReader` 的 `build_snapshot`、`Position.adjustments/events`、`Account.events/margins`、`cache.orders_inflight` 与 Strategy submit/cancel 路径均须现场核验。
- **HR-L3-2 / L3-005—008**：native fee/funding/mark/margin/拒单/部分成交/在途 ACK 的固定原生金标由用户验收环境管理；L3 包中不得携带 Golden、fixture 或测试结果。
- **HR-L3-3 / L3-009—016**：已提供配置、数量规划、撤单、风险与终止控制的 L3 生产候选；仍需证明 `ResolvedExecutionConfig` 与官方 venue model 实际一一对应、终止费用真实入账、`DQ-C` 不产生清算安全误述。
- **HR-L3-4 / L3-017—024**：Observation/Reward、Env、同步 Worker 与故障策略具有独立实现；尚缺已冻结 L1 DTO、L2 train-only scaler、真实 `gymnasium.check_env`、SB3 VecEnv 行为及 single/vector/audit native trace parity。没有 Q2/Q3 PASS，不允许正式 RL/ES 市场训练。

### 8.4 接口变更请求（ICR）增量

- **ICR-L3-006（需 Chief Reviewer 批准）**：L1/L2 接口必须提供可靠的 `data_quality_grade`，并承认风险层 `DQ-D` 必须 halt、`DQ-C` 与缺失清算能力降级；L3 `HardRiskGuard.apply` 不设置任何默认等级。若现有冻结版 L1/L2 无此字段，应先完成书面 ICR，不能修改其他包实现或静默默认 `DQ-A`。
- **ICR-L3-007**：L1 `EpisodeSpec` / run identity 固定 `config_hash` 与 `dataset_hash/fold_hash`；两个具有不同 `execution/risk/data semantics` 的 run 不得复用同一指纹。
- **ICR-L3-008**：经审核的 Native event decoder 必须让 L2 `MarketEvent` 映射到 `native.ts_init` 与 `VisibleBatch.timestamp_ns` 完全一致，且提供同 timestamp 完整性和跨 split 事件唯一性证明。绝不重写事件时序以适配 Gym。
- **ICR-L3-009**：终止流程需解决 `TerminalEvidence.snapshot` 及其 `observation_inputs`：即使原生 Engine 被释放，评分终止那一步仍必须以缓存的 L2 已可见特征和已脱离原生对象的 neutral AccountFacts 构建 observation；不能从已关闭的原生引擎回读。
- 原 **ICR-L3-001—005** 仍然有效，本轮未获得审批或修改其他开发线源码。

### 8.5 工程限制

1. 使用官方 API 的候选适配器不等于经过真实 wheel 核验；不从 mock/synthetic 校验推导 native 成交、资金费或利润真实性。
2. `ExecutionScenario`、`FaultDecision`、`PlanDecision` 和 `NativeAccountFacts` 是 **L3 内部对象**，不是另建的 L1 合同替身；禁止跨包持久化为独立账本。
3. Terminal NOOP 仅可在确认证明原生无持仓、无订单时成立。原生 terminal score 必须以交易后实际账户权益为准；窗口末不允许隐藏平仓成本或静默清空仓位。
4. Q2/Q3 需要包含 pending-cancel ACK 与 submit 间隙、原生拒单、partial fill、funding 正负/边界、mark/last 分离、净持仓与 margin capability、不连续 Bar 的 stress 场景。
5. 本交付 **不包含** Nautilus 官方源码、其他开发线、配置/数据副本、tests/fixture/mock、测试脚本、运行结果、Goldens、日志或研究工件，用户保有唯一 merge/release 决定权。

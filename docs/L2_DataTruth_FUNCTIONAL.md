# Nautilus-RLES · AI-02 / L2 Data Truth 功能说明

**设计依据**：`Nautilus-RLES 独立 RL+ES 优化设计方案 v0.4.0（2026-10-11）`；`02_L2 Data Truth AI-02 工作说明书`。本源码修订号 **v0.4.1**；未改变系统设计基线。**状态：PARTIAL，不得标记 Q1 PASS，也不得据此启动真实市场 RL/ES 训练。**

## 1. 所有权与源码清单

所有代码均在单包 `rles_data/` 内；不包含 `rles-contracts`、NautilusTrader、Gym、RL、ES 或其他开发线实现。

| 文件 | 生产职责 |
|---|---|
| `rles_data/core.py` | MarketEvent 本地事实模型、严格可用时间、稳定哈希、FeatureSpec、DatasetCard、DQ Report、FoldManifest、CuratedStore |
| `rles_data/schemas.py` | 各交易所已映射事件的严格校验：1m、Mark、Index、Funding Update/Settlement、Instrument Rule |
| `rles_data/io.py` | RAW 追加式持久化、Parquet/Arrow IPC 导入、CURATED 原子提交与完整性校验、JSON 工件只创建写入 |
| `rles_data/features.py` | 5m/15m/1h/4h/8h/1d UTC 已闭合聚合、点时可见特征、FeatureRow、Train-only Scaler |
| `rles_data/folds.py` | 60/15/15、360/15/15 滚动 WFO、warmup/purge/embargo、每折 scaler 哈希绑定 |
| `rles_data/quality.py` | GapAction：fail/skip/halt；规则生效时序与冲突拒绝 |
| `rles_data/__main__.py` | 模块自有 `validate` / `build-fold` 命令；JSON 诊断、非零错误码 |
| `rles_data/__init__.py` | 本包公开接口声明 |

## 2. 核心数据时间契约

- 以 **UTC int64 ns** 记录 `ts_event_ns`、`ts_init_ns`、`available_at_ns`。1m OHLCV 的 `ts_event_ns` 指**该分钟 K 线收盘时间**，而非开盘时间；每个 Feature 读取的 Bar 必须收盘且数据已到达决策屏障。
- `AvailabilityPolicy` 只允许 `native/derived/missing` 三态；`derived` 必须显式传入严格正数 `conservative_delay_ns`，计算为 `max(ts_event_ns, ts_init_ns) + delay`；未给出则保持 `missing`。原生时间早于事件/初始化时刻，直接拒绝。
- `Event` 的 payload 是深层不可变 JSON，带有事件 ID、类型、源 SHA256、provenance。事件按可见时间等稳定键排序；相同 ID 不同内容报错。同一 `instrument + close` 的不一致 1m K、同一结算的冲突资金费、同一生效时间的规则冲突均写入质量判断。
- `funding_update` 只包含该时刻已可用的预测/更新费率；`funding_settlement` 是单独事件，必须显式给出 `settlement_ns` 且与事件时刻一致；不允许使用未来结算数据替代当下已知费率。
- 规则必须包含 `instrument/effective_ns/rule_version/tick_size/step_size/min_qty/min_notional/maker_fee/taker_fee/initial_margin_rate/maintenance_margin_rate`；可选 `valid_until_ns`。已过期规则不能回退至更旧版，且冲突期间的规则不能用于风控或强平推断。

## 3. 原始数据与 CURATED

`store_raw()` 和 `store_raw_file()` 写入 `RAW/<source>/<kind>/<sha-prefix>/<sha256>.<parquet|arrow>`。先 staging + fsync，再 hard-link 独占发布，已有相同内容按 hash 校验复用；绝不覆盖冲突数据。`ingest_parquet()` 使用分批 Arrow 数据读取；`ingest_arrow()` 读取 Arrow IPC **file** 格式，两者使用同一严格字段验证，不静默补任何资金费/mark/规则。

导入必填列：`event_id: string`、`ts_event_ns: int64`、`ts_init_ns: int64`、`payload_json: JSON string`；选填 `available_at_ns: int64 nullable`。`payload_json` 必须为 JSON object，禁止非有限数和重复 JSON 属性。不同事件类型的必填字段见 `schemas.py`。

`write_curated(path, store)` 分批生成 Parquet，并提交 `.COMPLETED.json`；后者包含语义 snapshot 哈希与物理文件 SHA256。`read_curated(path)` 只允许读取 marker 完整、物理哈希一致且可重算语义哈希一致的快照。中断留下的不完整结果不可复用；需由外部 artifact 管理流程人工鉴别和清除，而非自动覆盖。

## 4. Feature/Scaler/可见性接口

- `aggregate_closed(store, decision_ns, timeframe_minutes, instrument=...)`：只返回所有 1m 子 K 完整、按 UTC 边界收盘且已可用的高周期数据。不完整桶不会出现；无对应分钟则不能伪造聚合值。
- `make_feature_row(store, spec, decision_ns, instrument=...)`：输出 `FeatureRow(schema_hash, decision_ts_ns, available_at_ns, instrument, values)`，可复算稳定 `hash`。`available_at_ns <= decision_ts_ns` 为硬不变量。多标的快照必须显式传 `instrument`。
- 支持 OHLCV `return/close/volume_mean/volatility`，按规范使用名称前缀或 `FeatureField.recipe` 显式指定。事件特征支持 `latest`（Mark/Index/Funding）与 `basis`（Mark 与 Index 可见时刻点时计算）。非 K 线事件必须显式给出 `max_age_ns`；K 线默认必须取当前应已闭合的最新一根，任何缺口或过期默认拒绝。`dtype=float32` 执行确定性 IEEE-754 量化，溢出拒绝。
- `fit_train(spec, train_window, samples)` 仅接收 Train **score** window 内的非重复时间戳；context warmup、validation、test 不能参加拟合。统计量只读、无 update API；`Scaler.transform(vector, feature_hash)` 校验 feature schema/hash/shape/有限值。

## 5. Fold 与终止边界

- `build_folds` 允许 `60/15/15` 或 `360/15/15`，15 天步长。`purge = max(feature_lookback_ns, execution_lag_guard_ns)`；`embargo_ns` 独立显式传入；按独立 Train→Validation→Test 评分窗口记录。`data_start_ns` 建议取经验证 RAW 最早纳秒时刻，CLI **强制要求**该值。
- `Window` 的 `[warmup_start,score_start)` 只能读取上下文，`[score_start,score_end)` 才能拟合/计入奖励。所有 split 使用 `terminal_policy=force_flatten_with_cost`；`reset_positions/reset_orders/reset_replay/freeze_validation_scaler` 均须为真。
- **禁止全折共用一个 `scaler_hash`**；完成每折 Train 拟合后使用 `bind_fold_scaler(fold,scaler)` 校验此 Fold 的 Train Window 哈希，再绑定新的 FoldManifest。
- L2 仅保存 terminal/reset **数据契约**，不执行订单平仓，也不能代替 L3/Gym 环境验证实际状态隔离。

## 6. 质量与使用权限

`CuratedStore.quality(start_ns=..., end_ns=...)` 返回包含 DataQualityGrade、数据缺口（含明确给出首尾范围时的边缘缺口）、事件计数、重复数、冲突数、发现项及 snapshot 哈希的 `DataQualityReport`。默认评估完整 Mark/Index、Funding Update/Settlement、Rule 和 1m 数据种类。

质量审计以预期标的集合为严格边界：传入 `expected_instruments` 时，集合外任一类型的市场事实都会被拒绝；未传时，OHLCV 中出现的标的构成允许集合，集合外的 mark/index/funding/rule 同样拒绝。这样同一快照不会悄悄携带未声明标的的辅助事实。

| 分级 | 基础处理原则 |
|---|---|
| DQ-A | 关键事件类别与时间字段初步完整、无检测到的冲突；仍需额外连续性/来源及规则覆盖专项验收，不能仅凭标签认定可安全强平 |
| DQ-B | 仅有经声明的保守推导时间；时间敏感型保证金/结算研究需显式敏感性审批 |
| DQ-C | 缺 Mark/Index/规则；只允许低杠杆方向研究，禁止强平安全结论 |
| DQ-D | 关键 1m 缺口、不可用时间、Funding 不可信、经济事实冲突等；阻断训练/评估 |

`CuratedStore.require_usable(report,purpose)` 强制按研究用途限制；`decide_gap(report,purpose,runtime=...)` 返回明确的 `fail/skip/halt` 决策。原始交易所数据的 **跨日完整规则覆盖/Mark 连续频率/历史资金费预期结算表**仍需真实数据源专项核验；本版不能凭少量事件计数做出风险精确性承诺。

## 7. CLI

1. 内置命令（等待 `L1-012` 冻结后由总集成人对接最终 `rles data` 顶层入口）：`python -m rles_data validate --manifest-json <path>`；manifest JSON 对象只接受 `inputs`（非空数组，元素 `path/source_hash/kind/provenance`，可选 `conservative_delay_ns`）、`start_ns`、`end_ns`。`--card-output <path>` 可写不可覆盖的 DatasetCard+DQ JSON。单文件路径可用 `--parquet --source-hash --kind --provenance [--delay-ns]`，常因缺少其他类型数据而被 DQ-D 阻断。
2. `python -m rles_data build-fold --start-ns ... --end-ns ... --data-start-ns ... --train-days 60 --warmup-ns ... --feature-lookback-ns ... --execution-lag-guard-ns ... --embargo-ns ... --dataset-hash ... --feature-hash ... [--output <path>]`。不完整的窗口、错误哈希/参数和未知配置硬失败，stderr 为 JSON，退出码 2。不得通过 `--scaler-hash` 向所有 Fold 复用同一模型统计。

## 8. 版本变化与后续工作（Work Report）

上一版 L2 源码的重要修复包括：深层不可变数据；所有事件入口的同一严格字段验证；时间序列遗漏最新 1m 时 fail-closed；非 K 线来源的可见性与 TTL；多标的隔离；RAW staging；CURATED 完成标记和重建验证；Arrow IPC reader；分批 Parquet I/O；跨来源经济事实冲突；期满规则不回退；每折 scaler 归属校验；可靠 JSON CLI 诊断。

- **L2-001~007**：开发功能已实现；仍待 L1-005 / 数据源协议验收。
- **L2-008~013、015~017**：生产功能已实现/加固；跨源质量覆盖和端到端环境验证未完成。
- **L2-014**：防未来污染的内部性质验证已执行；最终交付包按工作说明书不包含测试脚本、fixture、日志或证据工件。
- **L2-018**：Q1 正式 evidence bundle 与用户审查签核未完成，状态 **BLOCKED BY PRECONDITIONS**；本包内不包含 Q1 bundle（禁交付测试证据）。
- **跨线 ICR**：`ICR-L2-001` 请求 L1 冻结 MarketEvent、Schema、Manifest、hash 规范；`ICR-L2-002` 请求确认 Funding settlement 唯一键/规则时间线交易所具体映射；`ICR-L2-003` 请求 L1-012 统一 CLI 路由/异常退出码；`ICR-L2-004` 请求总集成人冻结 Mark/Index 历史时效阈值和规则持续覆盖策略。ICR 均未签核，源码**没有**修改其他包。

## 9. 当前已知限制（必须经过外部 Gate）

1. 当前执行环境为 **Python 3.13**，未完成设计基线 CPython **3.14.7** / G0-A 真实兼容性核验。
2. 当前环境 **缺少 pyarrow**，因此 Parquet/Arrow 实际导入、CURATED 往返及大规模负载路径尚未在本轮执行成功。PyArrow 作为外部运行依赖，不复制到交付源码。
3. 未提供真实 ETHUSDT 永续 1m、Mark/Index、Funding 结算以及历史费率/保证金规则数据；无法宣称来源时序正确或 Q1 已通过。
4. `CuratedStore` 在内存中持有事件对象，按块 I/O 不能等同于完全 out-of-core；2 年实测 RAM 峰值仍需评估后确定 worker 并发配额。
5. 本包仍为 **L2 内部模型接口**，不能与尚未签核的 L1 Frozen Contracts 冒充相同协议；L1-005、L1-012 和 HR-L2-1..4 未签核。
6. 不实现 L3 真正账户结算、订单/挂单重置、策略引擎事件屏障或训练系统；这些必须由各自工作包实现并用集成 Gate 验证。

**交付范围审计原则**：仅生产源码 `rles_data/*.py` 与本文；不携带 NautilusTrader 源码、测试代码、测试结果、下载数据、checkpoint、任何其他工作线源代码。最终合并、Q1 签核、release/tag 均由用户决定。

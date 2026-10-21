# 历史经济规则验收阻塞（2026-10-16）

行情覆盖已扩展，不能再沿用仅至 2026-08 的旧描述。本次用于 Train 诊断的持久行情包有 4,403,866 个市场事件；Parquet 时间列元数据的范围为 2023-12-30 00:01 UTC 至 2026-10-13 00:00 UTC。其冻结模型的 Train 结束边界更早，诊断只在模型原 Train 内选窗。时间列起止范围不表示所有种类的事件完整覆盖，也不表示全部区间可作独立验证。

行情原文件、完成标记、情景规则及规则引用的说明文件，已保存于 `.artifacts/audit-development-20261016/data/`；校验清单见 `docs/audit_evidence/20261016/persistent_artifacts_v2.json`。

## 官方来源复核（2026-10-16）

本次只核查公开官方资料，没有访问账户 API、读取凭据或修改情景规则：

- Binance 于 2023-10-27 发布 ETHUSDT 永续合约最小名义价值调整公告，规定最迟于 2023-11-02 10:00 UTC 从 5 USDT 调至 20 USDT。这个带日期的事实早于本地行情档案起点 2023-12-30，因此情景文件中的 20 USDT 至少与该项已知的样本前调整相符；但公告没有证明该值在 2023-12-30 至 2026-10-13 期间一直有效。未发现进一步的 ETHUSDT 专项最小名义价值公告，不代表已穷尽历史公告。[2023-11-02 最小名义价值公告](https://www.binance.com/en/support/announcement/detail/e4384cba297a4bd2a154be644d5d76f9)
- Binance USDⓈ-M 的 `GET /fapi/v1/exchangeInfo` 展示交易规则和 symbol filters（含 tick/step/min quantity 等字段），其请求没有历史时间参数，响应描述的是当下规则。它可作当前规则快照入口，不能单独重建 2024–2026 每个决策时刻的规则。相同官方页面列出可按时间范围请求的历史 funding rate；但它与历史费率阶梯、合约过滤条件、保证金阶梯不是同一类证据。[USDⓈ-M Futures Market Data API](https://developers.binance.info/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)
- Binance 在 2024-07-24 发布、于 2024-07-25 06:30 UTC 生效的 USDⓈ-M tick size 公告，列出该次调整涉及的合约及变更前后精度；公告所列合约不含 ETHUSDT。它只覆盖该次公告列明的特定变化，不构成 ETHUSDT 全历史过滤器档案。[2024-07-25 Tick Size 公告](https://www.binance.com/en-BH/support/announcement/detail/08f043a4f20d4f0290817ece13b3d7f4)
- `GET /fapi/v1/commissionRate` 是签名的 USER_DATA 接口，需要 API key，响应提供用户当前 symbol maker/taker 费率；接口没有历史查询参数。本项目没有账户授权或按日佣金单，因此不能核实账户过去实际费率及阶梯。[USDⓈ-M Futures Account API](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account)

来源 URL、访问日期和可支持的结论另记于 `audit_evidence/20261016/binance_economics_sources.json`。最小名义价值这一单项增加了一个样本前的官方历史事件，与场景值一致，但没有建立样本期内的连续有效性。资料也澄清了目前公开 API 的能力边界；仍未满足的事实包括：覆盖整个样本期的 ETHUSDT tick/step/min quantity/min notional 有效区间、账户实际 maker/taker 费率及阶梯、历史初始/维持保证金和清算映射、成交/延迟模型的可验证依据。没有用当前参数倒填历史，情景规则仍标为 scenario-only；归档和 SHA 完整性也不能把情景假设提升为历史交易所事实。

因此可以执行显式假设下的 Train 机制研究、工程验证和成本压力比较；最终历史经济验收仍未完成。Base/Stress 通过也只能说明在冻结场景下通过。旧 Test/Holdout 与已用开发窗口按项目登记处理，不能通过扩展行情日期、换快照或改输出目录变成新的独立数据。

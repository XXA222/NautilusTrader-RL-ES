# 长周期 RL 学习效果与经济有效性开发目标

## 总目标

持续改善 NautilusTrader_RL_ES 的强化学习样本利用率、策略表达、优化稳定性和费用后经济结果。研究覆盖连续动作空间与执行可达性、原生净权益奖励及其数值尺度、训练步数/完整 Episode/训练轮数、学习率和算法参数、因果观测特征与窗口长度、交易成本及 Funding 方向。最终目标是让策略在足量、多窗口、多随机种子的未消费样本上，经零成本、Base、Stress 对照后仍满足预先冻结的正收益门；不承诺盈利，也不把训练能运行、单折变好或回测偶然正收益作为完成。

## 研究顺序和验收

1. **训练覆盖与预算**：区分环境交互步数、每环境实际访问区间、完整 Episode 次数和优化 epochs；验证窗口是否重复、不同 Episode 起点是否增加独立市场状态覆盖。每次训练保存实际访问时间范围、回合数、部分回合和学习曲线。步数预算通过 Train 内收敛/稳定性诊断预注册，不能按固定倍数盲目放大。
2. **奖励与优化器数值**：保留权益变化的原生经济记账，分别比较未缩放 reward、固定数值缩放和单独预注册的轻度回撤 shaping。报告未缩放经济分项与实际送入 learner 的 reward，评估 PPO/SAC 的 value/advantage/梯度稳定性；奖励缩放不能改变 native account、费用、融资、净收益和风险门。
3. **动作和交易成本**：测量请求动作经过 soft risk、hard risk、deadband、最小数量和名义额门限后的可达比例、真实成交与成本。候选动作映射/持仓变化惩罚只在固定 Train 消融中逐项比较；hard risk 不放松，不以提高交易次数作为学习目标。每个候选同时对照零成本、Base 和 Stress，区分 alpha 缺失和成本侵蚀。
4. **输入和时间窗口**：以 legacy 为基线，逐组评估多周期趋势/波动、价量状态、Funding 收付方向与已知结算倒计时。保持因果可用时间戳、Train-only scaler 和明确预热/断档语义；任何 profile、窗口、参数变化都产生新模型身份。
5. **统计验证**：先在未消费 Train 内完成预注册的预算、消融和随机种子，再冻结至少三折时序 WFO、成本压力、基线和正收益准入门。Test 与 Holdout 只在 Validation 全部通过后按既定一次性查询策略使用。复用旧折只允许做标注清楚的机制诊断，不能作为新的独立确认。
6. **工程稳定和复现**：每个阶段执行针对性回归、Ruff、MyPy 和必要的安装后/真实引擎验证；保存冻结配置、代码身份、数据来源、随机种子、学习曲线、资源证据与失败产物。出现负收益、过拟合或不可行候选时保留结果并调整研究假设，不降低门槛。

## 本机资源调度

2026-10-14 实测本机为 Ryzen 9 8945HX（16 核/32 线程）、约 31.3 GiB 内存（当时可用 17.0 GiB），CUDA 可用的 RTX 5070 Laptop GPU（约 8 GiB 显存）。系统当前项目没有占用 GPU 训练，但 CPU 总负载约 33%，不能假定所有逻辑核始终空闲。

- CPU 优先承担 ES 候选评估和行情特征/回放；并发从 4 个 worker 的受控基准开始，测量吞吐、峰值提交内存和长尾时延后再逐步提高。不能直接开满 32 个重型原生 Engine。
- GPU 学习任务使用单独 owner lease；8 GiB 显存先限制为一个 learner，测得峰值显存及 OOM/恢复行为后再调整批量，不并发启动多个训练占用同一设备。
- 整机可用内存预留目标按用户 2026-10-15 指示调整为约 1 GiB；worker 继续受 Windows Job 提交内存限制，并在启动每个并行任务前检查物理可用内存、提交额度和显存。若内存接近 1 GiB、触发分页压力或显存不足，停止派发新 worker，并保留失败回执、回收其进程树。并行训练先以两个 learner 为试点，确认端到端吞吐有改善后再提高并行度；RTX 5070 Laptop GPU 总显存约 8 GiB，不默认超卖显存。
- A/B 候选可以并行计算，但共享数据/资源库的写入、研究查询账本、GPU lease、验证查询与 Holdout 必须串行并沿用现有身份/配额保护。源码并行改动须避免同文件冲突。

## 当前阶段与首项实现

当前合格的新研究数据不足：最新 2026-08-11 至 2026-10-11 行情没有适用的历史 instrument rule，DQ-C，且不够三折加 Holdout；已用或已消费窗口不能重命名为新验证。因此当前可以推进训练机制、合约、性能与资源工作，不对旧 Validation 重训选参，也不宣称任何策略收益改善。

首项实现是增加 `reward_scale`，默认 1.0 保持历史行为；非默认缩放只作用于 Gym 提供给 learner 的总 reward，分解中的原生 log-return 和 shaping 仍保留原单位，交易账户与回测净收益不变。配置进入环境/转移身份和 bundle，避免以错误尺度续跑或回放。下一步先跑回归并补齐 WFO/恢复身份检查，再建立 raw、1,000×、10,000× 的 Train-only 收敛与梯度诊断协议；没有新的独立数据前不据此选择经济策略。

2026-10-14 首项实现结果：新增 root `reward_scale` 配置（上限 1,000,000），`info.r_train` 标记缩放后的 learner reward，`r_econ` / `r_shape` 维持原单位；默认值不改变历史 application fingerprint、factory config hash 和 transition hash。bundle 保存尺度；Train、Validation、ES worker recipe、Train diagnostic、WFO Stress/Base/Test factory 都使用同一冻结值；非默认值与 bundle 不匹配时回放拒绝。定向集成回归 **116 passed**，后加的默认/非默认身份绑定测试所在文件组合 **42 passed**；全仓 CPU 回归 **417 passed、4 CUDA-only skipped、45 warnings**；独立 CUDA smoke **4 passed**；Ruff 通过，MyPy **107 个源码文件**通过。两次 CPU 回归相互重叠，不能相加宣称全仓通过数。以上验证工程行为，不是学习效果证据。尚未在新鲜合格行情上训练或比较 raw/1,000×/10,000×，也没有收益改善结论。

## 第二项实现：固定分段的并行 RL Episode

应用新增 `rl_num_envs`（1–4）和 `rl_episode_steps`（至少 2）两个独立训练参数。`steps` 继续表示所有环境合计的真实交互数；`rl_episode_steps` 表示每个原生账户 Episode 的决策步数；`rl_num_envs` 表示隔离的并行 Nautilus Engine 数。多环境必须显式声明 Episode 长度，不能悄悄复制完整窗口。每个 rank 在整个 Train 窗口内取得确定、均匀展开且互不相同的固定起点；窗口不足以提供不重复起点时启动失败。训练 scaler 仍拟合完整 Train，Validation/Test/Holdout 不采用训练分段。

并行 Engine 在 Windows spawn 子进程内从冻结的 environment recipe 和 scaler 重建，避开原生进程级注册表冲突。rank、环境数、Episode 长度和起点写入 transition/fold 身份与 `training_coverage`；默认单环境完整窗口继续使用旧身份。报告现在区分完整 Train 决策数、Episode 长度、各环境交互数、固定起点、每环境完整/部分 Episode 次数，以及所有固定片段并集能覆盖的 Train 决策比例。Gym reset info 同时给出实际 Episode 起止时刻与步数。

checkpoint 已实测覆盖两个隔离环境在各执行一步后的中间边界：两个 action journal 分别落盘，恢复时按 rank 重建同一子窗口、核对边界 token、继续完成剩余预算。该测试还暴露并修复了多环境下 SB3 algorithm seed 覆盖 L1 environment seed tree 的旧问题；环境 seed tree 现在保持冻结，后续 Episode 只按其确定性派生。WFO 在读数据、创建查询账本前预检并行采样配置。当前分段方式有意保持每个 rank 的起点固定，以维持 checkpoint 可证明恢复；尚未引入每次 reset 随机起点。

2026-10-14 工程验收使用已消费的 2026-08 Train 窗口做机制回放：Train 共 7,197 个决策步，4 个隔离环境分别从偏移 0、2,313、4,627、6,941 开始，各执行 256 步，合计 1,024 个互不重叠决策点，覆盖上界 14.23%。运行生成四份 episode replay、环境 seed tree、1,024 步检查点、完整性文件和最终强制平仓。该小预算 PPO 净收益为 -1.074%，费用 91.76、滑点成本 19.11，说明并行机制可运行，但不构成策略改善或参数选择证据。全仓 CPU 回归 **424 passed、4 CUDA-only skipped、45 warnings**，独立 CUDA smoke **4 passed**，Ruff 通过，MyPy **101 个源码文件**通过。

## 第三项实现：原生训练诊断证据

R1 每次 `learn()` 现在强制采集原生逐 transition 事实，包括未缩放 `r_econ`、`r_shape`、送入 learner 的 `r_train`、请求/执行动作、相邻目标变化、guard 干预、SUBMIT/NOOP 和完成 Episode；同时保存 SB3 optimizer 的有限标量，如学习率、更新数、解释方差、value/policy loss、entropy、KL、clip fraction 和 policy std。报告不以 TensorBoard 临时日志为凭据，而是在成功 R1 阶段写入规范化 `R1-learning.json` 及 SHA-256，诊断哈希同时进入 R1 stage payload 和最终 bundle；回测会验证文件、sidecar 与 bundle 三者一致。极短常数回报下数学上未定义的 explained variance 被显式列入 `optimizer_undefined_metrics`，其余非有限 optimizer 值继续 fail closed。

首份 4 环境、1,024 步机制诊断显示：训练期请求动作绝对值均值 0.6475，相邻目标变化绝对值均值 0.7728，97.07% transition 触发 SUBMIT；PPO policy std 仍为 0.9987，只有 2 次 update，解释方差为 -2.57，未缩放 reward 均值仅 -1.62e-5。训练后确定性回放虽有 +0.0344% 费用前毛收益，但费用与滑点合计约 1.109%，净收益 -1.074%。这把下一项优先级收敛为：先比较固定 reward scale 与更小的 PPO 初始探索方差/空仓初始动作，再研究 deadband、调仓频率和动作滞回；不能直接放大当前近满量程随机动作的训练步数。

数据审计同时确认，本地官方行情只到 2026-08-11，2026-07-17 至 07-31 还有约 14 天缺口，既有 2026-08 Train/Validation/Test 已消费。后续需先补官方档案并在读取结果前预注册切分；建议把最新约 15 天隔离为一次性 final Holdout、此前约 15 天作为 Validation。长周期档案仍缺 dated instrument rules，因此在补齐历史规则、margin tier 和实际 fee 证据前，经济结论必须标注为“真实市场记录 + 非历史规则情景”，不能升级为严格历史账户收益证明。

## 第四项实现：PPO 初始探索与空仓动作头

PPO hyperparameters 新增严格校验的 `log_std_init`（[-20, 2]）和 `zero_init_action_head`。两个字段只允许用于 PPO，进入 RL config hash、checkpoint 配方和恢复校验；未配置时继续采用 SB3 的 log std=0 和原动作头初始化，保持默认配方身份。空仓初始化会在模型创建后把 policy action head 的 weight/bias 置零，A4 的已证明 E1→R1 参数转移仍会随后覆盖这些初值。

Train-only 机制矩阵固定同一已消费 2026-08 数据、seed=31、4 环境、每环境 1,799 步片段和相同 PPO 优化器。8,192 步时，默认 std≈1 候选训练 SUBMIT 95.58%、确定性回放净收益 -1.385%；`log_std_init=-2` + 空仓初值把相邻训练目标变化从 0.778 降至约 0.150，但 1×/1,000× reward 的确定性策略均落在 deadband 内，回放为空仓、收益 0。1,000× reward 在首 20 updates 的 explained variance 约 0.75，高于未缩放候选约 0.50，但延长到 32,768 步/80 updates 后，std=-2 与 -1.5 候选仍为空仓，最后一段 explained variance 都降至约 -0.43。该结果说明低噪声成功阻止成本爆炸，但仅放大奖励和增加固定片段重复次数没有学出可交易优势；下一步必须让 Episode 确定性轮换覆盖 Train，并加入实际时间戳重复率，再继续长预算比较。

## 第五项实现：确定性轮换窗口与实际时间覆盖诊断

Train 决策时间轴被均匀切成 rank 独占 lane，每个 episode 只能在所属 lane 内取完整连续片段。每个环境的 episode 序号使用与随机种子相同的确定性 reset 调度，采用与合法起点数互质的步长遍历所有可能窗口，完成一轮后才重复；并行 rank 的采样时间段不会重叠。episode replay v2 增加 episode 序号和准确起始偏移，checkpoint 重建时以保存值为准校验调度，因此旧版 v1 replay 仍可读取，v2 中断恢复则绑定窗口和账户边界。环境数量、episode 长度、lane 规则和初始偏移进入 fold/transition 身份。

R1 每个 `learn()` 诊断新增决策时间戳首末值、唯一时间点数和重复比例，补上 coverage 上界与实际采样覆盖之间的证据缺口。`training_coverage` 现在报告 rank lane 长度、每环境访问步数以及实际唯一时间戳诊断位置。定向回归 **42 passed**；全仓 CPU/CUDA 可用环境回归 **433 passed**，Ruff 全仓通过，MyPy **101 个源码文件**通过。

机制长跑使用此前消费的 2026-07-31 至 2026-08-10 Train 行情，固定 PPO、1,000× reward、`log_std_init=-2`、空仓动作头、4 环境、每回合 256 步、总计 32,768 步；不运行 Validation/Test/Holdout。完成 128 个 episode、80 次 PPO 更新，覆盖上界 7,197 个 Train 决策步（100%）。四个 8,192 步诊断块内 unique timestamp 分别为 6,310、6,627、6,310、6,310；重复采样率 22.97%、19.10%、22.97%、22.97%。末块 explained variance 仅 0.052，policy std 由约 0.133 降至 0.124；请求动作绝对值均值 0.102、NOOP 25.32%、SUBMIT 74.68%、Guard 干预比例 100%。末块 `r_econ` 均值仍为负的 -5.51e-6。它证明确定性窗口轮换确实让长训练触及不同市场时点，并未证明学到正收益；更长的训练/更多重复本身不是下一优先项，应先拆解风险调整和下单门槛造成的交易可达性、奖励信号与净权益变化之间的差异。

该目标持续有效，完成一个工程阶段不等于目标整体完成。每次续开发先检查最新数据质量、使用台账、代码和硬件占用，再记录当期可验证的进展与下一项工作。

## 第六项实现：执行风险归因与新鲜区间基线探索

R1 训练诊断现在分别累计 SoftRisk 与 HardRisk 的动作调整均值/占比、请求动作到最终执行动作的平均绝对偏差，并按执行计划原因统计 deadband、风险减仓和普通 rebalance。诊断覆盖仍与每次 `learn()` transition 一一对应；环境若缺少任一归因字段就拒绝生成不完整报告。该项用于拆分策略本身、软风险变换、硬风险覆盖和执行门槛，避免把 100% guard 介入误当成策略自主学出的动作。全量回归为 **433 passed**；Ruff 全仓通过，MyPy **101 个源码文件**通过。

已为本轮预登记的新鲜 ETHUSDT USD-M 1m 数据导入 2026-08-12 至 2026-10-12。Binance Vision 日线 klines、mark/index klines 及 8/9 月 funding archive 使用 `.CHECKSUM` 校验；10 月当月 funding archive 尚未发布，原 API 访问受 HTTP 451 阻断，随后从官方 Binance funding-history connector 补录 36 条 10/1–10/12 funding 记录，保存响应 SHA-256 与采集查询信息。10/13 曾在预登记前经行情接口查询，明确从 Train、Validation、Test、Holdout 全部排除。市场区间含 89,280 根 klines、89,280 根 mark、89,280 根 index 和 186 个 funding settlement/update 事件；合并一日 8/11 的因果 warm-up 后为 272,532 个事件，重复/冲突均为 0。warm-up 只供窗口开始处的历史特征，不进入 scaler 拟合、reward、采样、选型或收益统计。行情检查验证了完整分钟连续性及 mark/index/funding 时间覆盖。

使用长周期目标中登记的三折 inner walk-forward：Train 8/12–8/25、8/12–9/1、8/12–9/8；对应 inner 验证分别为 8/26–9/1、9/2–9/8、9/9–9/14。外层 9/15–9/28 Validation、9/29–10/5 Test、10/6–10/12 Holdout 尚未查询。市场档案未附历史交易规则、账户 fee tier 和完整维护保证金阶梯；当前研究的 0.01 tick、0.001 step/minimum quantity、20 USDT minimum notional、maker 2bp/taker 5bp、初始保证金 100% 和 maintenance 0.4% 都显式标注为 scenario 假设，任何净收益都不能称为历史账户实盘可复制收益。

短 train-only 可复现性探针（inner-1，2,048 步）成功生成正式 R1 诊断和 bundle。之后预登记并行训练同一个 PPO 基线，seed 31/47/73 各 32,768 步、4 个 rank lane、episode 256、`n_steps=batch_size=512`、5 epoch、learning rate 3e-4、gamma 0.99、GAE 0.95、reward scale 1,000、`log_std_init=-2`、空仓动作头、观察归一化。三个模型完成并冻结后才对 inner-1 验证执行一次 Base/Stress 回放。Base 净收益分别约 **-0.0022%、0%、+0.0323%**；Stress 分别约 **-0.0012%、0%、-0.0849%**。三者均终端平仓、没有观察到清算，但只有一个 seed 的 Base 为正，且它在 Stress 下转负、周转率升至 0.648；同时所有 seed 训练末段 `r_econ` 均值仍为负。基线因此被拒绝，不升级到外层验证。

之后分别训练并评估了两个单变量候选：`causal_mtf_v2` 增加 15/60 分钟趋势效率特征，reward scale 从 1,000 提高到 10,000。两组各 seed 的 inner-1 Base 与 Stress 净收益均为负，均被拒绝；前者三个 Base 约 -0.02%，后者 Base 约 -0.01%、-0.02% 和略低于零。更高版本的 `causal_mtf_v3` 要求源声明未来 Funding 日历，本轮没有该证据，故在进入训练前阻断；不会把历史 Funding 时间伪装成事前日历。

deadband 0.075 候选的 seed 31/47 为零交易、零净收益，seed 73 的 Base/Stress 约 +0.04%/+0.03%，但跨 seed 门槛失败。SoftRisk 单步 turnover 上限 0.10（基线 0.25）候选也被拒绝：seed31 Base/Stress 约 -0.05%、seed47 空仓、seed73 Base -0.02% 且 Stress -0.14%；Stress turnover 升至 1.28，说明单纯降低 soft cap 没有稳定限制最终风险调整后周转。随后预登记的 entropy coefficient 0.01 候选已完成：seed31/47 仍为空仓，seed73 Base 约 +0.03%、Stress 约 -0.08%，未通过跨种子与压力成本门。当前所有已完成试验仍未找到稳健正收益策略。

## 第七项：扩大历史区间以检验市场状态覆盖

为减少短样本与单一市场状态造成的策略坍缩，正在从 Binance Vision 按官方 `.CHECKSUM` 并行导入 ETHUSDT USD-M 的 2024-11-17 至 2026-08-11 分钟线、标记价格、指数价格和已结算 Funding。官方归档支持月度期货 K 线；新增 `--archive-interval monthly`，默认日包路径仍兼容。2024-11-17 至 11-30 的真实月包试导入覆盖 20,160 分钟，OHLCV/mark/index 均零缺口，Funding 有 42 次结算及对应更新，重复与冲突为零。导入拆为三个不重叠时间段；另将已有 2023-12-30 至 2024-11-16 已校验档案按时间边界去重，并与本轮 2026-08-12 至 2026-10-12 新鲜数据拼接。归并后需再次核验完整分钟连续性、各事实流时间覆盖、重复/冲突、原始来源校验回执与快照 SHA-256，失败的区段不进入训练。

扩展区间只用于 Train 和新的 inner walk-forward，训练起点拟定为 2024-01-01，并提供 2023-12-30 起的因果预热。仍只使用明确标记为假设的固定场景交易规则；不同年份真实 fee tier、维护保证金阶梯与逐年交易规则并未从档案中得到证实。下一轮正式训练和验证配置须在快照哈希、DQ 报告确认后、模型启动前单独预登记；所有种子冻结后先查询未消费的 2026-09-02–09-08 与 09-09–09-14 inner 窗口，先前反复使用的 08-26–09-01 窗口只可作受污染的机制诊断。外层 Validation/Test/Holdout 继续封存。

### 扩展快照审计与正式候选预登记（2026-10-14）

归档快照已完成 v2 拼接，文件位于本机 `.tmp/rl-expanded-market-stage-20261014/expanded-market-v2/market.parquet`，SHA-256 为 `b2bd88627446593ef3f0fe53de1cbb42156f4c164a63a097ec9e77bd7ae14eb2`。快照含 4,403,866 个事件：OHLCV、mark、index 各 1,465,920 个，funding settlement/update 各 3,053 个。有效分钟线从 2023-12-30 00:01 UTC 连续到 2026-10-13 00:00 UTC（最后一根 bar 的收盘时刻为 2026-10-12 23:59 UTC）；三类 bar 时间戳对齐，缺失、重复和冲突均为零。归档检查发现 2026-06-29 mark/index 整日缺行及 2024-08-12 10:03–10:04 UTC 两分钟 mark/index 缺行，已分别用官方日包与官方 USD-M Kline 接口补齐，保留原始响应哈希和来源清单；不以静默前向填充掩盖缺口。Binance 月归档路径及 `--archive-interval monthly` 由回归测试覆盖，默认日归档兼容。

2024-01-01 至 2026-09-02 的全跨度 Train 配置校验为 DQ-B，原因仅为保守可用性延迟，缺分钟、重复、冲突为零；1,403,998 条 Train 决策样本只用于训练 scaler。scenario 规则文件和假设另行哈希，不能解释为历史真实手续费/保证金。为比较延长周期 PPO，已登记 2 个未消费 inner 窗口（截止 2026-09-02 与 2026-09-09）、每折 3 个 seed（101/211/307），总步数每候选 262,144。固定门要求六个冻结模型在两个未消费内折的 Base 与 Stress 都净盈利、终端平仓、零清算并且指标有效；任何失败都会拒绝，不升级到外层验证。当前最新预登记文件为本机 `.tmp/rl-expanded-market-stage-20261014/long-ppo-preregistration-v3.json`，其中保留 supersedes 哈希，未消费 Validation/Test/Holdout 仍为零查询。

全跨度小样本探针发现历史环境构造成本高且重复初始化会积累原生交易引擎内存。一次 8,192 行校验样本约 15 分钟仍未到 learner；收缩到 8×256 步后，RSS 约 12 GiB、系统剩余内存低于预留阈值，故在 learner 更新前按资源保护中止。两个中止均未产生候选模型或 Validation 查询，也不计入收益证据。随后定位到 `make_env` 每次重建时会转换整个 440 万事件快照，且为全部 140 万训练分钟生成一次完整批次分组；已新增按 `available_at_ns` 二分切片的 `available_event_window`，并让环境只组装 Episode 起点因果预热区与窗口内、本 instrument 的 market batches，按 Episode 帧而非全 Train 帧分组。候选固定 parity 样本采用单个确定性 256 步 Train 窗口，窗口起点、跨度和观测字节 SHA-256 写入 bundle。synthetic 回归验证了样本哈希稳定；定向回归 52 项与 Ruff/py_compile 通过。受 Windows sandbox 禁止 `os.link` 的影响，依赖临时 CuratedStore 落盘的 `test_historical_factory.py` 不能在当前运行权限下完成，尚待真实 Parquet fixture 的原生引擎 golden 验证。2,048 步 Train-only smoke 已按新事件切片代码重新启动。

## 第八项：全跨度 PPO campaign 结果与动作可达性诊断（2026-10-14）

v3 预登记的六个 Train-only 模型已全部冻结（每模型 262,144 步，seed 101/211/307，PPO A0、legacy 5 特征、reward scale 1,000、deadband 0.05）。随后只读取 inner Validation：inner-2 为 2026-09-02 至 09-09，inner-3 为 09-09 至 09-15；Base 与 Stress 共 12 个报告全部完成，campaign `status=rejected`、严格准入 `false`。外层 Validation/Test/Holdout 未查询。正式结果与完整逐笔报告保存在本机 `.tmp/rl-expanded-market-stage-20261014/long-ppo-campaign-state.json` 和 `long-ppo-validation/`；源快照 SHA-256 为 `b2bd88627446593ef3f0fe53de1cbb42156f4c164a63a097ec9e77bd7ae14eb2`，预登记 SHA-256 为 `12097e8536a8278444d35f2028ac768f5452cc621b416b382a5c61fe70d8bd4e`。

| Fold / seed | Base 净收益 | Base Sharpe | Stress 净收益 | Stress Sharpe | 判定 |
|---|---:|---:|---:|---:|---|
| inner-2 / 101 | +0.000003917 | 0.57 | +0.000003622 | 0.47 | 仅微小正值 |
| inner-2 / 211 | +0.000170169 | 1.97 | +0.000161426 | 1.91 | 单折为正 |
| inner-2 / 307 | +0.000345437 | 5.87 | +0.000237199 | 3.90 | 单折为正、Stress 下降 |
| inner-3 / 101 | -0.000338496 | -5.57 | -0.000284796 | -4.43 | 负值 |
| inner-3 / 211 | -0.000629162 | -4.46 | -0.001273811 | -8.38 | Stress 加剧亏损 |
| inner-3 / 307 | -0.000304128 | -14.15 | -0.000313176 | -14.55 | 负值 |

六个 seed/fold 组合在两个 fold 的方向完全反转：inner-2 全部为正，inner-3 全部为负；故没有跨折可接受候选，禁止读取外层区间。收益数量级也很小，正值不能解释为可实盘收益。scenario 规则、费率与维护保证金仍是假设。

实际回测动作诊断揭示执行参与度不稳，而非单一的 Risk guard 缺陷。inner-3 Base 每 8,638 个动作中，seed101/211/307 分别只有 21、87、2 个 SUBMIT；seed307 的请求动作绝对均值 0.01、P95 0.02，远小于 0.05 deadband，几乎完全空仓。seed211 请求绝对均值 0.17、P95 0.29，却有 8550 次 NOOP，说明持仓保持/增量目标与 planner deadband 之间也会造成动作可达差异。对照 inner-2 Base 的 2/10/34 次 SUBMIT，随机种子和相邻周已经足以把模型从微量交易切换到较高参与度或近空仓。下一轮须同时记录请求/soft/hard/实际敞口、deadband NOOP、成交数、换手、费用和滑点，不能只看训练回报或正的单折净收益。

训练日志中 `guard_intervention_fraction=1.0` 与 `soft_risk_intervention_fraction=1.0` 是“任意非零差异”的精确计数；它们本身不等于 100% 动作被实质阻断。Stress 报告的 seed307 请求到 SoftRisk 平均差仅约 `1.73e-6`，说明要结合差值幅度、risk-to-executed gap 与 planner reason 解读。更有诊断力的信号是 PPO value explained variance 跨区块多数接近零或为负、请求动作方向容易翻转、实际 SUBMIT 极少或随 seed 大幅波动。之前据 100% 比例推断风险层普遍过度改写动作的表述应以此更精确分析替代；HardRisk 绝不能为提升成交而放宽。

### 下一项预注册探索

保持动作映射、deadband、奖励、观测、episode 长度、训练步数和优化器其余参数不变，只比较 PPO 初始对数标准差 `log_std_init=-2.0` 与 `-1.0`。假设：更大的初始探索方差会减少确定性评估时的“近零请求/死区空仓”退化，并提高不同市场状态/种子间的可重复参与；若只提升换手和成本而没有跨期成本后正收益，则拒绝。只使用此前未消费的 2024-03-01 至 03-08 与 2024-04-01 至 04-08 内部窗口，各自训练截止于验证起点，训练起始为 2024-01-01；这些窗口不与 2024-04-15 后已用 WFO 重叠，也避开 2024-01-03 单日 Test。每折两种方差、三个共同 seed，12 模型全部冻结后才查询 24 个 Base/Stress 报告。失败时保留证据；不因此开放 2026-09-15 起外层 Validation、Test 或 Holdout。由于两组训练历史分别约 60 天与 90 天、scenario 规则非历史逐日规则，这一轮仍为研究筛查而非实盘可交易证明。

## 第九项：PPO 初始探索方差跨窗口完整验证（2026-10-15）

第八项预注册只改变 PPO 的 `log_std_init`（-2.0 对 -1.0）。12 个 Train-only 候选各训练 262,144 次交互，全部冻结后执行 24 个未消费的内部验证 Base/Stress 回放。窗口为 2024-03-01–03-08 与 2024-04-01–04-08，随机种子 101/211/307。预登记 SHA-256：`9dcb217fa4bd33cbecc080b1b87019d4b9282b0db412c14ffd288897de6d7406`；研究数据快照 SHA-256：`b2bd88627446593ef3f0fe53de1cbb42156f4c164a63a097ec9e77bd7ae14eb2`；逐项报告及摘要 CSV 在本机 `.tmp/rl-exploration-action-std-20261014/`。验证完成后 campaign 状态为 `rejected`，两臂严格门槛均未通过；外层 2026 Validation/Test/Holdout 查询数仍为 0。

| 设置 | 正收益 / 查询 | 严格通过 / 12 | 均值净收益 | 中位净收益 | 最差查询 | Base 与 Stress 同时为正的 seed-fold 对 / 6 | SUBMIT / 全部决策 | 平均换手比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| std-minus-2 | 6/12 | 6/12 | -0.006574% | -0.009594% | -0.368856% | 3/6 | 1,203 / 120,924 (0.99%) | 0.650 |
| std-minus-1 | 2/12 | 2/12 | -0.163165% | -0.099954% | -1.096398% | 1/6 | 3,739 / 120,924 (3.09%) | 2.128 |

| 窗口 / seed | std=-2 Base / Stress | std=-1 Base / Stress |
|---|---:|---:|
| q1-mar / 101 | +0.2649% / +0.2596% | -0.0517% / -1.0964% |
| q1-mar / 211 | -0.3283% / -0.3689% | -0.0670% / -0.0915% |
| q1-mar / 307 | +0.1672% / +0.1377% | +0.4116% / +0.2836% |
| q1-apr / 101 | +0.0127% / +0.0086% | -0.0775% / -0.1114% |
| q1-apr / 211 | -0.0422% / -0.1163% | -0.2911% / -0.6343% |
| q1-apr / 307 | -0.0278% / -0.0460% | -0.1084% / -0.1239% |

结果不支持提高初始探索方差：std=-2 的平均净收益损失较小（-0.0066% 对 -0.1632%），但仍只有 6/12 单项为正、仅 3/6 个 seed-fold 的 Base 与 Stress 同时为正；跨所有 seed、窗口和成本情景的严格准入未通过。std=-1 仅 2/12 单项为正，1/6 seed-fold 对双正。seed 211 在两个窗口、两种初始方差、Base/Stress 的 8 项里全部亏损；不能把 seed 307 的局部正值外推成稳定策略。

诊断显示探索方差主要放大交易活动，未带来稳健盈利：std=-1 总计 3,739 次 SUBMIT，std=-2 为 1,203 次（分别约占 120,924 次决策的 3.09% 与 0.99%）；std=-1 平均换手比 2.128，约为 std=-2 的 0.649 的 3.28 倍。交易增加伴随更差的平均净收益与更深的最差亏损。planner 的 `NOOP/deadband` 仍占绝大多数；PPO 产生的小幅目标变化并未转化为稳健的有效交易。不得放宽 HardRisk，也不能只为增加 SUBMIT 次数而调低执行成本门槛。

### 下一阶段开发目标

1. **比较 on-policy 与 off-policy 学习**：使用未消费且互不重叠的内部窗口 2024-03-08–03-15 与 2024-04-08–04-15，预注册 PPO（上一轮相对稳健的 `log_std_init=-2` 配方）与 SAC 两个算法臂，两个折、seed 101/211/307。固定每候选交互步数、观测、动作、环境、奖励、成本情景与硬风险规则；全部模型冻结后再读取 Base/Stress 验证。SAC 参数、采样/更新比和 checkpoint 完整性在训练前锁定。
2. **区分学习收敛与经济可交易性**：聚合 R1 训练分块的解释方差、critic loss、policy/entropy、动作标准差、奖励分项与时间戳重访率；将逐 seed 验证报告中的敞口变化、deadband NOOP、SUBMIT、成交、费用、滑点及净值并列。若 SAC 的 critic/策略不收敛，先定位学习机制；若学习信号改善而成本后收益仍为负，拒绝经济候选。
3. **硬性验收**：每臂须在两个窗口、三个共同 seed 的 Base 与 Stress 下全部净盈利，指标有限、终端平仓且零清算；同时须相对零交易和简单因果基准显示成本后改善。没有候选通过则保留拒绝状态并更新假设，不查询外层 2026 数据。任何正值仍是规则费率假设下的研究情景，不是历史账户利润或实盘保证。

下一轮在模型启动前单独预登记。2024-03-01–03-08 与 2024-04-01–04-08 已查询，永久视为已消费，不再用于选参。


## 第十项：PPO 与 SAC 算法对照（已完成，2026-10-15）

在开始训练前已冻结 12 个候选：两个互不重叠的内部窗口（2024-03-08–03-15、2024-04-08–04-15），每窗 seed 101/211/307，对照 PPO 与 SAC。预登记 SHA-256：`79d3c47183ec75feed46b892630d13cf72f5458f49eaab5d34a2481b90734c24`；源数据快照 SHA-256：`b2bd88627446593ef3f0fe53de1cbb42156f4c164a63a097ec9e77bd7ae14eb2`。模型各训练 262,144 次交互。PPO 使用上一轮较稳的 `log_std_init=-2` 与空仓动作头；SAC 使用 buffer=131,072、warm-up=8,192、batch=256、`train_freq=4`、`gradient_steps=1`、`tau=0.005`、自动熵系数 `auto_0.1`。共同固定 legacy 特征、reward scale=1,000、deadband=0.05、动作和硬风险、样本窗口采样、Base/Stress 成本设定。严格门槛要求每臂每窗每 seed 的 Base/Stress 净收益全正、指标有效、终端平仓且零清算；任何结果都不触碰 2026 外层 Validation/Test/Holdout。

注册配置加载、12 个配置哈希与折内训练工厂身份校验均通过；Python 编译检查通过。运行器 `.tmp/rl-algorithm-comparison-20261015/run_algorithm_comparison.py` 串行完成 12 个冻结模型与 24 项 Base/Stress 回放；没有访问 2026 外层数据。可复核摘要脚本为 `.tmp/rl-algorithm-comparison-20261015/summarize_algorithm_comparison.py`，输出 `analysis-summary.json` 和 `validation-summary.csv`。独立复算验证报告 SHA-256、12 个 bundle、12 份 R1-learning 诊断及 sidecar 全部匹配；每份回放的数值指标有限、终端平仓、观测到清算计数且为零。预登记 SHA-256 为 `79d3c47183ec75feed46b892630d13cf72f5458f49eaab5d34a2481b90734c24`，数据快照 SHA-256 为 `b2bd88627446593ef3f0fe53de1cbb42156f4c164a63a097ec9e77bd7ae14eb2`。最终 `status=rejected`，两算法臂的全候选准入均为 false。

| 算法 | 正收益报告 / 12 | Base+Stress 双正配对 / 6 | 平均净收益 | 中位净收益 | 最差报告 | 平均毛收益（已报告费用前） | 平均换手比 | 平均费用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 5/12 | 1/6 | -0.051217% | -0.030792% | -0.205123% | +0.014862% | 0.5848 | 4.4245 |
| SAC | 0/12 | 0/6 | -0.434909% | -0.210207% | -1.726716% | -0.092820% | 3.4180 | 23.6371 |

| 窗口 / seed | PPO Base / Stress | SAC Base / Stress |
|---|---:|---:|
| 2024-03-08–03-15 / 101 | -0.085398% / -0.132515% | -0.024399% / -0.086723% |
| 2024-03-08–03-15 / 211 | +0.030566% / -0.037291% | -0.071380% / -0.166476% |
| 2024-03-08–03-15 / 307 | +0.062562% / +0.025657% | -0.083508% / -0.266191% |
| 2024-04-08–04-15 / 101 | +0.053944% / -0.153306% | -1.369789% / -1.726716% |
| 2024-04-08–04-15 / 211 | +0.039318% / -0.024293% | -0.793337% / -0.317965% |
| 2024-04-08–04-15 / 307 | -0.188724% / -0.205123% | -0.253939% / -0.058486% |

PPO 优于本轮冻结的 SAC 参数配方：平均换手约为 SAC 的 17.1%，平均费用约为 18.7%；但 PPO 的平均毛收益仅略正，扣除费用、融资与滑点后转负，且只有一个 seed-fold 的 Base/Stress 同时为正，不能作为可交易策略。SAC 在 12 项上全负，Apr seed101 的 Base/Stress 分别亏损 1.37%/1.73%，且两个窗口平均费用均明显更高；这拒绝的是当前 SAC entropy/replay 配置，不能泛化成所有 off-policy 算法必然无效。

训练诊断支持下一步聚焦 PPO 收敛/训练预算，而不是继续增大 SAC 的采样和下单量。PPO 的每模型末块 `explained_variance` 均值为 -0.03066，只有 2/6 为正；policy std 均值约 0.0797。SAC 的末块 critic loss 均值约 0.000967、自动熵系数均值约 0.00401，但低 critic loss 没有转化为经济收益；训练末块 R1 动作活动度不能代替整段验证期间换手结果。PPO 262,144 步提供了 640 次优化更新，覆盖约 141,117 个 Train 决策时间点；需要用 R1 每 8,192 步的解释方差、奖励和时间戳覆盖诊断检验把训练步数翻倍是否有增量，而不是盲目延长训练。

### 下一阶段开发目标

第十一项只比较 PPO 训练交互步数 262,144 与 524,288。使用此前未消费的 2024-03-15–03-22、2024-03-22–03-29 两个内部窗口，每窗 seed 101/211/307；其余观测、动作、奖励、特征、episode、优化器参数、硬风险与成本规则保持一致。共 12 个 Train-only 模型，全部冻结后执行 24 项 Base/Stress 回放。除各预算臂全 seed/折/情景严格正收益门外，还比较同 seed、同折、同成本情景的 12 个长短预算配对差值；要求长预算至少 9/12 配对改善且平均净收益差为正才考虑继续。分析末段解释方差、Value loss、policy std、时间点重复率和净收益路径；如果仅更新次数增加、critic 不改善或成本后表现恶化，则拒绝翻倍预算并转查价值估计与因果状态表征。仍不读取 outer 2026 Validation/Test/Holdout，也不把假设规则场景收益表述为历史账户利润。

第十一项已在训练启动前登记，预登记 SHA-256：`2486a23f3a83b16d4194616cdce10f32dc8cd5c6a9048160abf944a3d79ce936`，其 supersedes 指向第十项预登记 SHA。注册脚本 `.tmp/rl-ppo-budget-comparison-20261015/register_ppo_budget_comparison.py`、固定运行器 `.tmp/rl-ppo-budget-comparison-20261015/run_ppo_budget_comparison.py` 与 12 个配置哈希已通过加载校验和 Python 编译检查；模型尚未开始训练，下一步按登记顺序串行训练，模型哈希冻结后才运行验证。

## 第十一项：PPO 交互步数对照（已完成，2026-10-15）

本项预登记 SHA-256：`2486a23f3a83b16d4194616cdce10f32dc8cd5c6a9048160abf944a3d79ce936`；冻结行情快照 SHA-256：`b2bd88627446593ef3f0fe53de1cbb42156f4c164a63a097ec9e77bd7ae14eb2`。按冻结顺序训练 12 个 Train-only PPO 模型（262,144/524,288 步 × 两个训练截止点 × seed 101/211/307），模型全部冻结后完成 24 个 Base/Stress 回放。独立摘要脚本 `.tmp/rl-ppo-budget-comparison-20261015/summarize_ppo_budget_comparison.py` 逐一复核了预登记和代码快照 SHA、24 份报告 SHA/数值有效性、终端强制平仓、零清算，以及 12 份模型 bundle、R1-learning 文件和 sidecar；所有模型的最终交互步数符合各自臂配置。没有查询 outer 2026 Validation/Test/Holdout。可复核输出为 `analysis-summary.json`、`validation-summary.csv` 和 `paired-budget-summary.csv`。

| PPO 交互预算 | 正收益报告 / 12 | 严格单报告通过 / 12 | 平均净收益 | 中位净收益 | 平均毛收益 | 平均换手比 | 平均费用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 262,144 | 8/12 | 8/12 | -0.004281% | +0.021495% | +0.078789% | 0.7516 | 5.7330 |
| 524,288 | 1/12 | 1/12 | -0.039833% | -0.035004% | +0.007884% | 0.4218 | 3.2025 |

12 个同折、同 seed、同成本情景的长短预算配对里只有 **2/12** 的长预算净收益改善，平均长减短为 **-0.035551 个百分点**，没有达到预登记的“至少 9/12 改善且平均差为正”门槛。短预算有 4/6 个折-seed 的 Base/Stress 双正配对，其中 3/3 集中在第二周；第一周仅 1/3 配对通过。长预算为 0/6 双正配对。两个预算臂全折/全 seed/全成本严格门都失败，因此预算比较状态为 `rejected`，不能据此选择可交易策略。

长预算确实降低了平均换手与平均费用，但毛收益也几乎消失，说明简单增加交互次数没有稳定学到可迁移的市场信号。R1 末训练块诊断中，长预算平均 explained variance 从 -0.0141 上升到 +0.0206；但平均 value loss 从 0.00136 上升至 0.00214，policy std 从 0.07845 降至 0.05249，请求动作绝对值均值从 0.1764 降至 0.1553，no-op 占比从 45.5% 升至 62.0%。每个 8,192 样本训练块内部的决策时间戳重复率为 0；这不表示全训练期没有重新访问同一历史窗口。现有 Train 窗口每模型只有约 106,557 个决策时间点上界，262,144/524,288 次交互相当于在有限历史数据上重复覆盖，增加预算使策略趋于保守却未改善成本后收益。下一步不再把步数翻倍作为默认办法，改做较低学习率对照，并继续将所有旧验证区间视作已消费。

## 第十二项：较低 PPO 学习率的 Train-only 诊断（已完成，2026-10-15）

在启动训练前冻结了单变量比较：学习率 `3e-4`（第十一项同折同 seed、262,144 步的冻结 reference）对 `1e-4`（新训练候选），六组相同训练截止点与 seed 配对；总共只新增六个 262,144 步候选。其它 PPO、观测、动作、奖励、episode、成本和风险设置固定。预登记 SHA-256：`446b482721c076ff6a5df13ed045df499d4574523fd79d18ec1ba974f4474d00`；源代码快照 SHA-256：`62d830b8d05fec401785615776d6acc6d7af3c3ccaf816bfe9ec2a4b007d0041`；数据快照 SHA-256 与第十一项相同。注册脚本、6 份配置、reference bundle/R1 哈希以及 runner 均在训练前固定，Python 编译及候选配置加载检查通过。

本项仅比较 Train 中的 explained variance、value loss、KL/clip/entropy、奖励、动作活动度、回合及时间戳诊断；**不做验证回放、不选择经济策略、不复用已消费窗口，不访问 outer 2026 数据**。即使较低学习率改善 Train 诊断，也只允许注册下一项独立实验；策略经济准入必须等取得质量合格的新鲜行情和有依据的逐日执行规则后重新预登记多折 Base/Stress 检验。

训练后独立核验完成：预登记 SHA-256 `446b482721c076ff6a5df13ed045df499d4574523fd79d18ec1ba974f4474d00` 与源快照 SHA-256 `62d830b8d05fec401785615776d6acc6d7af3c3ccaf816bfe9ec2a4b007d0041` 均匹配；6 个新模型与 6 个冻结 reference 均达到 262,144 步，bundle、R1-learning 和 sidecar 哈希通过，验证查询数为 0。候选最终 explained variance 在 4/6 配对较高；平均 value loss 为 0.001305，对照 0.001358，比例 0.9611。Train-only 门槛通过，故允许对低学习率做预登记经济确认；不构成收益结论。

## 第十三项：PPO 学习率在新内窗的经济确认（已完成，2026-10-15）

预登记 SHA-256：`0a670f3467cb7b5ac273833f076cc880bf813b793284f6b014fdd6af2c7e9cb3`；代码快照 SHA-256：`d11454c7cb50fd7eee69304732e9b50befa484a85a547cb33d668aa8b6014b7f`；行情逻辑快照 SHA-256：`b2bd88627446593ef3f0fe53de1cbb42156f4c164a63a097ec9e77bd7ae14eb2`。比较 PPO 学习率 3e-4 和 1e-4，在 2024-02-01/08 与 02-08/15 两个内部滚动窗口、seed 101/211/307、262,144 步下进行。所有 12 个 Train 模型先冻结，再串行执行 24 个 Base/Stress 回放。独立摘要验证 12 个模型、24 份报告、24 次已完成查询、逐份报告 SHA、账户终端平仓、零清算、配置及模型 sidecar 哈希；没有访问 outer 2026 Validation/Test/Holdout。结果与可复核 CSV/JSON 位于 `.tmp/rl-ppo-lr-economic-wfo-20261015/`。

| 学习率 | 严格正收益报告 | 双正收益折-seed 配对 | 平均净收益 | 中位净收益 | 平均毛收益 | 平均换手比 | 平均费用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3e-4 | 10/12 | 4/6 | +0.05442% | +0.03400% | +0.06577% | 0.09554 | 0.71937 |
| 1e-4 | 10/12 | 5/6 | +0.03534% | +0.03301% | +0.04162% | 0.05584 | 0.41588 |

低学习率减少约 41.6% 平均换手和费用，并多得到一个 Base/Stress 双正折-seed 配对；但 12 个同折、同 seed、同模式净收益配对中只有 6 个胜出，候选减基准均值为 **-0.01908 个百分点**。两臂都有 10/12 份严格正报告，均未达到全 12/12 门槛。低学习率第一折 seed 307 两种模式均为零收益且 Sharpe/Sortino 未定义，基准第二折 seed 307 的 Stress 为负。两者均判为 `rejected`，不能把窗口内正均值解读为稳定可交易利润。

## 第十四项：熵系数对 PPO 动作探索的 Train-only 诊断（已完成，2026-10-15）

以第十二项 1e-4 学习率、2024-03-15 与 03-22 Train 截止点、三个随机种子共 6 个冻结模型作对照，仅改变 PPO `ent_coef`：reference 为 0，候选为 0.005；各新增 6 个候选模型均为 262,144 步。预登记 SHA-256：`f32294806232b7d421ca976698ede389a93b00bdb0995d2723f4b9bcccb891a9`；源快照 SHA-256：`99c544516c2cbf655313cfabb594a0940ab257ab434b620baecd07bca0866bb6`。六组配置经加载校验，除熵系数及输出目录外逐字段一致；reference 模型文件存在且已登记其哈希，运行器与注册器通过 Python 编译检查。

本项不运行回测或验证、不访问 outer 2026 数据。预先固定诊断门槛：候选 explained variance 至少 4/6 配对提升，平均 value loss 不高于 reference 的 1.10 倍，请求动作绝对值至少 4/6 配对不降低，且平均 no-op 比例上升不超过 0.02。通过仅意味着保留作为继续研究的候选；无论结果如何均不能证明盈利。

本项预登记 SHA-256：`f32294806232b7d421ca976698ede389a93b00bdb0995d2723f4b9bcccb891a9`，源快照 SHA-256：`99c544516c2cbf655313cfabb594a0940ab257ab434b620baecd07bca0866bb6`。六个 candidate 和六个冻结 reference 均通过配置/模型/学习记录/sidecar SHA 校验，交互步数均为 262,144；没有 Validation/Test/Holdout 查询。平均 value loss 从 0.001305 降到 0.001264（candidate/reference=0.9683），但 explained variance 只在 2/6 配对提升，请求动作绝对值只在 3/6 不低于对照；平均 no-op 比例变化 +0.00063（对照 0.33695，候选 0.33759），policy std 也基本相同。四项预登记条件中两项失败，诊断门槛未通过。说明 `ent_coef=0.005` 没有一致扩大有效动作支持，不能进行经济确认。

可复核文件位于 `.tmp/rl-ppo-entropy-train-only-20261015/`：`analysis-summary.json`、`paired-train-diagnostics.csv` 与逐训练块 `R1-block-diagnostics.csv`。

## 第十五项：较强熵正则系数的 Train-only 响应曲线（已完成，2026-10-15）

因第十四项熵系数 0.005 没有显示一致的动作/价值改善，本项只将 `ent_coef` 从 reference 0 调到 0.02；继续使用同一组六个固定训练折/种子的 1e-4 reference，新增六个 262,144 步候选。训练前固定的预登记 SHA-256：`b2431af89062641004d14fe33e534d36687b4bf56e78fbf542b5b151e8b977ac`；源代码快照 SHA-256：`c16f2caea3b7ffde2150c5997363ade1475d8d9577c0a0e64e29f2dae00da205`。训练后独立汇总验证了 12 个模型、六份候选配置、source/preregistration 身份及 model bundle、R1-learning 和 sidecar 哈希；所有模型达到 262,144 步。candidate explained variance 只在 **1/6** 配对提高，value loss 均值为 0.001244（reference 0.001305，比例 0.9531），请求动作绝对值只在 **2/6** 配对不低，no-op 均值反而下降 0.00405。四项门槛有两项失败，Train-only 诊断未通过；未查询任何验证/回测，outer 2026 查询数为零。`ent_coef=0.02` 不进入经济验证。可复核证据位于 `.tmp/rl-ppo-entropy-strong-train-only-20261015/analysis-summary.json`、`paired-train-diagnostics.csv` 与 `R1-block-diagnostics.csv`。高熵没有解决 critic 可解释方差和动作支持问题，下一项转测 PPO 每批更新轮数。

## 第十六项：PPO 更新轮数 Train-only 诊断（已完成，2026-10-15）

单因素候选将 n_epochs 从冻结对照的 5 提高至 10，学习率固定为 1e-4；其余网络、观测、动作、reward、episode=8,192、262,144 次交互、数据窗口、seed、成本/风险规则及环境设置保持逐字段一致。预登记 SHA-256：61e85e7fc06315ebf22391fd0182fac66a749512f77e366e104ec727b102db26；源码快照 SHA-256：1efae9d1b885e7d3f7bc14fcc8b66936bf635b277f3a1951a5c6764eaab366f4。六个候选和六个冻结 reference 均完成 262,144 步；独立汇总验证了 12 个模型、配置逐字段差异、bundle、R1-learning 及 sidecar 哈希，validation query 为 0，outer 2026 query 为 0。证据位于 .tmp/rl-ppo-epochs-train-only-20261015/。

候选 explained variance 在 4/6 配对提升；mean value loss 从 0.001305 降至 0.001202（ratio 0.9210）；候选均值 approximate KL=0.00301、clip fraction=0.02569，低于稳定性上限。配对请求动作绝对值仅 3/6 不低；平均 no-op 从 0.33695 升至 0.40149（+0.06453）。因此未通过预登记门槛：增加 epoch 虽改善部分 critic 指标，但策略活动与空仓行为响应不一致且空仓比例明显增加；候选不进入经济测试。

## 第十七项：PPO 更新轮数下调 Train-only 响应曲线（已完成，2026-10-15）

六组候选只把冻结 reference 的 n_epochs 从 5 改为 3，各自完成 262,144 步。预登记 SHA-256：c2aa13418e675a8f322dfe34578480c03ab3eb9fabdc3b37d047b9d1c22a8154；源码快照 SHA-256：bc503ede6c723e4e079098bf0beeeb637ae876d2cb9f12c8eb4cacd1d3589bf1。独立汇总核验了 12 个 candidate/reference、配置、bundle、R1-learning 与 sidecar 哈希；所有 Validation 和 outer 2026 查询为 0。证据保存在 .tmp/rl-ppo-epochs3-train-only-20261015/。

n_epochs=3 的 explained variance 在 0/6 配对提升；value loss 比例为 1.0216；均值 KL=0.00202、clip=0.00529，更新稳定但学习拟合不足。请求动作幅度仅 1/6 配对不低，平均 no-op 从 0.33695 降至 0.31059（-0.02637），显示动作参与率增加但动作幅度整体变小，没有改善 critic。诊断门槛失败，3 epoch 不保留为候选。

## 第十八项：PPO 更新轮数 7 epoch Train-only 插值诊断（已完成，2026-10-15）

六组候选只把 n_epochs 从 5 改到 7，各完成 262,144 步。预登记 SHA-256：db6340fa48bfeff352c1b1fd0a4f4d6274aa3fecc843197c85aece064498a571；源码快照 SHA-256：332faecc64a5ee591e6021f972e84ade2b7bc2c27fd30296b72c39cc0dcce544。独立汇总验证 12 个模型、候选配置、source/preregistration 身份及模型/R1-learning/sidecar 哈希；validation 和 outer 2026 查询数均为 0。结果位于 .tmp/rl-ppo-epochs7-train-only-20261015/。

EV 只在 1/6 配对提升；平均 value loss 从 0.001305 降到 0.001188（ratio 0.9100）；平均 KL=0.00341、clip=0.02093，优化稳定。配对动作幅度只在 3/6 不低，平均 no-op 从 0.33695 升到 0.36464（+0.02769），同时未达到 no-op 上限；故门槛失败。7 epoch 不进入经济验证。连续三个轮数实验显示提高更新轮数降低了平均 value loss，但没有稳定改善解释方差或动作支持。

## 第十九项：多周期闭合行情特征 Train-only 诊断（已完成，2026-10-15）

预登记 SHA-256：`37aafdf77edb70aa1e7d84742c8f2c9d2f0560204553e70d9827dc2191651bc3`；源码快照 SHA-256：`293bdf6eecd82acfe8dc524641a5eac269d620da06239e9c1cb8807ab9f3113c`；行情逻辑快照 SHA-256：`b2bd88627446593ef3f0fe53de1cbb42156f4c164a63a097ec9e77bd7ae14eb2`。比较 legacy 与 causal_mtf_v1（增加闭合 5/15/60 分钟收益和 15/60 分钟波动特征），两折、三 seed、262,144 步；只读取 Train 诊断，不做验证或经济准入。

训练前曾有一次把 `feature_profile` 写在 config 根层的无效注册尝试（原 prereg SHA-256 `88f4591d0ef07fb7ed1ae0b3760e93d20dfbe78f5a4bf7c82176e650ab66977d`）。配置加载在任何训练启动前拒绝了该模式；没有生成模型，也没有验证查询。随后按项目 schema 将字段放到 `environment.feature_profile`，以新的上述 prereg 登记并成功完成训练。因此，只把修正后的第十九项计入结果。

可复核证据位于 `.tmp/rl-ppo-causal-mtf-v1-train-only-20261015/`。独立汇总验证 12 个 candidate/reference、配置与 source/preregistration 身份、bundle/R1-learning/sidecar 哈希和最终步数；六个模型均完成 262,144 步，验证查询和 outer 2026 查询均为 0。candidate explained variance 配对胜出 **0/6**；value loss 均值比例为 **0.95183**；请求动作幅度不低仅 **1/6**；no-op 均值变化 **-0.00425**。综合训练诊断门槛失败。虽然 value loss 略低且 no-op 略减，多周期输入使 EV 全面转差、动作幅度多数走低，不进入经济验证。

## 第二十项：PPO GAE 时间尺度 Train-only 诊断（已完成，2026-10-15）

本项冻结 candidate `gae_lambda=0.90`，对照第十二项六个 `gae_lambda=0.95` 模型；legacy 观测、其它 PPO 参数、动作/奖励、Train 窗口与 seed、262,144 步保持不变。预登记 SHA-256：`122a79d97c0f5d29cb447c17d641942cb4041fc93b133717dcccfe558705129b`；源码快照 SHA-256：`de5a7b399013593bc226d7332b22ab05b13c0332d2cba1190c172323f711bbda`。六个候选均完成训练，独立汇总验证 12 个 candidate/reference 的 config、source/prereg 身份、bundle、R1-learning 和 sidecar SHA，均达到 262,144 步；验证与 outer 2026 查询均为零。证据目录：`.tmp/rl-ppo-gae-lambda-090-train-only-20261015/`。

候选 explained variance 在 **4/6** 配对提升，平均 value loss 比例 **0.6077**，请求动作幅度在 **5/6** 不低，no-op 均值仅增加 **0.00411**；候选均值 KL=0.00552、clip=0.03053，均在稳定性上限内，Train-only 门槛通过。该结果保留 0.90 作为训练诊断候选，但没有进行经济检验，也不构成收益证据。

登记元数据有一个笔误：`preregistration.json` 的 `stage` 字符串因复制模板仍写“Stage 19”，但参数、六份候选配置、冻结 reference、代码快照、产物和实际实验均是 GAE lambda 对照。为保持原始证据链，未改写 preregistration；`.tmp/rl-ppo-gae-lambda-090-train-only-20261015/metadata-erratum.json` 记录原 SHA、误写内容和校正后的阶段说明。独立摘要通过不消除这项登记质量缺陷，后续注册必须单独审查阶段编号。

没有可用于该候选的合格新鲜盈利验证窗口：项目审计确认最新追加的 2026-08-12 至 2026-10-12 行情缺少适用历史 instrument rule、数据质量为 DQ-C，且不具备完成三折加 Holdout 所需的合格数据；既有 WFO 区间亦不可重新标为未消费。因此本项严格止于 Train-only，继续研究优化器诊断。

## 第二十一项：GAE lambda 0.85 Train-only 响应曲线（已完成，2026-10-15）

预登记 SHA-256：`3ec521358c40131a70bef5b73a92d49e5fdc3450f86ee0b55ba5a6066dd329c6`；源快照 SHA-256：`7e234cf68d7a7fba412c3c4d537d78484834d4615b0b9be6e09de775cf7bef04`。相对 Stage 20 冻结 reference `gae_lambda=0.90`，candidate 仅改为 0.85；六个候选完成 262,144 步。独立摘要验证 12 个配置/模型、source/preregistration、bundle、R1-learning、sidecar 与步数哈希；验证查询为零。证据在 `.tmp/rl-ppo-gae-lambda-085-train-only-20261015/`。

EV 配对提升为 **3/6**（门槛 4/6）；平均 value loss 比例 **0.8673**；请求动作幅度 **6/6** 不低，均值 0.2023 对 0.1730；no-op 均值增幅 **0.00165**；候选均值 KL=0.00654、clip=0.03787。除 EV 配对数未达门槛外，其余边界满足预设范围，因此 0.85 不保留为优选配置，当前诊断支持 0.90。Train-only 结果仍不构成收益证据。

## 第二十二项：GAE lambda 0.925 插值 Train-only 诊断（已完成，2026-10-15）

预登记 SHA-256：`9b912a443ac0402285c530a4e17c114320cd64fa85f1a8d0ac4fa1d579e136f8`；源快照 SHA-256：`b0f66ef6464969493f5899464c09e385e6b06a9be4571ae46a020d6d4ce58044`。candidate `gae_lambda=0.925` 与第二十项冻结 0.90 reference 配对，两折、三 seed，各 262,144 步。独立核验 12 个模型、配置、source/preregistration、bundle/R1-learning/sidecar 哈希均匹配；验证查询为零。结果在 `.tmp/rl-ppo-gae-lambda-0925-train-only-20261015/`。

EV 仅 **1/6** 配对改善，value loss 比例 **1.2288**，请求动作幅度不低 **3/6**；no-op 平均反而下降 0.00214，候选均值 KL=0.00507、clip=0.02962。学习与动作指标未达门槛，明确拒绝 0.925。结合 0.85、0.90、0.925 三点，当前 Train-only 诊断支持 0.90；此结论只选取下阶段研究的优化器配方，不代表经济或收益选择。

## 第二十三项：PPO gamma 0.95 时间尺度 Train-only 诊断（已完成，2026-10-15）

预登记 SHA-256：`4a0015021f6caf7324e53ab57f339117e5adf7ebbee82a758ee917a6f9355e79`；源码快照 SHA-256：`5b755ae788d366f549d131af11ce3154f5bcd67fa28624d920278c603d73ffb6`。candidate 只把 `gamma` 从 0.99 降至 0.95，固定 GAE lambda=0.90；六个候选均完成 262,144 步。独立摘要核验 12 个 candidate/reference、配置/source/preregistration、bundle/R1-learning/sidecar 哈希及最终步数；Validation 查询为 0。证据目录 `.tmp/rl-ppo-gamma-095-train-only-20261015/`。

EV 仅 **1/6** 配对提升；value loss 比例 **0.7049**，请求动作幅度不低仅 **2/6**，no-op 增幅 **0.00173**，候选均值 KL=0.00604、clip=0.0345。尽管 value loss 降低，解释方差和动作支持未达门槛，拒绝 gamma=0.95；继续保留 gamma=0.99、GAE lambda=0.90 作为训练诊断基线。

## 第二十四项：奖励函数回撤增量 shaping Train-only 消融（已完成，2026-10-15）

发现 `RewardConfig` 已有回撤增量惩罚，但历史环境工厂一直使用默认值，应用配置不能控制它。本轮增加环境字段 `reward_lambda_dd`（默认 0）并将非默认值绑定进 environment/config/transition 身份和模型 recipe；`lambda_guard` 暂保持 0。新增测试确认默认身份与旧模型一致、非默认 shaping 改变身份，且送给 learner 的 `r_shape` 不改写 native `r_econ`/账户净值。定向 reward、factory identity 测试 **15 passed**；CLI 定向组合 **37 passed**；Ruff 通过。历史工厂完整测试组合在此 Windows sandbox 中还遇到硬链接 `os.link` 权限拒绝和既有负时间戳夹具错误，不能宣称全量工厂回归通过。

研究比较 candidate `reward_lambda_dd=0.1` 与冻结 Stage 20 zero-shaping reference；gamma=.99、GAE lambda=.90、其它 PPO 参数、动作、特征、窗口/seed、交易成本和风险规则固定。该惩罚只作用于 learner reward 的 drawdown 增长项，账户记账、成本、native `r_econ` 与风险门保持原定义。预登记 SHA-256：`9bcc970b4cce3ef735c557ca87d86a7c7e988ccc3fe95177da20b9ca4a2210f1`；源码快照 SHA-256：`89f355c897a8223ce2eff2f1784b26835dcd845b759424532cea0f879457a0f4`。六个候选均完成 262,144 步，独立摘要确认 12 个 candidate/reference 的配置、源码/预登记身份、模型/learning/sidecar 哈希及最终步数，验证和 outer 2026 查询均为 0；证据目录 `.tmp/rl-ppo-reward-dd-010-train-only-20261015/`。EV 配对胜出 **3/6**，value loss 均值比例 **0.8497**，请求动作幅度不低 **0/6**，no-op 均值变化 **-0.00533**，候选均值 KL=0.00440、clip=0.01875；综合门槛失败。平均 learner shaping 为负（约 `-6.5e-7`/step），原生 `r_econ` 保持单独记账且未被改写。由于 value loss 改善伴随六个配对动作幅度全部降低，拒绝将 0.1 晋升为候选；这只是训练诊断，不能推断收益。

结果摘要 JSON 的 schema 字符串沿用了旧 GAE 实验名 `rles.gae-lambda-085-train-only-analysis.v1`，但它的 preregistration SHA、source SHA、数据身份、模型配对与代码执行都属于本 Stage 24。为保持原始产物不变，`.tmp/rl-ppo-reward-dd-010-train-only-20261015/metadata-erratum.json` 记录该报告标签错误及摘要哈希；应以目录和预登记 SHA 识别实验，不能把旧 schema 标签视作本实验身份。

## 第二十五项：较弱回撤 shaping 响应测试（已完成，2026-10-15）

Stage 24 中 `lambda_dd=0.1` 的 shaping 平均幅度约为原生经济奖励的 14%，却令所有六组配对请求动作幅度变小；因此本项沿用相同双折/三 seed Train-only 协议，只把候选改为 `reward_lambda_dd=0.025`，与 Stage 20 的零 shaping 参考配对。其余 PPO 配置、行情窗口、seed、训练步数、动作、特征和执行/风险规则不变。预登记 SHA-256：`bd90a815fff9146b8b3cb3769b9d9d00114c8d71c9793e86d46a0aeabe73827f`；源码快照 SHA-256：`519484ff50fb43cfdc25b40ab1745a0b70730d66d386db92c647144c661cee7e`；证据目录 `.tmp/rl-ppo-reward-dd-0025-train-only-20261015/`。六个候选均完成 262,144 步，独立摘要核验 12 个模型的配置、预登记/源码/数据身份、模型和 learning 文件与 sidecar 哈希，验证和 outer 2026 查询为 0。EV 胜 **3/6**，value loss 均值比例 **0.9812**，请求动作幅度不低 **0/6**（均值 0.1647 对 0.1730），no-op 变化 **-0.00132**，候选均值 KL=0.00510、clip=0.02957；门槛失败。候选平均 shaping 为 `-1.80e-7`/step，平均 `r_econ` 为 `-4.60e-6`/step，和零 shaping 参考非常接近；较弱惩罚减小了策略动作收缩，但没有恢复任何配对的动作幅度，故不保留非零 `lambda_dd`。训练证据不能证明经济盈利。

## 第二十六项：降低执行 deadband 的动作可达性诊断（已完成，2026-10-15）

连续动作会先变成目标净敞口，再经 soft/hard risk、deadband、最小数量和最小名义额规则。参考配置 deadband 为 0.05，训练末段约 34% transition 是 no-op；reward shaping 消融已否定为主要解决办法。本项只把 deadband 降至 0.025，使用相同两折、三 seed、262,144 步、零 shaping 和 Stage 20 冻结 PPO 设置。首次预登记因 runner 的 Ruff 检查未过、且在任何模型启动前阻断而保留为 `preregistration-prelint.json`，最终登记记录了该修正。有效预登记 SHA-256：`8c1e285d9b8fcf3e2da19a21e85b6da46c1819451ccb1f21f79a884f1fbcba28`；源码快照 SHA-256：`33238e1a7f1a5d19e8772341272e3a2dcc9bece9b748c418d3f9adeb242535b8`；证据目录 `.tmp/rl-ppo-deadband-0025-train-only-20261015/`。六个候选完成训练，摘要确认 12 个模型身份、配置、bundle/learning/sidecar 哈希和最终步数，Validation 与 outer 2026 查询为 0。候选 EV 仅 **2/6**；value loss 比例 **0.7878**；submit 均值增加 **0.16498**、no-op 同幅下降 **0.16498**；请求动作幅度不低 **0/6**；KL=0.00496、clip=0.02531。Train 原生 `r_econ` 均值由 reference 的 `-4.614e-6` 降至 `-4.898e-6`/step，差 **-2.84e-7**，约恶化 6.1%。因此 deadband=0.025 虽增加提交覆盖，但扩大的小额交易主要被成本拖累，综合门槛失败；保留 deadband=0.05。

## 第二十七项：PPO rollout 长度 Train-only 对照（已完成，2026-10-15）

奖励参数和执行阈值两项探索均没有通过训练门槛，下一步检查时间相关的 PPO 更新是否更合适。本项只把 `n_steps` 从冻结 reference 的 2,048 增至 4,096，`batch_size=512`、每批 epoch、学习率、gamma、GAE lambda、总交互数 262,144、legacy 输入、动作与交易约束均固定。沿用两折三 seed 配对，评估 EV/value loss、KL/clip、请求/执行动作和 no-op 轨迹；完全不运行收益回放或 Validation/Test/Holdout 查询。预登记 SHA-256：`5f884bf6cbf57be969131524f882f4cdf51fe98cd9c1cc790abcbc9485309e73`；源码快照 SHA-256：`15826bf8fde26dabb478ff63abfc4ad760a500ee023673d387e539d06659fdeb`；证据目录 `.tmp/rl-ppo-nsteps-4096-train-only-20261015/`。决策门槛要求 EV≥4/6、value loss≤1.10x、KL≤0.02、clip≤0.10、请求动作幅度不低于 reference 的配对≥4/6、no-op 均值增幅≤0.02。任何 Train-only 通过只说明优化器诊断更好，不构成盈利性证据。

候选六个模型全部完成 262,144 步。独立汇总核验了 12 个候选/基准模型的配置、源代码/预登记身份、bundle/R1-learning/sidecar 哈希、最终步数；Validation 与 outer 2026 查询数为 0。结果在 `analysis-summary.json` 与配对/R1 block CSV。候选 explained variance 仅 **1/6** 配对更高；平均 value loss 为 **0.90185×** reference；请求动作幅度不低为 **0/6**（均值 0.14517 对 0.17302）；no-op 均值降低 0.00279；KL=0.00298、clip=0.02161。value loss 与 no-op 小幅改善，但六个种子的请求动作幅度全部收缩、EV 只在一组改善，Train-only 综合门槛失败，故不保留 `n_steps=4096`；回到 `n_steps=2048` 基线。本结果不涉及经济回放，也不代表盈利性。

2026-10-15 并发资源实测：用户要求整机空闲物理内存目标改为约 1 GiB。正式 learner 为 CPU 设备，单 worker RSS 峰值约 6.5 GiB；RTX 5070 Laptop 总显存约 8 GiB，但该阶段 RL learner 未使用 GPU。单进程阶段系统可用内存约 12–13 GiB，CPU 总利用率约 3–8%（16 核/32 线程上的单环境 worker 主导）。独立 smoke 在 Stage 27 最后一模型运行时同时准备了相同 Train 环境并训练 16,384 步，独立目录、零验证查询；环境准备 263.6 秒、总耗时 322.6 秒，交互步数核对为 16,384。两进程期间物理可用内存最低约 6.4 GiB，未触及 1 GiB 线；该短程测试表明单环境 learner 的并行速度有竞争损耗，但完整折级并发可把两个折的准备阶段重叠，并让每 worker 在已准备环境内复用三个 seed。后续 formal worker 先限制为两个（每折一个、独立状态与输出），派发前/运行中检查内存与提交额度；可用物理内存低于 1 GiB、发生分页/OOM 或端到端吞吐下降时中止新增 worker 并回退串行。并发产物仍须独立 preregistration、配置、状态和哈希。

## 第二十八项：PPO rollout 1,024 步对照（无效执行，2026-10-15）

原计划以 `n_steps=1024` 对比冻结 Stage 20 的 2,048 reference。候选配置和预登记 SHA-256（`97a1a26cd56116dbe92c83b4f0cf86f575ba8b4e4e749e59a12f824c12ca70f8`）均正确；但 runner 在做 one-factor 配置比较时浅拷贝了顶层 dict，随后把 reference 的 n_steps 写入共享的嵌套 `hyperparameters`，实际传给 learner 的候选因此也变成 2,048。尽管六份模型都达到 262,144 步且产物哈希有效，六个 learner config hash 都与各自 reference 相等，32 个训练块的优化器/动作诊断也逐块完全相同。该批产物不是 1,024 对 2,048 的有效对照，**不作接受或拒绝 1,024 rollout 的统计结论**。独立错误记录 `.tmp/rl-ppo-nsteps-1024-train-only-20261015/metadata-erratum.json` 保留原始登记和产物；汇总器曾遗漏 learner config-hash 区分校验，故纠正重跑增加了这一断言及 SB3 更新总数（1024：1,280；2048：640）的核验。

## 第二十九项：PPO rollout 1,024 步纠正重跑（已完成，2026-10-15）

沿用 Stage 28 的单变量方案，但使用新代码身份、预登记、配置和输出路径。修正 runner 对候选/reference config 使用深拷贝，并断言比较后候选仍为 1,024；summarizer 断言两边 learner config hash 不同且最终 PPO 更新计数分别为 1,280/640。预登记 SHA-256：`a58a055f66372ad68511cf04dd139f1936f3cc03b81f5c0d094deb237f1b44c9`；源码快照 SHA-256：`5e0192c39b2b7442a16261f1c6f90e17d7a09d86f3861a86498a3136fbdf1db8`；证据目录 `.tmp/rl-ppo-nsteps-1024-corrected-train-only-20261015/`。Ruff 与 py_compile 检查通过；两折各由独立 CPU worker 并行，每折 prepare 一次并顺序跑三个 seed。物理可用内存运行中通常 9–10 GiB，准备/切换瞬时谷值约 1.4 GiB 后即恢复，未发生持续低于 1 GiB 或 OOM；验证与 outer 查询为零。

独立汇总器验证 12 个模型的 bundle/R1-learning/sidecar SHA、source/preregistration 身份和最终 262,144 步，并断言六个候选 learner config hash 均不同于 2048 reference，更新数均为 1,280 对 640。EV 改善配对 **0/6**，平均 value loss 比例 **0.7527**，请求动作幅度不低 **5/6**（均值 0.1960 对 0.1730），no-op 均值变化 **-0.00222**，KL=0.00471、clip=0.01888。门槛失败只因 EV 对数未达 4/6；拒绝 `n_steps=1024`，当前保留 `n_steps=2048` 作为对照基线。训练诊断不是收益结论。

## 第三十项：PPO critic loss 权重 Train-only 对照（已完成，2026-10-16）

只把 PPO `vf_coef` 从冻结 Stage 20 的 0.5 提至 1.0；`n_steps=2048`、batch/epoch、学习率、gamma、GAE、总交互步数、legacy 观测、动作、reward、风险/交易成本、两折与三 seed 均保持不变。预登记 SHA-256：`ede4e5bd117f9711e04bb1723de26592e39dd022cf40e2f5325b2a49b633b916`；源码快照 SHA-256：`082bd1c807c2dc69e51ca894a473222db5d3fd2b8c9a9db1aab2af2b84c9a005`；证据目录 `.tmp/rl-ppo-vf-coef-100-train-only-20261015/`。Ruff、py_compile 和候选/基准逐字段比较检查通过。训练按两个 CPU worker 并行，每折复用环境并顺序跑三个 seed；Validation/Test/Holdout 查询为零。

独立汇总核验了 12 个模型身份、bundle/R1-learning/sidecar SHA 和 262,144 步，并断言 candidate learner config hash 与基准不同、两个臂均为 640 PPO updates。候选 EV 配对提升 **5/6**，value loss 比例 **0.9972**，KL=0.00511、clip=0.02651；但请求动作绝对值不低仅 **2/6**（均值 0.16859 对 0.17302），no-op 均值微增 0.00047。Train-only 门槛因动作幅度条件失败；`vf_coef=1.0` 不单独晋级。它的 critic 指标较稳但策略动作支持收窄，后续仅作条件 reference 检查更强的初始策略方差。

## 第三十一项：高 critic 权重下的 PPO 初始动作方差对照（已完成，2026-10-16）

固定 `vf_coef=1.0`，以第三十项六个已冻结 candidate 为 reference，只把 PPO `log_std_init` 从 -2 提到 -1；`n_steps=2048`、其它网络/优化器/训练窗/seed、262,144 步、legacy observation、动作和成本规则保持不变。目的是检验更宽的初始连续动作分布能否缓解 Stage 30 的动作支持收缩，同时保持 EV 改善。只使用 Train-only 诊断，不做经济回测或任何验证查询。候选/reference learner config hash 必须不同，二者预期 update count 均为 640；每折独立 CPU worker，物理内存下限 1 GiB。预登记 SHA-256：`3655ba58f7537225f3f053e61f3e0284b7f45990ed4d996c3e075adb27c12842`；源码快照 SHA-256：`6a6d09cda3b9a5ab58f7a4217e23bb29a979561f6668f1c03a43e8f6770490b0`；结果目录 `.tmp/rl-ppo-vf100-logstd-minus1-train-only-20261016/`。Ruff、py_compile 和逐字段单因素检查通过。六个候选和六个冻结 reference 均达到 262,144 步；配置、模型包、学习记录及 sidecar 哈希全部通过独立汇总核验，PPO update count 均为 640，Validation/outer 2026 查询为 0。候选 EV 仅 1/6 配对提高，value loss 均值从 0.000791 增至 0.001169（1.478 倍）；请求动作绝对值 6/6 增大（均值 0.16859→0.23999），no-op 比例 0.34153→0.12087，但候选训练经济奖励均值从 -4.62e-6 降至 -1.05e-5。KL 均值 0.00115、clip fraction 0.00016。动作活动增加伴随 value fit 与经济奖励恶化，未通过预设门槛；不进入回测。


## 第三十二项：中间 PPO 初始动作方差 Train-only 诊断（已完成，2026-10-16）

在 Stage 31 的 `log_std_init=-1.0` 显著扩大动作、却恶化 EV/value loss/训练经济奖励后，测量 -2.0 与 -1.0 之间的中间点：候选 `log_std_init=-1.5`，reference 为 Stage 30 冻结的 -2.0 模型。固定 `vf_coef=1.0`、`n_steps=2048`、学习率 `1e-4`、`n_epochs=5`、GAE `0.90`、六个折-seed 配对、各 262,144 步及其余训练/观测/执行参数，仅改变初始策略 log standard deviation。维持两路独立 CPU fold worker和至少 1 GiB 可用物理内存；仅 Train-only 诊断，不做经济回测或任何验证查询。预登记的 EV/value-loss/动作/no-op 诊断门槛沿用上一项。目标是判断适度扩大初始探索能否保留 critic 学习质量；即使门槛通过，也只允许进入后续受控经济验证，不代表已盈利。预登记 SHA-256：`9ea982d77e3e7014b00f2f694f7102702ddd1d51a12361e3240188317395f2fc`；源码快照 SHA-256：`df279b9f5495205d5a90d61b765ab4a191dc386074f090dc5cbd98e14db9be82`。六个候选配置逐字段单因素校验与 Ruff/py_compile 通过后冻结预登记；实验目录 `.tmp/rl-ppo-vf100-logstd-minus1p5-train-only-20261016/`。Ruff、py_compile 和配置逐字段单因素核验通过后才启动训练。六个候选/reference 均完成 262,144 步；独立汇总验证 12 个模型、训练记录/sidecar 与所有哈希，PPO update count 均为 640，Validation/outer 2026 查询为 0。候选 EV 仅 2/6 配对提升，value loss 均值由 0.000791 上升到 0.001198（1.515 倍）；请求动作绝对值 6/6 更大（0.16859→0.20484），no-op 从 0.34153 降到 0.21407，但训练经济奖励均值仍由 -4.62e-6 降到 -7.47e-6。候选 KL=0.00368、clip fraction=0.01286。中间值没有保住 critic 学习质量，未通过门槛，不进入经济测试。


## 第三十三项：PPO critic loss 权重插值 Train-only 诊断（已完成，2026-10-16）

Stage 30 的 `vf_coef=1.0` 提高了 EV 配对胜率至 5/6，但动作支持门槛失败；Stage 31/32 显示通过放宽初始动作分布增加交易动作会明显恶化 value loss 与训练经济奖励。因此回到优化目标本身，测试 `vf_coef=0.75` 是否能保留部分 critic 改善并恢复动作支持。候选以 Stage 30 的六个冻结 `vf_coef=1.0` 模型为配对 reference；仅改变 `vf_coef`，其余使用一致的 `log_std_init=-2.0`、`n_steps=2048`、学习率 `1e-4`、`n_epochs=5`、GAE `0.90`、同六个 Train cutoff/seed、每模型 262,144 步及相同观测/动作/reward/成本规则。Train-only 诊断，不回测、不查询 Validation 或 outer 2026。预设 gate 继续要求 EV 配对提升、value loss 不显著恶化、动作支持不下降及 no-op 比例不增加过多；仅通过后才讨论独立经济确认，不代表盈利。预登记 SHA-256：`d9a251ce3bee6f0e4ad67a608315fd5c664f8907c564c450590a76937d37ac6b`；源码快照 SHA-256：`26f8935b901f6fe4b49bfee7f00cb9f37bd3ab4e6d01b2ecf425440b80a3e628`。六个候选相对 Stage 30 reference 已逐字段确认仅 `vf_coef` 不同。12 个模型全部达到 262,144 步，bundle、学习记录和 sidecar 哈希核验通过，update count 均为 640，Validation/outer 2026 查询为 0。候选 EV 配对胜仅 2/6；value loss ratio=1.0495，动作绝对值在 5/6 配对不低（均值 0.16859→0.17681），no-op 变化 +0.00035，KL=0.00585、clip=0.03460。训练经济奖励均值为 -4.62e-6 对 -4.61e-6，仍为负且只作诊断记录。预设门槛因 EV 不通过；但它显示其它指标保持稳定，故继续测更接近 1.0 的 0.9，不进入回测。实验目录 `.tmp/rl-ppo-vfcoef-075-train-only-20261016/`。


## 第三十四项：PPO critic loss 权重近邻响应 Train-only 诊断（已完成，2026-10-16）

Stage 33 的 `vf_coef=0.75` 满足 value-loss、动作幅度和 no-op 条件，却未满足 EV 配对胜率；针对 Stage 30 的冻结 `vf_coef=1.0` 模型再测较近的 `vf_coef=0.9`，观察 EV 信号是否恢复，同时保留 Stage 33 的其它诊断优势。只改变 `vf_coef`，固定 `log_std_init=-2.0`、`n_steps=2048`、学习率 `1e-4`、`n_epochs=5`、GAE `0.90`、六个相同 Train cutoff/seed、262,144 步及数据、观测、动作/reward/成本规则。只做 Train-only 诊断，不查询 Validation 或 outer 2026，不把训练收益视为盈利证据。按相同预设 gate 判定是否可进入单独的经济确认。预登记 SHA-256：`bdbbb581ffa09a5e4ec5b34b512d65595c53c09393d59034ad64f0eaa5d997da`；源码快照 SHA-256：`2cd6a4b874d7dd04973c517de36b72328c4afebd372eac8027c441c9be85feec`。六组候选与 Stage 30 reference 逐字段仅在 `vf_coef` 上不同。12 个模型均完成 262,144 步并通过 bundle/learning/sidecar 哈希核验，update count 均为 640，Validation/outer 2026 查询为 0。EV 仅 2/6 配对提升；value loss ratio=1.0058，请求动作不低为 4/6，no-op 变化 -0.00149，KL 与 clip 均在门槛内。仍未通过预设 EV 条件，不进入回测。实验目录 `.tmp/rl-ppo-vfcoef-090-train-only-20261016/`。


## 第三十五项：PPO critic loss 权重 0.95 Train-only 诊断（已完成，2026-10-16）

Stage 34 将 `vf_coef` 降至 0.9 后 value loss、动作幅度和 no-op 条件保持可接受，但 EV 仍只有 2/6 配对提升。继续靠近 Stage 30 的 EV 较强点，测试 `vf_coef=0.95` 对冻结 Stage 30 `vf_coef=1.0` 模型；仅改变该超参数，所有数据、Train 窗、六个 seed 配对、训练步数 262,144、`log_std_init=-2.0`、PPO 优化/观测/动作/reward/成本设置固定。使用两路 CPU worker，训练前后保留至少 1 GiB 可用内存；仅 Train-only 诊断、不访问 Validation/outer 2026。按统一 gate 判定；任何通过也仅保留为经济测试候选，不能据此声称盈利。预登记 SHA-256：`20be29a0ab9183f01801fe7e3671079283f3bce31b495a215540458a760fbcef`；源码快照 SHA-256：`dd171a38e80606d7663e8eb2ab8d28e57b5aeb8f9b34388af1b4d7c9c9d47732`。Ruff、py_compile 与 source identity 检查通过，六组候选/reference 已逐字段核验仅 `vf_coef` 不同。12 个模型全完成 262,144 步，所有哈希与 PPO 更新数通过独立核验；Validation/outer 2026 查询为 0。候选 EV 在 4/6 配对改善，value loss ratio=1.00743，动作绝对值在 5/6 不低（均值 0.16859→0.17199），no-op 变化 -0.00155；KL=0.00556、clip=0.03107，训练经济 reward 仍约为 -4.62e-6。预登记 Train-only gate 通过，故保留为研究候选，但还没有正收益的独立测试证据。实验目录 `.tmp/rl-ppo-vfcoef-095-train-only-20261016/`。


## 第三十六项：PPO 交互步数加倍 Train-only 稳健性诊断（已完成，2026-10-16）

在 Stage 35 `vf_coef=0.95` 通过 Train-only gate 后，检验增加学习步数是否继续提升 EV/动作诊断，或出现重复采样后的退化。候选将总交互数由冻结 Stage 35 模型的 262,144 增至 524,288；保持 `vf_coef=0.95`、`n_steps=2048`、batch 512、5 epochs、学习率 `1e-4`、gamma `.99`、GAE `.90`、观测/动作/reward/执行成本设置、Train 窗口和六个共同 seed 不变。候选预计 1,280 次 PPO update，对照 640。只使用既有 Train windows，不查询 Validation/Test/Holdout；每折一个 CPU worker，至少保留 1 GiB 可用内存。若训练步数加倍不能改善指标或出现过拟合迹象，继续维持 262,144 步基线。预登记 SHA-256：`25ebc66cf9ed72cdd5691b37791060506702eb57216862384b95d58cc2009745`；源码快照 SHA-256：`7cfe5dafa28d4369e089751a6061e31763299814a6cbfabc72946ae167c89a37`。六组候选与冻结 Stage 35 reference 逐字段核对，仅总交互步数不同；Ruff、py_compile 与参考模型完整性检查通过。六个候选完成 524,288 步，六个参考为 262,144 步；PPO updates 分别为 1,280 和 640。训练过程正常，所有模型 bundle/learning/sidecar 哈希通过，Validation 查询为 0。候选 EV 只在 1/6 配对提高，value loss ratio=1.869，动作幅度仅 2/6 不低，no-op 增加 0.0603；训练经济奖励虽然平均略少负，但仍为负，不能抵消其余退化。加倍训练明显过拟合/策略退化信号，gate 失败，维持 262,144 步。冻结的第一版汇总器错误假设 learner config hash 会随总步数改变；未改动预登记源文件，改用独立 `summarize_interaction_steps.py` 核验 12 个模型及更新数，source/preregistration 哈希仍匹配。结果目录 `.tmp/rl-ppo-vfcoef-095-steps512k-train-only-20261016/`。


## 第三十七项：扩展市场历史下的 PPO critic 权重稳健性 Train-only 诊断（已完成，2026-10-16）

Stage 35 的 `vf_coef=0.95` 在短历史窗口通过 Train-only gate，但 Stage 36 加倍步数导致显著退化。本项固定交互步数 262,144，扩展 Train 历史到 2026-09-02 与 2026-09-09 两个截止点，使用完整已审计 market snapshot；每折/seed 从头训练 `vf_coef=0.95` 与 `1.0` 配对模型，仅该参数不同，保持 Stage 35 的学习率、GAE、rollout、观测/动作/reward/风险成本等设置。只读取 Train 数据与原生训练诊断，严禁任何 Validation/Test/Holdout 查询（特别是已锁定的 2026-09-15 起外层区间）。预登记 SHA-256：`298d3ec3d87930a645c60c15ee125910beba2617714cc3edf83ca70fc9dd2f8e`；源码快照 SHA-256：`e2d107d3003c441c47dd5f14821995390f30db1eb482fa5861ab13873fca6e4a`。12 个模型全部完成，各自 262,144 步；两折 campaign 状态为 `complete`，Validation 查询为 0，外层 2026 数据查询为 0。

独立汇总验证了 12 份 bundle、R1 learning 产物、sidecar 哈希、配置哈希和单因素参数差异。配对结果为 explained variance 改善 3/6（两项历史截止点走势不同），候选/reference value-loss 均值比 1.0324，requested-action 幅度不低于对照 3/6，no-op 差值 -0.00018，候选 KL 均值 0.00543，clip fraction 均值 0.03081。经济训练奖励仍为负且候选均值略差（`-4.5471e-6` 对 `-4.5361e-6`）。按预登记 gate（EV 至少 4/6、value loss ratio ≤1.10、动作幅度至少 4/6、不增加 no-op 超过 0.02、KL ≤0.02、clip ≤0.10），因 EV 与动作幅度两项失败，`vf_coef=0.95` 不通过长期 Train-only gate，不作为默认配置；保持 `vf_coef=1.0`。

正式 `summarize.py` 在训练结束后因预登记 config 记录不含运行时 bundle/learning 哈希字段而触发 `KeyError`，不是模型或训练产物失败。未改动已冻结的注册/训练/汇总程序，另建 `summarize_independent.py` 从 campaign-state 中读取运行时哈希，对所有不可变产物执行独立验证并计算上述结果。独立结果在 `.tmp/rl-ppo-vf095-latest-train-only-20261016/analysis-summary-independent.json` 与 `paired-train-diagnostics-independent.csv`；原始输出与所有模型保留。

**阶段判断：** 不扩大该 critic 权重改动；Stage 35 短窗通过不足以证明跨区间稳健，长期实验未达到准入门槛。本项是训练诊断，不能证明策略有正收益，也没有运行交易回测或访问任何外层数据。

## 第三十八项：扩展历史下 PPO 折扣因子 Train-only 诊断（已完成，2026-10-16）

测试更长信用分配跨度是否改善分钟级 PPO 的训练诊断。候选 `gamma=0.995`，对照 `gamma=0.99`；使用相同 2024-01-01 起点、2026-09-02/09-09 两个 Train 截止点、seed 101/211/307、每模型 262,144 步及其他完全相同配方，仅训练 6 个新候选，对照复用 Stage37 已完成的 6 份 reference bundle。原预登记 SHA-256：`cababda5d2e81e387b74a65e50d6cd1a755ccdf26bc722f05210ed3cbdeacd83`；源码快照 SHA-256：`f6eb1aa4e5e621515e1429eac496a35725807e4045b49877822e961411d1617e`。恢复预登记 SHA-256：`f7a175e5f052eb4afff3d48137718e97c92cf45190290dcd184e560397ef6405`；恢复 runner 源快照 SHA-256：`2d2633169439c93639ed9dfd4a7d2bc80ef5d4cbb5473e60365b240aeae7fe1f`。恢复用于保存并续跑同一 gamma 候选，没有读取收益或验证数据。

独立汇总验证 12 份 bundle/R1 learning/sidecar 哈希、配置单因素差异和每模型 262,144 步；Validation 查询和外层 2026 查询均为 0。`gamma=0.995` 的 EV 高于对照 3/6，均值从 -0.04519 升至 -0.00150，但 value-loss 比值为 1.2394，超过 1.10 上限；requested-action 幅度不低于对照 5/6、no-op 差值 -0.00187、KL 0.00316、clip 0.01688。经济训练奖励均值则略降为 `-4.6007e-6`，对照 `-4.5361e-6`。因此候选未通过预登记 gate，保留 `gamma=0.99`。

恢复过程发现两类边界并均保持 fail-closed：第一次并行准备造成 Stage38 long-sep09 legacy quota lease；资源账本没有 Job recovery 证明，未手工删除。随后跨输出目录恢复 checkpoint 被应用拒绝（项目只接受同一 run 内的 R1 边界），因此将原 `run.json`、零 Validation 查询的 research ledger 与 16,384 步 checkpoint 复制到隔离输出目录，排除旧 quota 数据库和失败标记，并在恢复预登记中绑定复制 checkpoint 的 manifest/model 哈希；候选最终续训到 262,144 步。原始失败目录完整保留。独立摘要为 `.tmp/rl-ppo-gamma-0995-train-only-20261016-recovery/analysis-summary-independent.json`，逐种子配对诊断在同目录 CSV。

**阶段判断：** 更长折扣没有通过长期 Train-only gate，也未改善经济训练奖励，故不升级 gamma。下一项单因素测试 GAE `lambda=0.95` 对比基线 `0.90`，复用同一冻结 reference 及 Train 截止点；仍不触碰 Validation/Test/Holdout。训练诊断不是正收益证明。


## 第三十九项：扩展历史下 PPO GAE λ=.95 Train-only 诊断（已完成，2026-10-16）

预登记 SHA-256：`f966945c325b72b7ca6f6647ba99146ab80218660fff5a6512e6ef2e811aeb2a`；源码快照 SHA-256：`d6690084353fea4fef44b12159d59d17b5b474fc1a0388276c07d2ade213450b`；数据与情景规则沿用 Stage38。候选仅把 GAE λ 从 `.90` 改为 `.95`，其余参数、2026-09-02/09-09 Train 截止点、seed 101/211/307、262,144 步保持相同，复用 Stage37 冻结 reference。全部六个候选完成；验证与 outer 2026 查询均为 0。

独立预登记摘要核验 12 个模型的 bundle、R1 learning、sidecar、配置哈希和步数，所有模型均完成 262,144 步。候选 explained variance 仅 2/6 配对改善（均值从 -0.04519 降至 -0.07290）；value loss 比值 1.5448，超出 1.10 上限；requested-action 幅度不低于对照为 0/6（均值 0.13820 降至 0.12174）。no-op 变化为 -0.00393，KL=0.00464、clip fraction=0.02223；Train 经济奖励均值为 `-4.56875e-6`，对照 `-4.53610e-6`，两者都只是训练诊断且仍为负值。按预登记 gate，候选失败；保持 GAE λ=.90，不进入经济确认。

结果文件：`.tmp/rl-ppo-gae-095-train-only-20261016/analysis-summary.json`、`paired-train-diagnostics.csv` 与 `R1-block-diagnostics.csv`。训练按折串行进行，运行器每次检查物理可用和提交内存均保留至少 1 GiB；历史数据准备阶段峰值提交内存约 30.3/36.4 GiB、物理可用约 11.1 GiB，未触发资源下限。结果不能证明策略盈利，也未查询 Validation/Test/Holdout 或 2026-09-15 起外层数据。

**阶段判断：** 提高 λ 明显恶化 critic 与动作活动诊断，不能采纳。下一项改测 λ=.85，比较仍只触及该参数；若较短优势跨度也不能稳定改善诊断，暂停 GAE λ 的近邻搜索，回到输入信号/标签质量和交易规则可识别性问题，而不是无限扫超参数。


## 第四十项：扩展历史下 PPO GAE λ=.85 Train-only 诊断（已完成，2026-10-16）

预登记 SHA-256：`c957dbba4b2b8ee08f3ab1809786a9b471aba2790da43572f777c818340f21f5`；源码快照 SHA-256：`c7308109c6a36919295aba8845ba6c4edf42fc9e772ef603ca4e8cf0ba7c0ef6`。以 Stage37 的 `lambda=.90` 六个冻结模型作配对 reference，仅将候选 GAE λ 设为 `.85`；其余超参数、特征、动作/reward/执行成本、两个 Train 截止点、三个 seed 和每模型步数均保持相同。新候选均完成 262,144 步；摘要器核验 12 个 reference/candidate 模型和 bundle、learning、sidecar/config 哈希。Validation 与 outer-2026 查询为 0。

候选 EV 在 4/6 配对改善，均值从 -0.04519 提高至 -0.03427；平均 value loss 比值 0.8602；requested-action 幅度不低于 reference 为 5/6（均值 0.13820→0.14665）；no-op 差值 +0.00181，候选 KL=0.00607、clip fraction=0.03991。Train 经济 reward 均值仍为负，且候选 `-4.55233e-6` 略低于 reference `-4.53610e-6`。预登记 gate 要求经济 reward 不低于 reference，因此整体仍失败，不升级 λ、不做经济确认。KL/clip 与 Train reward 是优化器/训练诊断，不是交易回报证据。

结果位于 `.tmp/rl-ppo-gae-085-train-only-20261016/analysis-summary.json`、`paired-train-diagnostics.csv` 和 `R1-block-diagnostics.csv`。训练顺序为两个折各自串行，一个 learner；运行期间物理可用内存最低约 13 GiB，远高于 1 GiB 保留要求。对比 Stage39 的 `.95`，`.85` 改善了 EV、value loss 和动作支持，但没有改善训练经济 reward；该差异说明 λ 邻域对 critic/策略活动有影响，却仍未找到正收益信号。

**下一项：** 暂停 GAE 扫参，预登记做 Train 内部的输入信号可预测性审计：按因果时间戳计算现有特征对多个前瞻期限的方向、波动调整后收益及交易成本覆盖能力，采用时间顺序的内部滚动切分，并报告每特征/特征组的跨窗稳定性。该审计不训练、不查看 Validation/Test/Holdout 或 2026-09-15 起数据；只在识别出可复现、成本后仍存在且跨窗稳定的增量信号时，才把相关特征单变量接入新的 RL 训练实验。无论 audit 结果如何，都不得据此声称已找到盈利策略。


## 第四十一项：Train 内观测特征前向信号筛查（已完成，2026-10-16）

预登记的输入快照包括 expanded-market Parquet/manifest、scenario rules 与特征实现 SHA-256；计算窗口限于 2024-01-01 至 Stage40 的 2026-09-09 00:01:00.001 UTC Train 截止。20 个日历月块（2025-01 至 2026-08）逐月以之前 365 天作校准，前瞻标签最多 60 分钟，校准标签严格 purge 至测试月之前；每 5 分钟采样。筛查 legacy 市场字段（1m return、5m volatility/volume、basis、last settled funding）及 MTF return/volatility/trend-efficiency 候选，考察 5/15/60 分钟方向。未读取验证分区、outer 2026 或 cutoff 之后的数据。

汇总器核验了 720 条月块×期限×特征结果，36 个特征—期限组合均未达到“中位测试 IC 为正且至少 15/20 个月的校准方向 gross 为正”预登记筛选门槛。相对最好的 60 分钟单变量例子是 60m volatility：中位 OOS Spearman IC=0.0154，但只有 12/20 月 IC 为正；校准方向中位 gross=0.130 bp，正向月份 12/20、方向命中率中位 50.35%。5/15 分钟收益字段也没有跨块稳定方向。全部 2/10bp 比较都只是滑点 round-trip 底线，不含手续费、资金费、风险与实际执行，因此即使超过底线也不能解释为净盈利；本次中位 gross 仅 0.13bp 的量级，没有成本后正收益证据。

审计脚本和可复核结果位于 `.tmp/rl-feature-signal-audit-stage41-20261016/`；预登记 SHA-256（最终执行版）为 `2cf292bcc9d3cfc88560d48dc5948ac8a42cbba718d5239c7e650a5b609ca89a`，脚本 SHA-256 为 `8de9d53e55828bf4585705db9f1cef059cbab37f2f444d7f829b0b9fdad12f68`。启动前发生了依赖导入、PyArrow 列参数、Windows 路径序列化和 cutoff 可用时间边界问题；每次均在计算预测/信号统计前失败，原登记和修订链保留，最终修订绑定实际 2026-09-09 截止与完整输入哈希后才计算结果。独立核验检查了 20 个日期块、720 行及零 outer/Validation 查询。该单因子筛查只排除简单稳定方向信号，不能排除非线性交互、微观结构或风险状态价值。

**阶段判断：** 当前 observation 的单字段方向映射不足以支持策略收益；单纯再加一列相似 momentum/volatility 指标没有数据依据。下一项在相同 Train-only 滚动窗口中对冻结特征集比较线性 Ridge 与二次交互 Ridge，并对信号按 2/10bp 滑点底线做稀疏触发统计。只有在测试块稳定且毛边际覆盖更严格 hurdle 时，才预登记新的 RL 状态/动作实验；否则转查交易动作与成本频率、可用微观结构数据和非价格信息缺失。


## 第四十二项：Train 内多特征方向模型筛查（已完成，2026-10-16）

固定使用 Stage41 的 12 个因果 OHLCV、basis、last-settled-funding 与 MTF 特征，不再挑列。预登记 SHA-256：`499c77c6f461d413ed615249e5889d4ee7ac393135a67a2f586b37df36364c1f`；绑定 Stage41 脚本/登记 SHA、行情快照/manifest、规则文件和特征实现哈希。20 个 Train 内月块、前 365 日校准、5 分钟采样、5/15/60 分钟目标；标签 purge、规范化和 1/99% clipping 都只用各折历史校准数据。比较固定 alpha `.01` 的线性 Ridge 与固定 alpha `.10` 的二次 Ridge（12 个主效应+全部平方及两两交互）；不调参、不访问 Validation/Holdout 或 outer 2026。

独立核验 120 份模型×月×期限统计和 20 个日期块，所有 source/config/hash 通过，Validation/outer 查询为 0。六个模型—期限组合全部未达到门槛。二次 Ridge 最好的一项为 5 分钟目标：中位 IC=0.0038、IC 正向 13/20 月、方向 gross 中位 0.076bp、gross 正向 14/20 月；预测幅度触发 2bp 的中位频率仅 2.1%，触发后 realized gross 中位 0.508bp，仍远低于 2bp 且仅 12/20 月为正。二次 15 分钟模型在 2bp 触发后中位 gross 为 -0.448bp；60 分钟模型 10bp 触发后中位 gross 为 -5.946bp、正向月仅 9/20。线性模型各期限方向 IC/收益也不稳定。2/10bp 仅是滑点下限，结果不含显式手续费、资金费、回撤/保证金或账户执行效应，未证明净收益。

结果和复现脚本位于 `.tmp/rl-multifeature-ridge-audit-stage42-20261016/analysis-summary.json`、`monthly-walkforward-ridge.csv`、`ridge-summary.csv` 与 `audit_multifeature_train.py`。计算为单进程，物理可用内存始终远高于 1 GiB。此前所有失败都发生在数据/统计计算前，并通过版本化预登记和源码哈希记录；Stage41 与 Stage42 都是在内部 Train 时间窗的探索性滚动筛查，窗口反复参与研究不构成独立测试或泛化证明。

**阶段判断：** 现有分钟级价格/资金特征的单变量和简单二次方向边际都不足以覆盖滑点，继续调 GAE/critic 参数不会创造市场信息。下一项优先审查资金费事件与规则，预登记“仅根据已结算费率预测下一结算方向”的 carry 基准，并按实际资金现金流、交易费、base/stress 滑点与仓位变化核算；如果它也没有成本后优势，暂停当前数据集上的 RL 奖励/超参数搜索，优先引入有可验证来源的逐笔成交或最佳买卖报价/订单簿数据，再以新鲜 Train 时间段重开策略实验。任何内部 Train carry 结果仅用于筛查，不得当作独立收益证明。


## 第四十三项：Train 内滞后资金费 carry 基准（已完成，2026-10-16）

以当前 expanded-market-v2 行情、manifest 和 scenario-only 费用规则作为冻结输入；最终预登记 SHA-256 为 `fefa159ee42123218513d93b64e67031fe63ff4309ebc17d482d221d2815a845`，复算脚本 SHA-256 为 `a51cce70546cea87f37735df93439324e1123966d283e46e395857ddcd69ae08`，产物位于 `.tmp/rl-funding-carry-audit-stage43-20261016/`。范围只到 2026-09-09 00:01:00.001 UTC；核验 2,946 次结算，结算与同刻、同可用时间的 last-settled 更新逐一匹配，8 小时间隔允许最多 1 秒源时间戳抖动，并要求结算点有新鲜因果 mark。规则使用 $1,000 固定名义额、$10,000 记账资本；结算费率为正则转空、为负则转多，持有至费率符号变化。它仅是已结算费率的滞后 carry 规则，未预测未结算费率。费用 provenance 明确是 scenario-only，基础/压力成本分别采用 5/10 bp taker fee 和 1/5 bp turnover 滑点假设。

基础情景价格盈亏 **-611.53 USDT**、资金费现金流 **+252.48 USDT**、执行成本 **551.23 USDT**，净值 **-910.28 USDT（-9.10%）**；33 个月中 17 个月盈利，月末最大回撤 **17.01%**。压力情景净值 **-1,737.13 USDT（-17.37%）**，盈利月 16/33，月末最大回撤 20.70%。两个情景都未过预登记门槛。独立核对 CSV 月度净盈亏和摘要合计，差异小于 `1e-8` USDT，33 个月与盈利月数相符。费率方向延续命中率 84.45% 不能抵消持仓期间价格损失和转仓成本。Validation 与 outer 2026 查询均为 0；这是费用情景下的 Train 筛查，不是可交易收益证明。

**阶段判断及下一项：** 已结算资金费 carry 失败，分钟级价格/资金方向模型也未发现成本覆盖优势；当前 PPO 参数邻域搜索暂缓。Binance 官方说明 USDⓈ-M 历史 `bookTicker` 已停止提供，故下一阶段改用带官方 checksum 的 `aggTrades` 档案，而不把仅有的 2024-01-01 单日报价误作跨期样本。Stage 44 以 2024-01 至 2026-08 每月 15 日构造 12 个校准样本日与 20 个测试样本日，先扩展导入器为每秒主动买卖流量，再做固定历史校准—次月测试筛查；只用 Train，截止仍为 2026-09-09。若方向增量不能稳定覆盖费用 hurdle，不将其接入 RL；再检查成交流是否能预报波动状态，作为风险降仓信号，而不是方向交易理由。[Binance Public Data 说明](https://github.com/binance/binance-public-data)列出了 USDⓈ-M `aggTrades` 数据列与校验文件；其[官方仓库问题说明](https://github.com/binance/binance-public-data/issues/334)记录历史 Futures `bookTicker` 已停止提供。


## 第四十四项：Train 内逐秒主动成交流特征筛查（已完成，2026-10-16）

Binance 官方 `bookTicker` 历史在 2025 抽样日不可取得（HTTP 404），并且官方维护仓库确认不再提供历史 USDⓈ-M `bookTicker`。在计算任何策略结果前，将 Stage44 数据源修订为官方 checksum 校验的 USDⓈ-M `aggTrades`。数据导入器新增 `trade_flow_1s` 事件：按秒精确累加主动买/主动卖数量、名义额、成交数、价格范围和 trade ID，再应用登记的 1ms 派生可用延迟；这避免在内存中保留全量逐笔 Event。修复 curl TLS fallback 在 Windows 沙箱下错误使用系统临时目录的问题。32 个归档日均保留源档案、官方 hash、导入回执与 Parquet SHA；源清单 SHA-256 为 `c16b820b4200b25a0e68a8ec4348f060cd788c5ccecc5a668cf4ba7d1496b975`。

预登记含 2024-01 至 2024-12 每月 15 日的 12 个校准日，以及 2025-01 至 2026-08 每月 15 日的 20 个次月测试日，最终预登记 SHA-256 `b01637c805b968173173707e631db39cdc9d283d1a263d5dc8935f48ea4a1e27`。每个决策点每 5 分钟一次，前向期限为 1/5/15 分钟。比较 Ridge alpha=0.1 的冻结旧特征集与增加主动买卖量/名义额失衡、成交活跃度及 1/5/15 分钟滚动流量的模型；拟合、标准化和 1/99% clipping 只使用此前 12 个固定校准日。每个期限共 5,460 个 OOS 样本。成本门槛采用 scenario-only 的基础 12bp、压力 30bp round trip，不声称是已认证的历史费用或成交。

三个期限都未通过预登记 gate。成交流模型中位测试 IC 在 1/5/15 分钟为 **-0.0130/-0.0196/-0.0394**；旧基准为 **+0.0052/+0.0013/-0.0221**。成交流模型方向 gross 中位数分别 **-0.072/-0.191/+0.193 bp**，旧基准为 **-0.001/+0.106/-0.604 bp**；增广模型只在 **7/20、10/20、13/20** 个月胜过基准。12bp 基础成本下，1 分钟与 5 分钟 OOS 预测没有触发样本；15 分钟仅 **0.15%** 的 pooled 样本触发，触发后的 pooled gross 为 **-2.24bp**、base net **-14.24bp**，stress net **-32.24bp**。跨月模型表现弱于基准，当前不把成交流方向列接入 RL，也不继续围绕方向动作调 PPO。上述是固定稀疏样本日的筛查，不是完整日历月模拟或独立盈利验证。

证据位于 `.tmp/rl-quote-microstructure-stage44-20261016/`：`preregistration.json`、`source-manifest.json`、`analysis-summary.json`、`monthly-trade-flow-screen.csv`、`oos-predictions.csv`、`acquire_quote_days.py` 和 `analyze_trade_flow_train.py`。汇总摘要 SHA-256 `8a712125039d005720f0196b635cde944577f6e31d1c685e9de9381aba694734`；分析脚本 SHA-256 `7078cd254050de73cf7e3ecda8582f13835f9fa211022cf92b56d7e6720dad62`。独立复核重新从逐样本预测计算了 pooled Spearman/gross、20 月胜出数和成本触发净值，均与摘要一致；Calibration/test 日期、源/模型哈希及 0 次 Validation/outer 查询也已核验。aggTrades parser 的聚合、乱序拒绝、TLS 重试与回退相关 5 项定向测试通过；Ruff 与 py_compile 通过。pytest 默认用户 Temp 目录和部分 Parquet 原子写测试受到 Windows 沙箱 hard-link/路径权限限制，故未宣称完整包测试通过。

**下一项：** 不根据 Stage44 的测试结果改造方向奖励。以相同冻结输入新登记 Stage45：比较旧特征与旧特征+成交流对未来 5/15/60 分钟实现波动率的滚动预测；以训练样本的波动率第 90 百分位建立风险状态门槛，检验预测校准、尾部召回与跨期稳定性。只在不读取 holdout 的前提下发现可复现风险预测价值时，才另开 PPO 风险降仓对照。波动预测改善仍不是盈利证据，最终仍须通过独立验证与逐成本交易回放。


## 第四十五项：Train 内未来实现波动率与高风险状态预测（已完成，2026-10-16）

Stage45 复用 Stage44 的 32 个 checksum-bound 日期和同一滚动校准/测试窗口，不触碰 Validation、Holdout 或 2026-09-15 之后数据。以未来 5/15/60 分钟一分钟 close-to-close 对数收益的 RMS（bp）为目标，拟合 `log(RV+1e-6)`，比较 Ridge alpha=0.1 的 legacy 12 特征与同基线加 12 个逐秒成交流特征。校准数据只用测试月前 12 个固定月中日；每 5 分钟决策，15 分钟清洗期和完整 60 分钟目标尾部均按因果边界剔除。每折风险标签阈值是校准期 RV 第 90 百分位。Stage45 预登记 SHA-256 `d088adbbea91d884c25c5a64bf34c4d170f05441f14f1bdfbe68a1fa170dd323`，分析脚本 SHA-256 `688ff4f92bc7a159a310b4649b19daed51e552d4748f6ced1b61ac5c555fa671`。

15 分钟目标出现可复核的总体波动排序增益：增广模型月度中位 Spearman **0.5052** 对基准 **0.4797**，月度中位 MAE **1.964 bp** 对 **2.093 bp**，相对 MAE 改善 **6.14%**，且 **15/20** 月份误差更低。5 分钟 MAE 改善 **3.33%**（17/20 月份较低），低于 5% gate；60 分钟 MAE 恶化 **1.70%**，仅 14/20 月份改善。方向标签和收益未用于这一阶段。

不过 q90 高风险识别并未改善：15 分钟增广模型中位召回 **7.14%**，基准为 **9.76%**；中位 precision 也从 **26.50%** 降为 **13.64%**。5 分钟召回从 4.26% 增至 10.27%，但 MAE 改善率未过 gate；60 分钟两模型中位 precision/recall 都为 0。所有期限均未通过预登记 gate；仅有总体排序改善不足以作为仓位缩放器。Stage45 不是收益回测，也未开始新的 RL 训练。

证据位于 `.tmp/rl-volatility-risk-state-stage45-20261016/` 的 `analysis-summary.json`、`monthly-volatility-risk-screen.csv`、`oos-volatility-predictions.csv`、`preregistration.json` 和 `audit_volatility_regime.py`。独立复核重算了 120 行月度结果、32 个样本日、32,400 条 OOS 预测的 pooled IC/MAE 与每期召回，均与摘要一致；验证/outer 查询为 0。Ruff 与 py_compile 通过。该滚动窗口已反复用于 Train 内探索，不能视为独立泛化证据。

**下一项：** 检查上述 15 分钟波动排序增益能否支持一个冻结的连续风险缩放方案。只用 Stage45 的逐月 OOS 风险预测和预先固定的方向信号基准，比较固定名义额与波动目标仓位的成本后盈亏、回撤和风险降低/收益牺牲；同时保留 scenario-only 手续费与滑点标签。该项若只降波动但进一步降低净值，不将风险门接入 RL；通过也只允许另行登记的训练实验。


## 第四十六项：OOS 波动预测驱动的连续仓位缩放（已完成，2026-10-16）

将 Stage45 已冻结的 15 分钟波动率逐月 OOS 预测直接接入固定诊断策略，不重新拟合任何测试月模型。共享因果方向信号为过去 15 分钟 close return 的符号，决策和未来收益区间均为不重叠的 15 分钟；每个测试样本日单独从 $10,000 起始资本和空仓开始，$1,000 目标名义额，日终强制平仓。风险仓位为 `clip(train_q75_RV / forecast_RV, 0.25, 1.0)`；q75 只由该测试日前 12 个登记校准日估计。对比 full exposure、legacy-risk forecast 与 trade-flow-risk forecast，按基础 12bp 与压力 30bp round trip 情景成本核算。

预登记 SHA-256 `cfb2271d83047e8caf1162b676d855f3383666b6224a977c2d51dc3352827579`，模拟脚本 SHA-256 `be2fd9168a1cde03353501ab8e1c07500209efbd6192bb05f71a51075c1a5831`。三个 sizing 策略在两种成本场景的 **20/20** 样本日都亏损。Base 情景平均日净盈亏：full exposure **-65.45 USDT**、legacy scaler **-63.67**、trade-flow scaler **-63.97**；压力情景分别 **-153.20/-149.71/-149.59 USDT**。增广缩放的平均仓位权重仍为 95.8%，base 中位日内最大回撤从 full 的 0.736% 仅降至 0.721%，stress 从 1.557% 降至 1.546%；净值仍明显为负，没有一条策略达到 15/20 盈利日。两个 scenario gate 均失败，不把该 risk scaler 接入 RL。成本后负值也表明这项 15m 趋势控制信号本身没有正边际；波动预报能排序风险，不等于有方向优势。

结果在 `.tmp/rl-continuous-risk-scaling-stage46-20261016/analysis-summary.json` 与 `monthly-risk-sizing-results.csv`。独立按 120 行逐月 CSV 重算平均净 PnL、盈利样本日和最大回撤，与摘要一致；外层/验证查询为 0。Ruff 与 py_compile 通过。该结果是 20 个固定月度样本日的稀疏 Train 筛查，不能拼成连续账户曲线，也不等同于 RL policy 回放。

**下一项：** 当前单资产方向特征、逐秒成交流方向与连续风险缩放均没有生成可盈利净值。先扩充同一固定样本日上的可验证市场信息：按月中固定日导入 Binance USDⓈ-M BTCUSDT 日线级 OHLCV、mark/index 与 funding，使用 ETH 与 BTC 的相对收益、领先收益、滚动相关、波动及 basis/funding 状态做严格的先 12 日校准—后 20 月测试。比较 ETH legacy、加 BTC context、再加已锁定 trade-flow 特征的固定 Ridge 方向模型；只有若在更严格成本 hurdle 下持续增量才开 RL 状态实验。仍不得读取验证/外层数据；正收益最终必须独立复现并通过逐成交成本和稳健性验证。


## 第四十七项：Train 内 BTC 跨资产上下文筛查（已完成，2026-10-16）

按 Stage44 固定的 12 个校准日与 20 个 2025-01 至 2026-08 测试日，导入 BTCUSDT 官方 checksum 验证的 USDⓈ-M 1 分钟 OHLCV、mark/index 与 funding；共 32 日，每日约 4,410–4,412 条事件。导入期间最低可用物理内存 **21.03 GiB**，高于至少预留 1 GiB 的约束，单进程串行完成。输入只到 2026-08-15，未查询 Validation、Holdout 或 outer 2026。预登记 SHA-256 `61f6f18535234336b9c63311359101edb2f3ebbe727a335059433cba2f5671ee`，BTC 源清单 SHA-256 `5d895ba8475b82bb59de8424cc639be1e2d94ae6f2d5c30ae8be11b9c4c71fad`，分析脚本 SHA-256 `8b816aacf9e03fa3dcf96837b3902051f3f86beef63184e79f2d33b764369685`。

在固定 ETH legacy 12 特征基础上加入 BTC 1/5/15/60 分钟收益、波动、趋势效率、basis/funding、ETH-BTC 收益差及过去 60 分钟相关；第三个模型再加入 Stage44 已冻结的成交流特征。固定 Ridge alpha=0.1，每 5 分钟做 5/15/60 分钟 ETH 收益预测；标准化与 clipping 只拟合前 12 个样本日。20 个测试月每模型每期限均保留逐样本结果，共 60 行月度结果和 **48,600** 条 OOS 预测。交易方向按预测符号计算，12/30bp 往返成本仍是情景假设，不代表历史真实成交。

BTC 增量及 BTC+成交流增量在三个期限的预登记 gate **全部失败**。单加 BTC 的中位 OOS IC/方向 gross：5m **-0.0253 / +0.046bp**，15m **-0.0576 / -0.936bp**，60m **-0.0790 / +0.273bp**；对应 legacy 中位 IC/gross 分别为 **+0.0013/+0.106bp**、**-0.0221/-0.604bp**、**-0.0362/-0.832bp**。BTC 模型胜过 legacy gross 的月份数分别为 **10/20、5/20、8/20**，12bp 成本后净值为正的月份数为 **0、0、4**。BTC+成交流模型的 gross 胜出数分别 **10/20、4/20、6/20**，基础成本后正收益月份数 **0、1、5**。任何期限均未同时提高中位 IC 和 gross，也未达到 15/20 月稳定门槛，因此不将 BTC 或其组合特征接入 RL 状态。

独立 verifier 从 OOS 逐样本预测重新计算 60 行各模型/期限指标、月份胜出数和 gates，**48,600** 条记录全部核对通过；Validation 与 outer 2026 查询数均为 0。Ruff、py_compile 通过。完整证据位于 `.tmp/rl-cross-asset-btc-stage47-20261016/`：预登记、源清单、分析摘要、月度 CSV、OOS 预测 CSV、导入器、分析脚本和 `verify_btc_context_screen.py`。摘要 SHA-256 `4af1324b9951bb1507798d0fb64a2e92ad8d6cccb489d7c7f462697a3e370b5a`；独立核对程序 SHA-256 `adc4c22bd557e793455cebf92e01ddca890a386018624caf8e9dff7ba83d9db6`。该固定月中日样本是稀疏信号诊断，不是完整连续回测或策略盈利证明。

**下一项：** 既有线性预测筛查没有发现可覆盖成本的方向增量。Stage48 在相同冻结样本上检验简单、可解释且严格 train-only 的条件方向策略：只预登记少数动量/反转规则和波动/basis 状态分组，阈值仅由每折此前 12 个校准日拟合；比较未条件化与条件化方向的月度毛/净结果和成本后覆盖率，并保留不交易基线。不得根据测试月挑参数或扩大候选集；如果仍无稳定正净收益，继续转向数据时间分辨率、市场状态标签和动作可达性诊断，不把较差输入硬塞给 PPO。


## 第四十八项：Train 内条件动量/反转与市场状态规则筛查（已完成，2026-10-16）

在 Stage44 的同一组 checksum-bound ETH 样本上，每个期限分别预登记 20 个候选：动量/反转、全信号/q75 强信号门槛，以及全样本、低/高波动、正/负 mark-index basis 状态。持仓期限分别为 5/15/60 分钟，采样间隔与持仓期相等，使相邻收益区间不重叠。每个测试月只用此前 12 个固定校准日选择训练净收益最高且至少 30 笔、基础成本后净收益为正的候选；没有符合条件的候选时选择空仓。测试期不调门槛。预登记 SHA-256 `c1a8511968ef90c3d943840aac6a43108042678cc4a4a8f5d5d1080578821d5f`，源清单 SHA-256 `3c64ffc8e563c7c18a8d2f1cbf576f32611526dd9f192befc2085c8790f9711b`，分析脚本 SHA-256 `19f3b88fc8c148b4bb9543cceda9fab967522989fd3114a8587188fc8ea022f2`。成本仍为 scenario-only 的基础 12bp 与压力 30bp round trip。

滚动选择器在 5 分钟期限 **20/20 月选择空仓**；15 分钟中位交易覆盖率为 0，均值 base net 为 **-0.094bp/决策**，没有基础成本盈利月份；60 分钟中位覆盖率 **4.55%**，base/stress 平均分别 **-1.82/-3.70bp/决策**，只有 **5/20** 与 **2/20** 月份净值为正。三个期限都不满足 15/20 base-positive 且 12/20 stress-positive 的 gate。固定动量策略在 5/15/60 分钟的平均 base net 为 **-11.89/-12.75/-8.55bp/决策**；固定反转分别为 **-12.02/-11.22/-15.45bp/决策**。每期限全样本方向规则持续付出成本，但训练期选型倾向于减少或停止交易；没有正收益规则可交给 RL。

独立核对器从 **31,040** 条 OOS 决策记录重新计算 60 行月度结果、base/stress 盈亏、覆盖和 gate，全部一致；Validation/outer 2026 查询数为 0。Ruff、py_compile 通过。证据位于 `.tmp/rl-conditional-direction-stage48-20261016/`：预登记、源清单、摘要、月度 CSV、训练选择 CSV、OOS 决策 CSV、分析与独立核对脚本。摘要 SHA-256 `2ebbf36a69cc2c3ced7c0fe2bd7f2bc242627bff33e279748f8bde75f7bcf332`；独立核对脚本 SHA-256 `5a3269e7a16cbb34903988721d75667311a6509f3db425029d94c7973104a88f`。Stage48 是固定月中日稀疏筛查，不能当作连续回测或盈利保证。

**下一项：** Stage48 的校准器在不确定时空仓，但 2024–2026 稀疏样本不足以判断更长市场周期。Stage49 扩展到连续、可校验的 Train-only 历史分钟行情，先做数据覆盖/时间戳质量检查，再预登记跨时期 walk-forward 的简单低频趋势与反转候选；训练、校准与 OOS 按时间严格分离，按真实可复现的成本情景报告。只有稳定超过成本后再回到 RL 连续动作、奖励和 PPO 超参实验；不得以增加迭代次数替代有效市场信号。


## 第四十九项：连续 Train 分钟历史与 56 月 walk-forward 条件方向筛查（已完成，2026-10-16）

为弥补 Stage44–48 每月单日样本的稀疏性，使用 Binance 官方 USDⓈ-M 月度 ETHUSDT `1m` kline、mark-price kline、index-price kline 与 fundingRate 归档，从 2020-01 到 2026-08 共 **80 个月、320 个 ZIP**，每个 ZIP 均按官方 `.CHECKSUM` 文件验证 SHA-256。Binance 官方[公开数据仓库说明](https://github.com/binance/binance-public-data)列出 daily/monthly 归档、USDⓈ-M kline 数据列和 checksum 校验方式。将 2020–2021 作为历史 warm-up，之后从 2022-01 至 2026-08 做 **56 个月**逐月 OOS；每次使用前 24 个月校准，测试持有期限 15/60/240 分钟，并按相同间隔决策以消除样本收益重叠。只使用截至 2026-08-31 的数据，零 Validation/outer 查询。

源归档总计约 **318 MB**。数据审计得到 **3,506,400** 根连续 1 分钟 ETH kline 和 7,305 条 funding 记录；OHLCV 时间戳连续、价格为正。部分早期 mark/index 归档存在缺失分钟（最大 gap 1,441 分钟）。在尚未计算策略结果前，将 Stage49 预登记修订为 revision 2：按决策时间向前匹配最新已收盘 mark/index，并要求最旧状态不超过 120 秒，过期样本剔除；共排除 5,806 个决策点。funding 使用事件时间因果对齐，持仓期间实际 funding rate 现金流计入方向收益。预登记随后增加 revision 3 仅报告性的风险摘要，不更改候选/选择规则和 gate。最终预登记 SHA-256 `f45827787e979208f6493b2c59dba26392ebaf87fd7de3c80ea1bd219a5fa1b2`；来源清单 SHA-256 `db95f7e1abd4fe8bcc0894c05f434c8b64bbef062a1a890c506952b7755618e5`。导入期间最低可用内存约 **20.07 GiB**，高于 1 GiB 保留线。

固定候选集每期限 20 项：动量/反转 × 全部/q75 信号强度，再分别按全状态、低/高实现波动及正/负 basis 筛选。阈值只从此前 24 个月估计；训练期至少 100 笔且基础成本净收益为正时才允许选型，否则空仓。基础/压力成本为 12/30bp round trip，仅为可重算情景假设。三个期限全未通过预登记 gate。15m、60m 选择器在 **56/56 月全为空仓**；240m 仅 **9/56** 月 base 为正、**7/56** 月 stress 为正，累计净值分别 **-3,930.99bp** 与 **-14,136.99bp**。2024 的 4h selected 结果略正（base **+419.45bp**、stress **+23.45bp**），但 2022、2023、2025 等年度亏损，2026 的 base 正值也在 stress 下转负，因此不满足年度稳健 gate。按 $10,000 起始资本、每次用当前权益的 10% 做无杠杆、顺序复利的描述性诊断，4h 选型器最终 equity base/stress 为 **$9,606.87/$8,674.62**，最大回撤 **7.67%/13.41%**；该账户假设只作风险描述，不代替成交模拟。固定动量与反转策略回撤更大，成本后明显亏损。没有发现能接入 RL 的稳定正收益规则。

完整证据在 `.tmp/rl-continuous-train-data-stage49-20261016/`：80 月源归档、320 项校验清单、`preregistration.json`、`source-manifest.json`、`analysis-summary.json`、168 行月度结果、56 月训练选型、856,188 条逐决策 OOS 记录及导入/分析/核对程序。摘要 SHA-256 `52d0e9a078b214869fa0f71014ae3b8b79979292d97d978e39aaf49d1adac1c2`；分析脚本 SHA-256 `b7c1657af5778fb35bf45e6ca8510cfe071b044e82a1dfea4140abd6205fd742`。独立复核从逐决策数据重算交易数、成本后净值、月/年合计、复利 equity 与最大回撤，**168 行月度数据和 856,188 条 OOS 决策全部通过**；Ruff 和 py_compile 通过。早期数据的 120 秒 basis 新鲜度处理已在任何策略计算前登记，所有结果均限于 Train 区间。

**下一项：** Stage49 显示极低频信号偶尔可在单年覆盖情景成本，但滚动选择器仍然空仓或亏损。Stage50 扩展持有期到 4/12/24 小时，测试固定方向预测是否能通过多尺度趋势、均值回归、funding carry 和连续仓位风险预算捕获更慢的市场结构；继续使用连续 Train 数据与逐月 24 月滚动校准，预先锁定模型/阈值网格，不在 OOS 月调参。同步评估现有 RL 的执行成本、动作 deadband 与仓位缩放机制是否让可学习信号被压制。任何一项过 gate 前不启动昂贵 PPO 网格训练；过 gate 后仍需隔离的 Validation 与外层真实成本验证。


## 第五十项：4/12/24 小时多尺度 Ridge 方向模型与波动仓位缩放（已完成，2026-10-16）

复用 Stage49 80 个月 checksum-bound 连续数据和 56 个月 OOS 样本，不增加归档读取。预测期限为 4/12/24 小时，决策步长与持仓期相同；固定特征集比较价格/波动/趋势特征与增加 basis、funding、taker imbalance 的上下文模型，Ridge alpha 为 0.01/0.1/1/10。每月严格以前 18 个月拟合、后 6 个月内校准；校准期预测绝对值 q75 决定开仓门槛，候选只按 base 净收益、至少 20 笔且均值净收益为正进行选择；未满足时空仓。再比较全额与按当前实现波动缩放的仓位，缩放权重限制为 0.25–1.0。成本 12/30bp 为 scenario-only，funding 现金流计入。

Stage50 预登记 SHA-256 `07346cc9d40420eacce27cc2d323781f1d57d09795cd8534c7d5ce830e0c1d08`，源身份 SHA-256 `5ba90791bf702a8e8fd572102e2ac63d311082e0aa32a060fbe4926efaf73b5c`，分析脚本 SHA-256 `7298ba8852a8edfd91ee0fcfeb3c53e07bf41bd0f4be0071226b84d1098b87e3`。三期限 gates 全失败。4 小时 selected-full pooled base net **+2,709.73bp**，但正 base 月只有 **13/56**，stress pooled net **-16,784.27bp**、正月 **6/56**；年度净值只有 2022 同时正，2023–25 亏损，2026 stress 也为负。12 小时 base/stress pooled **-18,488.07/-28,280.07bp**，24 小时 **-6,726.94/-14,016.94bp**。4 小时按 10% equity 仓位复利的描述性结果，full 结束权益为 **$10,258.15/$8,441.16**（base/stress），最大回撤 **6.41%/17.58%**；波动缩放结束权益 **$10,072.92/$8,493.98**，最大回撤 **5.85%/16.52%**。风险缩放在 stress 下稍微减少亏损和回撤，但没有增加盈利月份或满足 gate。无可晋升给 PPO 的稳定方向模型。

证据位于 `.tmp/rl-multiscale-linear-stage50-20261016/`：预登记、Stage49 来源绑定、analysis summary、月度结果、选型记录、45,438 条逐决策预测及独立 verifier。最终摘要 SHA-256 `a4c5f88f380efa9fc14a44977a0fa9b8d3f06894b1aa1e177e6de72e28efde11`；独立 verifier SHA-256 `df484883ac283476ca5ec3c1413abaa03e71deb7bc87dadc760a293c3a1c4312`。逐月成本归因、年汇总、仓位复利/回撤以及 168 行月度、45,438 条 OOS 记录复核一致；Ruff 和 py_compile 通过，Validation/outer 查询为 0。

**下一项：** Stage50 的 4 小时模型显示少数月份有较大 base 盈利，却不能覆盖更高 stress 成本，说明信号置信度与交易成本门槛可能是主要瓶颈。Stage51 固定相同训练/OOS 切分与模型特征，预登记 q75/q90/q95 三个预测强度阈值，并只在内校准期按 stress 净值挑阈值/模型；按 full 与波动缩放两个动作幅度输出 OOS 结果。如果高置信度候选仍无法跨期覆盖 30bp 往返成本，停止扩展方向阈值网格，转为诊断预测符号、执行成本及市场状态漂移。


## 第五十一项：用内校准 stress 净值选择置信度阈值（已完成，2026-10-16）

沿用 Stage50 相同数据、18 个月拟合 + 6 个月内校准 + 次月测试的 56 月 walk-forward；模型仍为两套特征与 Ridge alpha=0.01/0.1/1/10。唯一新增因素为固定预测绝对值 q75/q90/q95 阈值，内校准选择规则改为最高 stress-net/决策，且至少 20 笔、stress net 为正才开仓；否则空仓。比较 full 与已登记的波动缩放仓位。这样直接把 30bp 往返成本纳入挑选，但该选择仍只用过去内校准月份。

Stage51 预登记 SHA-256 `155dc8054cce11f50a69d90e57f2073d54ffa44ba0c53c610233bea95328c6d8`，来源清单 SHA-256 `11944ab448ed8ea2a04ac27fe34824ed7f912447d73fc9e5a4775bd80c2fdee4`，分析脚本 SHA-256 `bb4b3c0ee8e071e977a8468e5b8240e3fe1532378caea88aea4b7c2da0d25587`。三期限全部未通过稳定盈利 gate。4 小时 selected-full 在 **9/56** 月 base 正、**7/56** 月 stress 正，pooled base/stress 为 **-1,692.17/-7,416.17bp**；12 小时为 **13/56、11/56** 月正，pooled **-5,487.75/-11,247.75bp**；24 小时 **17/56、11/56** 月正，pooled **-7,952.82/-13,784.82bp**。Vol scaling 均未提高正收益月数。选择器 4h 多数月份选择 q95（23 月空仓），12h 多数选 q90（32 月）、24h 多数选 q75（39 月）；跨期限阈值偏好不稳定。Stage51 的 OOS 成绩还低于 Stage50，显示即使以内校准 stress 收益选型，也不能抵消小样本模型选择噪声与阶段性漂移。

结果证据位于 `.tmp/rl-stress-threshold-stage51-20261016/`：预登记、source manifest、摘要、月度结果、模型/阈值选择表及 45,438 条 OOS 决策。摘要 SHA-256 `c34a617a79f869699f00c10c5b76879dd6e006731628e321adf545ac7485118a`；独立 verifier SHA-256 `df484883ac283476ca5ec3c1413abaa03e71deb7bc87dadc760a293c3a1c4312`。三期限每月收益/成本、权益曲线和回撤独立复算通过；168 行月度汇总、168 个选择记录、45,438 个 OOS 决策一致。Ruff 和 py_compile 通过；没有验证集、outer 2026 或其他 cutoff 后数据查询。

**下一项：** Stage51 未找出能稳住 30bp 压力成本的信号，继续扫更多阈值会加剧复用 OOS 的研究过拟合。Stage52 先冻结一个机制诊断：按每年、实现波动、basis、funding 与预测符号分解 Ridge 的信号命中率/可交易毛利和成本损益，量化特征系数漂移及不同市场时期是否反向；不再按该同一 OOS 选策略。如果存在可重复、预先可描述的单一状态机制，再以全新的后续训练区间做确认；如果没有，则暂停用该样本做方向调参，转向补充独立数据或修复执行成本假设，随后再进行 PPO 动作/奖励实验。


## 第五十二项：已有 OOS 信号的年份/波动/basis/funding 条件诊断（已完成，2026-10-16）

本项不训练模型、不选择策略、不改 Stage50 OOS 行为，仅将已冻结的 4/12/24 小时预测/交易记录映射回连续 Train 源数据，按年份、内校准期 q75 实现波动状态、正/负 basis、正/负 last-settled funding 做描述性分解；同时按原先选中的特征集/alpha 在既有 18 个月拟合窗口重新取标准化 Ridge 系数，检查符号是否稳定。预登记 revision 2 只修正状态分组校准日期边界，未改变任何 OOS 信号。

4 小时 selected-full 的全部 1,083 笔交易 base pooled net **+2,709.73bp**，压力 pooled net **-16,784.27bp**。高波动子组 485 笔，base **+2,643.76bp**，stress **-6,086.24bp**；低波动子组 598 笔，base **+65.97bp**，stress **-10,698.03bp**。负 basis 子组 997 笔，base **+2,217.97bp**，stress **-15,728.03bp**。可见少数条件对基础情景毛利贡献集中，但所有状态切片仍无法覆盖压力成本，且这些状态是在反复使用的同一组 OOS 月份上回看，不能当作新策略证据。12/24 小时在各主要切片仍呈现负 pooled 成本后结果。

4 小时 market-context Ridge 的动量系数显示部分 sign 稳定：60/240/1,440 分钟过去收益系数在 **12/12** 个被选月份均为负；但 funding 仅 **4/12** 月为正，basis **2/12** 月为正，240 分钟 flow **1/12** 月为正。系数符号的部分稳定未转化成能覆盖 stress 成本的 OOS 收益，不能据此直接设定奖励或高波动动作规则。

Stage52 预登记 SHA-256 `bc0b4d3bbc0f26dff7bf9b1b373ad24a2aa84de41ded71903e6646ba51a2428f`，source manifest SHA-256 `4d8a84807e00030e02574559fcae4203cd8827cc4938e77548760f2c06a7ec19`，分析脚本 SHA-256 `ecf0b1dbc3cafe98d0590c50e7e2b7efa5a680fab4724f4cf9da630ce76a216f`。产物含 1,237 行月度分组、36 行聚合和 1,646 项逐月模型系数。独立核对器复算了每个分组的 pooled/月度 net、交易数和系数符号摘要，全部一致；Ruff/py_compile 通过，未查询 Validation 或 cutoff 后数据。证据位于 `.tmp/rl-mechanism-diagnostics-stage52-20261016/`。

**下一项：** Stage52 没有发现任何能稳定覆盖压力成本的已验证市场状态。为了测试此前稀疏 Stage47 BTC 信息是否因抽样而漏检，Stage53 扩展到 2020–2026 连续 BTC 1 分钟数据，完全复用 Stage49 ETH 序列与相同时间切分；预登记跨资产收益、波动、ETH-BTC 相对收益和滚动相关特征，只做固定多尺度模型增量比较。该阶段不依据条件回看结果选择仓位规则；如果 BTC 增量仍为负，就停止跨资产方向特征扩张，转向 execution/fill 成本与模型校准误差诊断。


## 第五十三项：连续 BTC 跨资产特征增量检验（已完成，2026-10-16）

从 Binance 官方 USD-M 月归档取得 2020-01 至 2026-08 的 BTC 1 分钟 K 线、mark/index 与 funding，共 320 个文件、333,546,870 字节；归档和官方 SHA-256 checksum 全部复核。与 Stage49 连续 ETH 训练集严格按分钟对齐，截止 `2026-09-09T00:01:00.001Z`，只覆盖 Train。56 个 OOS 月（2022-01 至 2026-08）沿用 18 个月拟合、6 个月校准和次月测试；新加标签边界 purge，确保训练/校准的收益及 Funding 区间没有跨入下个时间分区。固定 Ridge alpha=0.1，比较 ETH price-only、ETH context 和 ETH+BTC 多尺度收益/波动/趋势效率、basis/funding、taker flow、ETH-BTC 相对收益和 60 分钟相关；阈值是过去校准期预测绝对值 q75，不在测试月择模型。

BTC 增量没有通过任何可提名后续 RL 状态实验的门槛。月度中位 Spearman IC（ETH context → ETH+BTC）在 4h 为 **0.0279 → 0.0260**，12h 为 **-0.0072 → 0.0187**，24h 为 **0.0546 → -0.0082**，只有 1/3 个期限改善，未达到至少 2/3。BTC 模型的阈值全仓毛收益超过 ETH context 的月份分别为 **23/56、24/56、28/56**，均少于 34/56。BTC 模型成本后 pooled base/stress 净收益在 4h 为 **-30,904.94/-82,276.94 bp**，12h 为 **-14,904.67/-30,438.67 bp**，24h 为 **-1,584.84/-9,306.84 bp**，没有一个期限在两档情景成本下同时为正。全仓毛收益摊到成交分别约 **+1.17、-5.27、+8.31 bp/笔**，低于 12bp base 往返成本或本身已为负，说明当前这组方向预测无法覆盖执行成本。波动缩放与基准比较仍保留在逐决策输出中，不据其另行挑选策略。

独立复核重新对 320 份 BTC 归档及 checksum 做哈希校验，并从 **90,486** 条逐决策记录重新汇总 168 个期限/月报表、1,008 个策略/仓位/月份组、504 个 OOS IC，逐行净收益误差为 **0 bp**；预登记提名门槛独立重算一致，三个条件全未达成。预登记 SHA-256 `093212e426cdf03ba00b2490433a2b6ec75d23e456f776e85e00f0f26c40a312`，来源清单 SHA-256 `b261ea2fff200c382056e64fe8784655925234a172a6c4f4d48533622871956f`，摘要 SHA-256 `2ea22ec02c79fb14fbb475aacd587812a0347a0d1994104fc23d38adb5fdd434`，分析脚本 SHA-256 `a112509cd0f680ce254747fe4b79d250f69fc0840176e6793e46b05fc3facb3f`，独立验证器 SHA-256 `22e0f9163f929f46dd25b85aa0cb87be630ca77f82778ee2b387ace5ed4263ba`。Ruff、py_compile 和独立复核通过。所有时期均曾用于先前 Train 内探索，这不是独立验证；成本仅为情景假设，亦不包含撮合级成交模拟。

证据位于 `.tmp/rl-continuous-crossasset-stage53-20261016/`，包括预登记、320 项源清单与归档、滚动阈值、月度屏幕、90,486 项 OOS 决策、摘要、独立验证报告及复跑脚本。

**下一项：** 停止继续扩张 BTC 方向特征或回扫 q 阈值。Stage54 对现有 RL 执行路径做成本与奖励尺度一致性审计：从策略动作到名义仓位、成交费用/Funding、逐步净值变化、奖励及训练日志建立逐笔守恒关系；用确定性小型行情用例验证开/平仓、反手、持仓和终局结算。若发现影响可学习信号的实现偏差，先修复并加有意义的回归测试，再用固定种子和同一 Train 切分对照；不触碰 Validation/Holdout，也不把成本敏感性筛选包装成盈利证明。


## 第五十四项：native 账户成本到 RL reward 守恒核验（已完成，2026-10-16）

审查连续动作经 SoftRisk/HardRisk 与名义仓位规划后进入 Nautilus native engine 的路径；账户 snapshot 的 authoritative equity 包括 native fill、commission、Funding 和未实现损益，Gym 每一步以 `log(equity_t)-log(equity_{t-1})` 作为原生经济 reward，固定 reward scale 只改变 learner 输入尺度。终局先按 native engine 强平并结算，再计算最后一次 reward。未发现把费用/Funding 重复扣除或让终局强平免成本的实现差异；成本仍取决于场景配置，不能推出实际历史账户收益。

增加真实 ETHUSDT golden 回归：478 个账户观测中逐步 `r_econ` 累加值必须与最终 native equity/初始 equity 的 log 比例相等，并逐步核对 Env 返回 reward 与 `r_train` 一致。Golden 的最终账户仍为 **9998.21338803 USDT**，native 报告费用 **0.49911263**、滑点成本 **0.10028**、Funding **-0.15498934**，以实际结算值验证 reward 守恒。执行专项测试时先因默认 Windows Temp 沙箱权限报错；改用工作区内隔离 basetemp 后，RewardCalculator、native episode 和真实行情 golden 共 **38 passed**。新增守恒断言后，RewardCalculator 与真实行情 golden **15 passed**。Ruff/py_compile 与 Stage53 的独立验证器均通过；未查 Validation、Test、Holdout 或 outer 数据。

该审计的测试结果不是市场策略表现，也没有改变策略算法或降低风控。证据包括 `tests/test_real_golden_ethusdt.py` 中的新逐步对账断言，以及 Stage53 既有 `oos-decisions.csv` 对成本情景的独立复算。当前优先问题回到学习器能否获得方向信号：Stage53 的全仓毛收益每笔只有约 1.17/ -5.27/ 8.31 bp，而往返 base 成本为 12 bp；单纯扩大探索或交易频率无法补足这个缺口。

**下一项：** 暂停模型尺度和交易频率近邻扫参。Stage55 回到当前 PPO Train-only 执行日志，按相同窗口/seed 把 `requested_action → risk-adjusted target → planner action/reason → fill qty → fee/Funding → Δequity → r_econ/r_shape/r_train` 对齐成逐决策闭合表，并按市场波动/方向/持仓转移分层；先审计已保存 campaign 的原始日志，不消费新数据。如发现动作死区以外仍有系统性执行偏差，再登记单一动作表示或可观测特征实验；若环境会计闭合且多数亏损来自入场方向毛收益，则暂停超参搜索，转向新的独立、微观结构或可解释外生信息来源。

## 第五十五项：冻结 Train 训练成本与 reward 日志审计（已完成，2026-10-16）

对现有训练材料做离线核对，不重新训练、不读取 Validation/Holdout。确定性 native 账户报告有 7,199 行账户状态/7,198 个分钟收益，初始权益 10,000 USDT、最终权益 9,993.12775822 USDT，净收益 -0.000687224178；扣除已报告费用、滑点与 Funding 前为 -0.000477234002，报告含 61 笔成交、终局空仓、0 次强平。R1 诊断包含 8,192 个采样转移、6,310 个唯一时间戳、revisit fraction 0.229736、`r_econ_sum=-0.04926267896`、`r_shape_sum=0`、reward scale 1000、`r_train_sum=-49.26267896`。请求绝对动作均值 0.1073、执行绝对动作均值 0.07264；计划原因包括 deadband 1,927、hard-risk reduce 2,194、rebalance 4,071，提交 76.48%、noop 23.52%。

决定性限制是 native 账户报告与 R1 诊断来自相同 Train 时间区间、不同动作轨迹，因此不能逐步配对，也不能把两个来源合并成已训练策略的净收益证明。预登记 SHA-256 `3b9cec31161c5d80564bab7b91f4f9c7e0c6d86454687248eacae178c3576592`，分析器 SHA-256 `fd1de6703de5dfbfbbbe72fa9415042abe5244d75a3b8e788d4440a11032c015`，结果 SHA-256 `533497cbb54153db145f17d952b981f40040f9ed2719e5c31ffb29830cd30cf0`，证据在 `.tmp/rl-training-cost-reward-audit-stage55-20261016/`。

**下一项：** 会计和成本字段没有发现重复扣费，但已有诊断缺少请求动作幅度与逐步经济结果的交叉统计。Stage56 只添加按请求绝对动作分箱的描述性训练统计，校验各箱样本数及 reward 分量守恒；不改 policy、动作、奖励或风控。该统计只解释当前训练转移，不做因果推断或策略筛选。

## 第五十六项：R1 动作幅度与经济 reward 分箱诊断（已完成，2026-10-16）

在 `TrainingDiagnostics` 中加入 `<0.05`、`0.05–0.15`、`0.15–0.30`、`0.30–0.60`、`0.60–1.00` 五个请求绝对动作箱；每箱输出样本数、`r_econ` 求和/均值、`r_shape` 求和、hard-risk intervention 占比、submit 占比及平均执行绝对敞口。写盘 schema 从 `rles.r1-learning.v1` 升至 `rles.r1-learning.v2`，读取旧版 resume 证据仍兼容。新增训练 episode 分箱顺序/计数/reward 对账断言，并保留 Stage54 真实行情 reward 守恒 golden。

Ruff、py_compile 通过；Mypy 对 107 个 source 成功。包含 provider、应用 CLI、reward scale、native episode 和真实 ETH golden 的相关测试 **62 passed**。另有 4 个 hardlink/curated snapshot 参数用例被 Windows 沙箱禁止 `os.link` 阻断，属于环境权限限制，没有绕过文件系统安全边界。实现及测试 SHA-256：`provider.py` `7a12dfcd4252d46dd1c4704e3f1dcc4259e04ac29008423bd18f3debbf49a849`；`rl_stage.py` `8a263ff24d731f63020756a663bbcf57cfe99d9bae89cd1124bb0a9e8123529d`；`historical.py` `0e588b6531d9c9b79192f1d2bed9aa668b3e582e94b3bca202f07ceed5745f79`；`test_native_training_episode.py` `ab4ba0568189595ad0cd78e778e3a283ca64c5df032118019691820ff7aa32aa`；`test_historical_factory.py` `7622edd69f1571802e50e15ce2fb53f34a91c2b1f5ea009de1f826c4e5aab9fe`；`test_real_golden_ethusdt.py` `6ed0cc7505ca9eb64f3a1fb00894fd25ad2a716243f75aff0edc9bdd9ee90cc6`。

**下一项：** 用同一既有 Train 区间和冻结 PPO 参数作短程诊断复现，确认 v2 分箱可经真实训练写盘，并按 seed 检查统计稳定性。必须关闭 Validation 调用，不将该固定 Train 样本称为收益证据；如短程诊断显示时间窗过窄，应先解决样本覆盖，再讨论调参。

## 第五十七项：R1 v2 动作经济分箱的 Train-only 短程复现（已完成，2026-10-16）

使用固定 PPO 参数、reward scale 1000、单环境、8,192 步、单次 8,192 步 episode、既有 Train 数据与 cutoff `2026-09-09T00:01:00.001Z`。通过 `train(..., evaluate=False)` 运行，`COMPLETED.json` 的 report hash 为 null；所有复现的 Validation/Test/Holdout/outer 2026 查询均为 0。首次 CLI 调用因 CLI 默认 `evaluate=True`、而 config 的 `evaluation=null` 在 learner 创建前被拒绝，未触发验证集查询。seed 103 的第一次 Python 运行因恢复 1 GiB 内存余量而中断，之后在独立干净输出目录重跑完成。seed 101 的完成产物因 config 文件 SHA-256 `9e4bf342d2dd20792868dd4ed77335d8037fcdcdbec2d50a42723fdc10ae7d5c` 与其预登记的 `0f91bea0ac17bbd37a102bbce1cd714a8fb03c91865ef0bc2d0cc5090c361310` 不一致，从正式三 seed 汇总中排除，仅保留为探索性材料。

预登记及配置哈希匹配的 seed 102/103/104 均完整 8,192 步，v2 五箱合计 8,192 样本，分箱 `r_econ` 求和与总 `r_econ` 差异小于 `1e-10`，`r_train = 1000 × r_econ`，`r_shape=0`，sidecar SHA-256 与日志文件一致。三 seed 的 `r_econ_sum` 为 **-0.04877/-0.04827/-0.05111**，critic explained variance 为 **-0.471/-0.867/-0.607**；请求绝对动作均值约 **0.108**，执行绝对动作均值约 **0.072**，hard-risk intervention 占比约 **26.7%**。三个种子中 0.05–0.15、0.15–0.30 等动作箱的经济奖励累计都为负。这里只是训练 reward 的描述性相关，不能推断动作幅度导致亏损，也不是完整策略净值。

训练实际从 `2024-01-01T00:03:00.001Z` 连续走到 `2024-01-06T16:34:00.001Z`，只覆盖约 5.7 天；此前反复使用过此 Train 区间。这项实验不足以证明模型学到了跨市场阶段规律，更没有正收益结论。独立核验器 SHA-256 `673baed5b9009f9231b4238024d0a9af199efe2149cea53949da5709c337191`；验证摘要 SHA-256 `b3d834ccb24f9cc4a174b96ceebe8225671030664bc2406fdce01eff9325fd75`。证据分布于 `.tmp/rl-stage57-action-econ-bins-20261016/` 与 seed102–104 子目录。

内存监控显示单路完整训练工作集可升至约 8.5 GiB。三路并发曾使系统空闲内存跌破 1 GiB并触发工作集回收，因此停止一条训练；后续两路并发稳定完成。当前策略是在保留至少 1 GiB系统余量的前提下按实测并行度运行，不用无关进程占内存。训练完成时可用内存约 23.5 GiB。

**下一项：** Stage57 暴露的首要问题不是 PPO 超参，而是 8,192 步仅覆盖 Train 开始的 5.7 天，无法代表多年份市场。Stage58 冻结优化器与动作/奖励，按预登记、时间上分散且互不重叠的 Train 窗口重新安排短 episode；优先顺序执行或最多两路并发，逐次验证实际 decision timestamps 覆盖，确保保留至少 1 GiB 空闲内存。先证明训练样本跨越不同波动/趋势阶段，再决定是否扩大总步长或改变观察特征；不读取 Validation/Holdout，不通过重复试验挑“盈利窗口”。

## 第五十八项：分散 Train 时间窗的固定 PPO 短程诊断（已完成，2026-10-16）

为避免 Stage57 只接触 2024-01-01 开始的短段，冻结相同 PPO seed 102、奖励/动作/执行规则与 8,192 步训练配置，预登记 2024-01、2024-09、2025-05、2026-01 四个相隔约八个月的 Train 窗口，每窗 8,192 个连续决策。config 中把 funding settlement calendar 限制在各自已登记区间内，防止边界外结算点被错误纳入质量检查。第一次运行因全量 Funding 时间表超出窗口被 DQ 校验拒绝，未开始 learner；修订预登记后四个窗口均完成。两路并行，空闲内存大致保持 7–16 GiB，没有跌破 1 GiB。

四窗 `r_econ_sum` 分别为 **-0.04877、-0.04896、-0.04953、-0.04838**，critic explained variance 为 **-0.471、+0.004、-0.819、+0.164**。请求绝对动作均值为 **0.1073–0.1078**、执行绝对动作均值 **0.0714–0.0724**，hard-risk intervention 均约 **26.0%–26.6%**。经济 reward 在四个相隔时期都为负，动作尺度与风险调整比例也接近；但这只是训练 reward 与状态统计，不能当作净收益或时段泛化证明。2024-01 窗口和 seed102 与 Stage57 完全相同，结果逐项复现。四窗合计 32,768 个决策，全部来自既有 Train，Validation/Test/Holdout/outer 2026 查询为零。

独立验证器逐窗核验 source/config 哈希、training schema、样本数、action bins/reward 守恒、时间范围和完成收据；摘要 SHA-256 `a869f4adcbfa8cfada3b8ae988896b443a50fa22db953842669a504aa09a73c9`，验证器 SHA-256 `9e9d9573648a72e716456c72b2ffda73164a31e252a7271db93e5ee0bddd69a8`。证据位于 `.tmp/rl-stage58-train-window-coverage-20261016/`。

**下一项：** 多窗训练仍显示价值拟合不稳定，legacy observation 只有 1 分钟收益、5 分钟波动/成交量、basis 与已结算 funding。Stage59 保持同四个时间段、seed、奖励、动作和训练步数不变，只切换到已实现且有回归覆盖的 `causal_mtf_v2` 特征（多周期收益/波动与 trend-efficiency）。按预登记门槛比较四窗 critic explained variance 和训练经济 reward；只有稳定改善才扩大到长程 Train 学习，否则丢弃这组输入增量。仍不触碰 Validation/Holdout，也不据短段收益宣称盈利。

## 第五十九项：causal_mtf_v2 多周期输入四窗 Train-only 筛查（已完成，2026-10-16）

在 Stage58 四个相同 Train 时间窗、相同 seed 102、8,192 步、奖励/动作/训练超参与同一训练起点下，仅将 observation profile 从 legacy 切到 `causal_mtf_v2`，加入已关闭 bar 的多周期收益/波动和 trend-efficiency 特征。所有 run 的 Validation 调用关闭；四窗均按各自 Funding settlement interval 裁剪时间表。source/config/sidecar、样本数 8,192、五档动作计数及 reward 守恒全部通过独立 verifier。

预登记门槛是 median explained variance 至少提高 0.1、至少 3/4 窗改善，且平均经济 reward 不比 legacy 差超过 0.01。结果 median explained variance 从 legacy **-0.2333** 降至 **-1.6015**，差 **-1.3682**，0/4 窗改善；平均 `r_econ_sum` 从 **-0.04891** 到 **-0.04925**，差 **-0.000345**。经济 reward 非劣门槛通过，但 critic 两项门槛均失败，因此**拒绝 causal_mtf_v2 进入更长 RL 训练**，不把“观察维度更多”视为学习更好。四窗收益依然全部为负，且仍是已复用 Train 历史的 5.7 天短 episode，不能作为 OOS 收益证据。

摘要 SHA-256 `326d8f9dde552f97c2763e411bf323cd8805570198f4e3e618dbc0d4d58aceaa`；独立验证器 SHA-256 `07418ffa968bc9de042d71852d7e715effd279d87ea7bb304ea477c6f9f952c4`；证据位于 `.tmp/rl-stage59-causal-mtf-v2-train-only-20261016/`。

**下一项：** 多周期输入没有提升 critic，继续加特征缺少证据。Stage60 保持 legacy observation、四个时间窗、奖励函数、交易动作语义和全部 PPO 参数不变，只把 `log_std_init` 从 **-2.0** 调到 **-1.5**，预登记检查策略是否能探索到更广动作，同时经济 reward 与 hard-risk 干预是否恶化。该对照用于隔离动作探索尺度，不是把更大仓位当成盈利来源；若预定门槛失败，恢复原尺度并转向更长、跨时期样本覆盖，而非继续加大方差。

## 第六十项：PPO 高斯动作探索尺度对照（已完成，2026-10-16）

固定 Stage58 的四个时间窗、seed 102、legacy observation、reward、执行/风控规则和训练预算，只把 `log_std_init` 从 **-2.0** 调为 **-1.5**。四次结果都通过 config/预登记哈希、8,192 步计数、v2 action-bin 与 reward 守恒核验，未查询 Validation/Test/Holdout/outer 2026。

平均请求绝对动作从 **0.10756** 增至 **0.17707**（**1.646 倍**），平均执行绝对动作从约 **0.0719** 增至约 **0.1058**，hard-risk intervention 均值反而略降 **0.0044**；中位 explained variance 从 **-0.2333** 微升至 **-0.2125**。但平均经济 reward 从 **-0.04891** 恶化至 **-0.07229**，差 **-0.02338**，未通过预登记的 **-0.005** 非劣门槛；其余三个 gate 通过。提高探索确实放大了动作和执行敞口，却明显拉低 Train reward，因此拒绝 `log_std_init=-1.5`，不以“动作更多”作为改进。

独立摘要 SHA-256 `ad99beacba89d18f99a26da982e01c2889c9bbe3b12f6e34d31fb07102f4abe2`，验证器 SHA-256 `942dbddf9b3be26e6d7432ee9e0939af5e6d21597ada35949dfdc03493e247b5`，证据位于 `.tmp/rl-stage60-action-exploration-20261016/`。这是已用 Train 样本上的短程训练 reward，不是独立回测或市场盈利证据。

**下一项：** Stage60 显示更频繁、更大的可执行持仓在本次 Train 样本上付出更大损失。Stage61 固定恢复 `log_std_init=-2.0`，只把执行 deadband 从 **0.05** 提高至 **0.10**，在相同四个窗口测量经济 reward、rebalance 数和 hard-risk 干预；通过门槛预先要求平均 `r_econ_sum` 改善至少 0.005、至少 3/4 窗改善、rebalance 降低至少 15%，且 hard-risk 干预不增加超过 2 个百分点。若成本收益门槛失败，就不继续扩大 deadband；保留原规则并转向完整行情覆盖下的学习方案。

## 第六十一项：执行 deadband 0.10 的四窗 Train-only 筛查（已完成，2026-10-16）

使用 Stage58 同一组四窗和 seed 102，固定 PPO、legacy observation、奖励与风险规则，只把执行 deadband 从 **0.05** 提到 **0.10**。四窗配置/预登记哈希、R1 sidecar、8,192 步样本数、五箱计数与经济/训练 reward 守恒均通过独立核验，Validation/Test/Holdout/outer 2026 查询为零。

平均训练 `r_econ_sum` 从 legacy **-0.04891** 改善到 **-0.04383**，差 **+0.005079**；**4/4** 窗改善。平均 rebalance 从 **4,021.25** 次降到 **2,756.25** 次，下降 **31.46%**；hard-risk intervention 平均下降 **5.01 个百分点**。四项预登记 gate 全部通过，所以将 deadband 0.10 提名为下一轮长窗训练候选。但每一窗仍是负经济 reward，不能说策略盈利；短 episode、已反复使用的 Train 日期以及 scenario 成本也限制结论。

独立摘要 SHA-256 `dc349ba3965ede89b70929aa6dfd3a570f84453f7b510e7dcff33c864309fbbf`，验证器 SHA-256 `db3dc53cb00621fa2798022bdd4a9799507c6ca48b4d8a7e907a1b43053aca7a`，证据位于 `.tmp/rl-stage61-deadband-cost-control-20261016/`。

**下一项：** Stage62 固定 deadband 0.10、legacy observation 和 PPO 参数，改成一个连续 learner 依次遍历 4 个 rank-rotated、互不重叠的 8,192-step episode（共 32,768 步），让同一模型在四段跨年份 Train 窗口更新；运行两个预登记 seed，验证实际时间戳覆盖、episode 数与 reward。单环境顺序重置以限制内存；只在已注册 Train 内评估，仍不触碰 Validation/Holdout。若多窗口完整训练能稳定覆盖各时期，再运行固定策略的全 Train native 账户回放；否则继续检查状态/动作能否学到毛利而不是再扫阈值。

## 第六十二项：deadband 0.10 下跨时期轮转 Train 训练（已完成，2026-10-16）

将 Stage61 deadband 候选 **0.10** 放入同一个连续 PPO learner，固定 seed **102/211** 两个预登记运行；每次 32,768 步、单环境、每个 episode 8,192 步，episode 之间按 Train lane 的 coprime rotating offset 重新选窗。每 seed 四个 episode 实际起点相同，分布于 2024-01、2024-08、2025-08、2026-04 左右，首尾跨度约 **839.6 天**；每个 episode 8,192 个唯一时间戳，合计 32,768 步，无重复。两个 seed 都满足预登记 episode 数、时间跨度、schema、hash 和 reward-bin 守恒门槛；Validation/Test/Holdout/outer 2026 查询为零。

两 seed 的各 episode `r_econ_sum` 随学习更新从早期约 **-0.044/-0.046** 逐步到后段约 **-0.039**，后段 critic explained variance 转正；但 episode 会重置账户，累计训练 reward 不是同一账户复利曲线，起点顺序也不是严格时间递增，因此不能解释为跨年盈利或单调策略改善。模型仍未在未见数据上验证。空闲内存最低约 **1.66 GiB** 后回升；两路训练完整结束，没有低于 1 GiB。

独立摘要 SHA-256 `d5a7e9833b96d3224a9bd01f69363ef647a4a10132894e74311c28fb6e5f1d85`，验证器 SHA-256 `3a137324f90c71e2cfbc535653b834bdbaa1931ac99b1ffc2cb2e52dbc24fb4e`；证据位于 `.tmp/rl-stage62-rotating-full-train-episodes-20261016/`。

**下一项：** Stage62 证明单环境 episode 轮转可以在低内存下覆盖多时期。Stage63 注册完全相同的 32,768 步、两 seed、四窗序列与所有 learner 参数，只把 deadband 回到 legacy **0.05**，和 Stage62 的 0.10 做逐 seed/窗配对；检查经济 reward 差、rebalance 数和风险干预。若 0.10 的改善不能在更长训练复现，就撤销候选；即使复现，也只能进入完整 Train 固定策略成本回放，不算最终正收益证明。

## 第六十三项：deadband 0.05/0.10 的长程配对对照（已完成，2026-10-16）

使用 Stage62 的相同 PPO seed **102/211**、32,768 步、四个 rank-rotated Train episode、初始化、legacy observation、reward、学习超参与时间戳序列，仅将执行 deadband 固定为控制组 **0.05**，与 Stage62 候选 **0.10** 配对。两组每个 seed 的四段 episode 完全对齐；所有运行均在 Train 中，未查询 Validation/Test/Holdout/outer 2026。独立 verifier 检查配置与来源、步数和 episode 覆盖、奖励及 action-bin 守恒。

预登记比较门槛通过：0.10 相对 0.05 的平均配对 `r_econ_sum` 差为 **+0.005972**，**8/8** seed-episode 配对均改善；平均 rebalance 从 **4,211** 降至 **2,850**（下降 **32.32%**）；hard-risk intervention 下降 **5.50 个百分点**。这在长训练复现了 Stage61 的交易成本/执行稳定性优势，故保留 deadband 0.10 作为当前研究候选。不过每个 episode 的经济回报依然为负；该指标是训练 reward 而非账户净收益，也没有独立数据验证，不能称为盈利策略。

证据位于 `.tmp/rl-stage63-deadband-long-train-control-20261016/`。验证摘要 SHA-256 `f4f97d8a17c9ebe80e8d74652967310093ab2344d8da3de46ac793675465da39`；独立 verifier SHA-256 `b15ef92216c003b2b85f923358c4aa2c2e3764b35fc6b4fc12bca7b439cb49f8`。

**下一项：** Stage64 冻结 Stage62 deadband 0.10 两个 seed 的模型，在完整 Train 时间范围做 native account 回放，复算净值、回撤、换手、成交成本、风险事件并与同 seed 的 0.05 控制策略比较。该回放是全 Train 训练区间机制诊断，不是独立 OOS 检验；只要资源允许，单进程依次运行以控制内存，Validation/Test/Holdout 继续保持零查询。完整 Train 若显示账户级损失，就按归因继续改进状态、奖励或执行，而不把 reward 指标冒充盈利。

## 第六十四项：deadband 0.10 冻结策略全 Train 原生账户回放（已完成，2026-10-16）

Stage64 在 Stage62 seed102 bundle 上运行 `diagnose-train`，固定预处理 scaler 和训练时 environment/execution identity，对完整 Train 区间 **1,414,078** 个分钟观察进行原生账户回放；没有拟合 scaler、没有查询 Validation/Test/Holdout。单进程运行约 **19 分钟**，最低可用内存约 **3.9 GiB**，未低于 1 GiB。报告 SHA-256 `1caeff01931cd791453b6c653b11c48ee83ab900d21036980f4d23241dc3e244`；独立验证器检查完成收据、bundle/report 哈希、时间边界、零 evaluation 查询、scaler 冻结与账户/动作统计，全部通过。验证摘要 SHA-256 `be0c440a2d0c66eb98039b8ed6af20843bbb0275ebd28dfc03a263a4cff31bd0`；验证器 SHA-256 `1bbe8619d2c24be5a6077f41805d79f8782fb3e6b44f607bea9089175cdc3934`。证据位于 `.tmp/rl-stage64-full-train-account-replay-20261016/`。

结果揭示 Stage62 候选是**全程空仓策略**：策略请求动作绝对值均值 **0.01965**、p95 **0.04432**，99.9999% 请求动作非零且几乎全为负向，但执行 deadband 为 **0.10**，因此 **1,414,077/1,414,077** 次计划均为 `NOOP / deadband`，执行非零仓位比例为 0。净收益 **0**、回撤 **0**、换手/手续费/资金费/滑点成本均为 **0**，无强平。该零损益是未交易，绝非正收益；Stage61–63 经济训练 reward 的改善主要与压制交易动作一致。Stage63 的 deadband 配对结论因此只代表 reward 与调仓频率变化，不能作为策略质量改善的证据。

**下一项：** Stage65 对同 seed102、相同完整 Train 窗口运行 Stage63 的 **deadband 0.05 控制 bundle**，判断冻结策略是否同样空仓，并对比动作幅度相对于执行阈值的分布。如果控制策略也无交易，Stage66 将改用预登记的小 deadband/连续目标仓位映射做短窗筛选，并以全 Train 策略输出校准动作幅度；若 0.05 能交易，则进一步检查 0.10 对交易信号的抑制影响。每次完整回放仍串行，保护 1 GiB 可用内存底线，不将账户净值与 Train reward 混为一谈。

## 第六十五项：首次 0.05 全 Train 回放的内存保护中断（已停止，2026-10-16）

Stage65 首次尝试使用完整 `account_rows` 和逐分钟全量账户/订单快照，对 Stage63 seed102 deadband 0.05 控制策略做全 Train 回放。运行约 2 小时 15 分钟后，可用内存降至约 1.02 GiB，按预先设定的 2 GiB 预警线立即中断；进程正常退出、内存恢复，没有完整报告，因此本次不产生策略结果，也不推断盈亏。

**工程修复：** 为诊断专用历史快照增加紧凑保留选项：中间时点只留账户净值、当步 fill/funding 增量和统计所需字段，终点保留完整的终端持仓/订单及累计成交/资金费历史；增加 `summary_only` 报告参数，完整经济指标仍照常计算，但不把 140 多万行 `account_rows` 序列化进诊断 JSON。正常 backtest 默认仍保留完整账户明细。修复前 Stage64 0.10 的报告约 303 MB；紧凑诊断报告约 143 KB，候选回放工作集峰值约 11.6 GiB、用时约 18 分钟。该模式保留用于计算净值、费用、资金费、滑点、成交和终端状态所需的数据。

## 第六十六项：紧凑快照模式的完整 Train 候选/控制账户回放（已完成，2026-10-16）

按 Stage66 预登记顺序、串行回放两个冻结 seed102 bundle；完整 Train 窗口为 `1704067260001000000..1788912060001000000`，共 **1,414,078** 个观察，scaler 冻结、Evaluation 查询为零。候选 0.10 报告 SHA-256 `5281016794fb3390837c5472e414c06ac076e9a99a26c1f1437bb71f49959eb5`，bundle SHA-256 `ea2ea516780f25756c7dae35636ee59eca7e3443a24259c6b58526b2536fec68`。控制 0.05 报告 SHA-256 `2d36b60737a663484a049cde209300f6be2f0ba311b2d3dfd2aae1489c7a4a42`，bundle SHA-256 `cec246832d0f2d9ac75eec762e016f9efd63b02968740c8fa068cc3a8be7899c`。两个完成收据、report sidecar、配置/bundle 身份和窗口边界均经独立断言核验；摘要在 `.tmp/rl-stage66-memory-bounded-full-train-replays-20261016/`。

对照揭示 0.10 **1,414,077/1,414,077** 个计划都是 `NOOP / deadband`，执行非零仓位为零，净收益和成本为零，仍是空仓策略。0.05 虽有 **45** 笔成交（44 次策略提交，另含终端平仓）、**2,939** 个资金费事件，但 **1,414,033/1,414,077（99.9969%）** 个计划仍为 `NOOP / deadband`；策略几乎持续维持约 **5.51%** 的负向敞口。完整 Train 净收益 **-0.13056%**（约 -13.06 USDT/起始 10,000），最大回撤 **0.6750%**，Sharpe **-0.125**，换手比 **0.2443**，手续费 **1.2193 USDT**、Funding PnL **+12.5997 USDT**、滑点成本 **0.24749 USDT**；终端空仓、无强平。`0.10` 对 `0.05` 的表面训练 reward 改善由交易阈值把本就偏小的策略动作完全压成空仓所致，不能作为学习改善或盈利证据。

紧凑候选回放约 18 分钟；0.05 对照运行约 2.5 小时。控制进程工作集峰值约 **11.1 GiB**，系统可用内存最低约 **8.8 GiB**，明显高于用户的 1 GiB 保留目标和本次预登记 2 GiB 停止警戒线。指标/奖励相关单元测试最初 **18 passed**，4 个合成 Parquet 用例被 Windows 沙箱禁止 `os.link` 阻断；随后用 `.tmp` 下的 multiprocessing 兼容测试入口，并仅在测试 runner 中将 `os.link` 暂时替换为目标独占复制，相关测试 **22 passed**。这验证了快照工厂与诊断路径，但不验证 hardlink 本身的原子性语义。Ruff 与 `git diff --check` 通过。

**下一项：** 保留 deadband 0.05 为机制对照，不把 0.10 空仓误判为优秀。下一轮先检查请求动作的绝对值和几乎恒定的负向偏置，设计可学习、可执行的动作映射与仓位变化机制；用短 Train 窗口筛除“空仓/单边钉住/高换手”方案，再只对预登记通过者做一份完整 Train 账户回放。完整窗口验证依然只属 Train 机制诊断，任何独立收益判断必须留给未参与筛选的数据。

## 第六十七项：回放热点 profiling 与不可变身份缓存（已完成，2026-10-16）

用 Stage63 seed102 冻结 actor，在同一 Train 初始时间片逐项对比 8,192 步真实策略回放。缓存前回放 10.239 秒（1.250 毫秒/步），缓存 `FeatureSpec.hash` 后为 9.512 秒，再缓存训练 scaler 与执行配置 fingerprint 后为 7.152 秒（0.873 毫秒/步）；相对基线约快 **30.1%**。三次的 8,193 个账户观察、动作与计划计数、请求/执行敞口、净收益 **+0.02450%**、回撤 **0.03969%**、费用、资金费、滑点和换手报告逐项完全相同。该正值只是 Train 开头短片的描述性结果，绝不代表完整策略或 OOS 盈利。

热点确认每步重复将不可变 `FeatureSpec`、训练 `Scaler`、`ResolvedExecutionConfig` 转为字典并重新 SHA-256。为 Event、FeatureRow、FeatureSpec 和 Scaler 的稳定 hash 加懒缓存；为冻结执行配置 fingerprint 加最多 128 项 LRU 缓存。20,000 次相同 FeatureSpec 查询微基准比每次重新 digest 快约 **2,622 倍**。真实性能剖析显示 8,192 步 rollout 由 10.071 秒（首轮含导入/初始化）降至 7.152 秒，主要剩余耗时为环境 step、Nautilus 账本推进/预检、计划生成及观测构建。环境数据解析/特征行准备仍约 **255 秒**，是下一处需要定位的启动成本。

整个调优与测试过程的系统可用内存最低约 **12.2 GiB**，显著超过用户要求保留的 1 GiB；此前 Stage66 两个完整账户回放的最低可用内存约 8.8 GiB。未关闭事件哈希、执行身份、账本 preflight 或账户核验。`test_l2_gym_bridge.py`、`test_historical_factory.py` 共 **42 passed**，`test_execution_contract_bridge.py` **25 passed**，Ruff 与 `git diff --check` 通过。首轮 suite 暴露多处微型 Train fixture 沿用默认 256 步蒸馏语料、但实际只有数个决策点；将相应测试 fixture 明确设为 2 步后，全部相关集成测试通过。profiling 证据及缓存前后 JSON 位于 `.tmp/rl-stage66-memory-bounded-full-train-replays-20261016/`。

**下一项：** Stage67 只验证执行与报告等价性及短片速度，不重新筛选策略或消耗 Evaluation 数据。Stage68 继续定位约 255 秒环境准备成本，尤其是经过校验的 Parquet 事件解码、CuratedStore 建立和逐分钟多周期特征行生成；只优化可证明与现有因果/时间戳/快照 hash 完全等价的路径，再以同一短片和完整回归测定提速。之后回到动作映射和长窗学习机制，不把更快回放误称为更好的策略。

## 第六十八项：Curated 环境构建与事件快照哈希提速（已完成，2026-10-16）

对 Stage63 的完整 Train 数据环境构建做 CPU profile：读取和验证 **4,403,867** 个事件、构建 **1,414,080** 个分钟特征行。剖析器有明显开销（profiled wall time 761 秒），但函数调用显示单次市场读取及再次合并规则时会完整遍历/序列化事件，`_snapshot_digest` 上有 **8.8 百万**次事件规范化；多次 `make_feature_row` 查 bars 和每行 rule lookup 是剩余特征准备热点。另发现 `read_curated` 先调用 `validate_payload`，随后每个 `Event` 构造器再执行相同的严格校验。

保留 Event 构造器的 schema 与 payload 校验，只删除读取路径中重复的那一遍。`_snapshot_digest` 改为按已验证、冻结 `Event` 的固定字段直接输出规范 JSON，仍使用相同的排序键、分隔符、UTF-8、无 NaN 规则和数组括号；不改事件字段或身份格式。优化前后完整市场 snapshot hash 都是 `258834aba27e77b0c97fbb4a5ca9fc6c18699f2f8d0d56c77812d1946bbc1adb`，feature hash 都是 `894af47c4ebeecc5cfeedccfbe4e994920905344d033b0deb821bc2bfe8d3c3d`，事件数和特征行数相同。正常环境构建从 **251.75 秒**降至 **180.14 秒**，约快 **28.4%**；其中该机器重复解析和系统文件缓存会影响绝对时长，后续比较继续用同一冻结数据和 warm-cache 条件。

事件守卫、快照性能基准、真实 ETH golden、历史环境、执行契约与 L2 bridge 回归共 **76 passed**；Ruff 和 `git diff --check` 通过。最近运行的最低系统可用内存约 **12.2 GiB**，远高于 1 GiB 保留要求。直接事件编码的完整性回归仍确认 `_snapshot_digest(events) == digest(events)`，真实数据 commit marker 也通过。结果与 profiles 保存在 `.tmp/rl-stage66-memory-bounded-full-train-replays-20261016/`。

**Stage69 进展（2026-10-17，部分完成）：** 在 `CuratedStore` 为规则建立按 `max(available_at_ns, effective_ns)` 排序的前缀索引，保留旧的 as-of 选择、过期拒绝和同生效时间冲突行为；对 136 条合成规则的 1,000 次查询与旧算法输出一致，查询约快 85 倍。多周期特征对有界事件窗口跳过冗余排序、直接聚合冻结 `Event` 并用事件时间 `groupby`。新增每个 timeframe 的懒加载闭合桶索引，lookback 通过二分查询；含重复 close 的历史保留原始逐决策路径。360 天/535,680 事件合成样本上比较 20,000 个分布式决策行，旧聚合 19.13 秒，索引路径含冷建索引 2.91 秒（约 6.58 倍）；哈希逐项相同，峰值工作集增量 630.5 MiB。60 天连续 20,000 行旧/新耗时 18.78/11.94 秒，哈希相同。规则延迟/冲突、延迟 bar、闭合桶、重复 close 和 gap 定向回归合计 29 项通过，Ruff/MyPy 通过。

**Stage69 剩余验收：** 60/360 天合成 workload 已测，但尚未验证真实授权 Train 数据上的完整特征矩阵、构建时间和 RSS，也没有独立 Validation 的性能前后对照。取得登记允许的 Train-inner 窗口后再冻结比较；在此之前 Stage69 保持 `PARTIAL`，不得把合成提速写成真实研究周期或策略表现改善。无需为了追求更高合成倍数继续扩大滚动缓存复杂度。

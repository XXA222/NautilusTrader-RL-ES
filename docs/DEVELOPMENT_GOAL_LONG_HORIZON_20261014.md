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
- 整机内存至少预留约 8 GiB 给 Windows 和其他进程。worker 继续受 Windows Job 提交内存限制；每次提并发前检查可用提交额度，任何越限或异常退出都保留失败回执并回收进程树。
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

该目标持续有效，完成一个工程阶段不等于目标整体完成。每次续开发先检查最新数据质量、使用台账、代码和硬件占用，再记录当期可验证的进展与下一项工作。

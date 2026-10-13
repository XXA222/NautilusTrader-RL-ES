# Nautilus-RLES — AI-04 / L4 RL SB3 v0.4.1 功能说明

## 2026-10-13 集成增量：v2 E1 输入域与 BC 预算

- `RLProvider.calibrate_observation_normalization(raw_observations, training_split='train_inner')` 仅允许尚未训练/恢复的新 provider，在 observation-only training VecNormalize 上校准一次。拒绝 Validation、Test、Holdout、重复校准及已训练状态。它只用于新 v2 E1 的 Train teacher 语料，不更改 L2 feature scaler；恢复时使用 checkpoint 内已保存统计，不再次拟合。统计 count 包含校准语料，不能再等同于 R1 采样步数；采样预算以 `num_timesteps`/stage usage 为准。
- v2 E1→R1 迁移先将固定输入尺度及已校准的 L4 均值/标准差组合到首层参数，再执行原有参数覆盖和动作 parity；失败仍进入明确 BC 路径，不能把蒸馏称为权重继承。原生 SB3 更新公式和奖励语义保持不变。
- Train-only 校准将 clip_obs 固定为覆盖完整 Train 语料的边界（至少 10，保留 0.1% 余量），并保存到现有 normalizer metadata。新 v2 E1→PPO 的标准 Tanh 网络显式增加标量输出 Tanh 节点，再连接初始化为 action scale/零偏置的最终 PPO head；这复现 CanonicalActor 的末端 tanh。源最后线性层映射到该节点，构造 head 也计入完整目标参数覆盖，最终仍要求原容差内的 parity。旧 checkpoint 通过 run manifest 保持原结构。
- `RLProvider.distill(..., early_stop=False)` 新增显式开关；打开后每个完整 epoch 检查整份 Train 语料的动作 parity，仅满足原容差才停止。`ParityReport.epochs_completed` 返回实际次数。应用 Hybrid 默认打开，配置 `distillation.epochs` 为上限、`distillation.early_stop=False` 可要求跑满；L5 `DistillationEvidence.epochs` 记录实际次数，配置/run recipe 保留预算上限。旧 L4 无实际计数字段的报告按其原始跑满预算语义处理。

以下为原始模块基线，当前运行证据见 `docs/DEVELOPMENT_GOAL_CURRENT.md`。

**设计契约**：仍为 v0.4.0（2026-10-11），本次仅为 L4 内部实现修订 v0.4.1；不调整其他开发线冻结协议。stage.v1/v2 checkpoint 不能直接恢复到 stage.v3。

**工作包**：L4-001 ~ L4-018。**所有权**：仅 `rles-rl-sb3`。**交付类型**：生产源码和本功能说明；无其他开发线源码、测试代码、数据、日志、检查点或证据包。

## 1. 组件与边界

- `rles_rl_sb3.config`：严格校验 `RLConfig`、`PolicySpec`、`TransitionFingerprint`、`SeedSpec`；所有超参数封装为只读映射，错误类型、未知字段、无效激活或 dtype 硬失败。动作固定为单维 `float32` `Box([-1,1])`，观测固定 `float32` 向量。计算模式在 L6 解析后仅接受 `cpu` 或 `cuda`。
- `rles_rl_sb3.provider`：依赖注入式 `RLProvider`，原生 SB3 `PPO`、`SAC`、`TD3`、`DDPG`；统一创建、训练、预测、评估、保存、恢复、迁移、行为克隆、性能采集。插件入口：`nautilus_rles.rl:sb3` → `rles_rl_sb3:provider`。
- `rles_rl_sb3.envs`：仅封装 L3 所提供 Gymnasium 环境。单环境使用 `DummyVecEnv`；多原生环境通过 L3 deadline barrier 在独立 spawn 进程同步前进，只重置已完成的环境；没有异步跨屏障或陈旧动作重用，也不实现任何交易 step / 风控 / 奖励。
- `rles_rl_sb3.normalize`：训练阶段可选 `VecNormalize(norm_obs=True,norm_reward=False)`；验证/测试阶段冻结为 `training=False`，恢复的均值、方差、样本量必须严格匹配。上游市场特征 scaler 由 L2/L3 管理，不在 L4 偷偷重拟合。
- `rles_rl_sb3.noise`：TD3、DDPG 使用独立生成器的 normal exploration noise；`SeedSpec` 中保留 algorithm、每环境、noise、replay 四类种子。SAC 的熵机制由 SB3 原生算法承担；不将 DDPG 偷换为 TD3 超参数配置。
- `rles_rl_sb3.checkpoint`：在已结算的阶段屏障保存原生 SB3 模型、optimizer、off-policy replay、噪声、NumPy/Python/Torch RNG、归一化统计、每环境 seed schedule 和外部 L3 阶段 token。使用 staging、同卷原子重命名、SHA-256 校验和 `COMPLETED` 标记。拒绝跨指纹恢复与非受信任反序列化。
- `rles_rl_sb3.actor`：4 算法的独立 Actor 网络层映射、逐项 shape/dtype 校验、失败回滚、transfer parity、独立 BC 蒸馏、仅 Actor 的 L0 推理工件；最终 Hybrid 验收由 L5 和用户决定。

**严格无关内容**：不实现 Nautilus engine、订单、funding、Risk、Execution、WFO、ES、L5 CanonicalActor 生成或 L6 资源调度；不提供交易连接或密钥。

## 2. 公共接口

| 对象 / 方法 | 职责与输入约束 |
|---|---|
| `RLConfig.parse(mapping)` | 严格 schema；四算法 `hyperparameters` 使用独立白名单。|
| `PolicySpec` | 观测维度和两个 L1 schema hash、动作维度/边界、激活/隐藏层。|
| `TransitionFingerprint` | env、数据、feature、feature scaler、observation、action、action transform、execution、hard/soft risk、reward、normalization、engine、adapter、fold、terminal 的 SHA-256 整体指纹。|
| `SeedSpec` | 注入 L1 SeedTree 的 algorithm/env/noise/replay 种子。|
| `provider(config, env_factory, gate_authorizer)` | 插件工厂；无默认绕过 Q0/Q3 的路径。|
| `RLProvider.create()` | 验证 gate、设备、每个 Env 转移指纹；初始化原生 SB3 模型。|
| `.train(total_timesteps, resumed=False, callback=None)` | 同步采样预算整数对齐；每次环境 step 检查冻结指纹；返回 `Performance`。|
| `.predict(raw_observation, deterministic=True)` | float32 `[N, F]` 输入；应用 L4 的 VecNormalize（若启用），输出 `[N,1]`。|
| `.evaluate(episodes, max_steps)` | 新建独立评估环境，冻结 normalization，仅以确定性策略运行；返回 Gym reward/episode length，不声称是账户收益率。|
| `.save(destination, boundary, attempt_id)` | L1 阶段边界 checkpoint；禁止覆盖现存路径，写出完成标记。|
| `.load(source, trusted_source=True, restored_boundary=...)` | 先调用 env reset 让 L3 从注入的 EpisodeReplay 重建保存边界，再校验文件 hash、配方/环境/数值运行时、L3 token+边界原始观测+SB3 last obs；恢复每环境 seed schedule 与模型/optimizer/replay/RNG。|
| `.mapping()` | 显式的逐算法 LayerBinding、激活顺序、输出函数、dtype/shape、action scaling、签名。|
| `.transfer(weights, actor_signature, policy_observations, reference_actions, tolerance, source_actor_signature=None)` | 仅空白 learner，100% 层覆盖、原始 dtype/shape、动作 parity 超差并回滚。`actor_signature` 是 L4 算法专属 destination mapping 签名；可选 `source_actor_signature` 单独标识 L5 canonical source。通过 parity 后，报告携带 source 签名；有结构错误仍抛 `TransferBlocked`，只有结构/权重已验证后的动作不一致抛 `ActorActionParityMismatch`，供 L5 明确选择 Train-inner 蒸馏。|
| `.distill(policy_observations, reference_actions, epochs, batch_size, learning_rate, tolerance)` | 监督行为克隆初始化；仅返回 imitation error/通过标记，绝不宣称 100% 权重继承。|
| `.parity(...)` | L5 固定观测集 action error 分布（max/mean/p50/p95）。|
| `.export_inference(path, fixed_raw_observations)` | 只导出推理 Actor 的权重、Schema、normalizer、fingerprint，不导出 critic/replay。|
| `InferenceActor(path).predict(obs)` | 默认先校验 L0 完整性，`allow_pickle=False`；执行冻结 MLP 及对应 clip/tanh 映射。|
| `.performance` | 获取最近一次训练 samples/s、updates/s、RSS/VRAM（依赖可用性）。|
| `.close()` | 释放本线控制的向量环境。|

L3 `env_factory(rank, env_seed, phase)` 必须在 `phase="train"` 与 `"eval"` 返回彼此独立、同一冻结 Schema 的 `gymnasium.Env`；环境必须公开当前 **动态** `rles_transition_hash` 与 `rles_boundary_token`。为了签核后的无重置阶段恢复，L3 还须提供只读 `rles_boundary_observation`，返回 token 对应的**当前原始 float32 观测**（单 Env shape=`(observation_dim,)`）。L4 对各 Env 与 checkpoint 中的观测做严格值比较；没有该接口时拒绝 `.save/.load`，不会生成或采用临时补丁假装已恢复。前置 Q0、Q3、L1-013、L3-020/021、L5-003 的批准由注入的 `gate_authorizer(required_gate_ids, config) -> bool` 检查；L4 不自行签发门禁。

## 3. 算法语义与参数隔离

- **PPO**：SB3 原生 `n_steps × n_envs` rollout、policy/value 双头、`log_std`、原生 optimizer；`batch_size` 必须整除完整 rollout；确定性 Actor mean 由 SB3 clip 至动作范围。迁移时不加载 value 网络、log_std、optimizer。
- **SAC**：SB3 原生 stochastic squashed Gaussian、双 Q、target networks、entropy 和 replay；迁移仅 mean body/head，log_std、Q、target、replay 原生新建；禁用 gSDE。`noise_sigma` 在 SAC 不生效。
- **TD3**：SB3 原生 deterministic actor、target actor、双 Q/targets、delayed actor update、target smoothing noise/replay；专属 `policy_delay / target_policy_noise / target_noise_clip` 仅 TD3 允许配置。
- **DDPG**：SB3 的 `DDPG` 类独立实例，single Q、无 TD3 target policy smoothing、每步 actor update，使用原生 action-noise；拒绝所有 TD3 专属训练参数。虽然 SB3 内部复用了 TD3 实现，但不会偷换为 TD3 实例。

**迁移输出差异**：PPO 最终 mean clip；SAC 的 `mu` 后 tanh；TD3/DDPG 的 deterministic actor 内置 tanh。v0.4.0 L5 CanonicalActor 指定 tanh：PPO 权重严格拷贝后未必达到动作 parity，必须 **BLOCK HYBRID** 或经用户批准另立 ICR；绝不可强行把 PPO 改为 tanh policy 或改环境动作变换掩盖差异。L4 先独立核对 algorithm-specific destination mapping；L5 source signature 只在权重验证和动作 parity 通过后作为 lineage 身份回报，两者不得直接比较或互相替代。

## 4. Checkpoint、Replay 与可信度

- Off-policy buffer 单独保存；每次恢复时 `TransitionFingerprint` 中的 Execution、HardRisk、SoftRisk、Action Transform、Data、Reward、Normalization、Engine 等任意一项变更均产生 `ReplayMismatch`，不能沿用旧 buffer。运行期间每个 VecEnv step 同样检查 L3 当前 fingerprint，未通过即异常终止，不写入下一 replay。
- L1 仅承诺 **阶段边界**；不承诺 rollout 中途、中途订单/撮合 engine 内部状态完整恢复，也不承诺跨 Python/OS/CPU/GPU 数值环境 bitwise identical。stage.v3 严格拒绝数值运行时不一致，并恢复后续 episode 的 seed 游标。
- `Boundary` 要求 `step_cursor`、与 `n_envs` 一一对应的不透明 L3 恢复 token、无在途订单、无未完成环境步。`step_cursor` 定义为该 Run 累计**同步向量步数**，满足 `step_cursor × n_envs = SB3 num_timesteps`。PPO 还必须满足完整 `n_steps × n_envs` rollout 边界。检查 L3 `rles_boundary_observation`、SB3 `_last_obs`、`_last_original_obs`（off-policy）、`_last_episode_starts`、训练归一化值严格一致，才允许 `force_reset=False` 恢复。L3 需把 restored Engine/queue 与这些字段正确对齐，ICR 未批准前不得认定 resume 已通过集成验收。
- SB3 zip 与 replay 的反序列化涉及 `cloudpickle`/pickle。SHA-256 防止无意损坏，**不能证明来源可信**；只能加载可信本地生成、由外部签名/访问控制确认的 artifact；API 要求显式 `trusted_source=True`。无外网、无密钥研究沙箱运行。
- Stage resume 是否满足与 fresh-run 对照，必须在实际 L3 reset / dataset cursor / SB3 版本锁环境中复测。L4 无权提供虚构的 L3 engine snapshot。

## 5. 开发任务映射与用户审核点

| Task | 模块实现与验收条件（不代表 Gate PASS）|
|---|---|
| L4-001/002 | Provider 接口、严格四算法 RLConfig/PolicySpec；需要 Q3 签核。|
| L4-003/004/005/006 | 原生 4 算法独立创建/learn/predict/save/load/evaluate 路径；待真实 SB3 train 证据。|
| L4-007 | 单环境 DummyVecEnv，多原生环境使用 spawn 隔离及 L3 同步 deadline barrier；只重置已结束 worker。CPU 合成与原生双环境 PPO 已联调。|
| L4-008 | 观测 normalization 训练/评估生命周期隔离；reward norm 关闭。待 L3-020 确认。|
| L4-009 | 冻结 transition hash，逐步漂移检测，off-policy replay 恢复拒绝。待 L3 动态 hash 支持。|
| L4-010 | 算法/env/noise/replay 种子记录与 RNG 快照；待 L1 SeedTree 契约冻结。|
| L4-011 | L0 actor-only、normalizer、schema、固定观测动作 parity；上游 scaler 以专用 feature_scaler SHA-256 引用。|
| L4-012 | L1 阶段模型/optimizer/replay/RNG/stage token 的打包恢复；待 L3 boundary 真恢复对照。|
| L4-013/014/015 | 精确 Actor destination、失败回滚、动作误差统计已实现；新增 source/destination signature bridge，SAC/TD3/DDPG DIRECT 集成回归通过。L5-003 正式签名/归一化 ICR 仍待签，PPO clip/tanh 行为仍需 BLOCK 或独立批准的蒸馏路线。|
| L4-016 | BC 蒸馏初始化和独立 imitation error；不混同无损 transfer。|
| L4-017 | 性能指标查询；最终统计由 L6 BenchmarkRunner 调用。|
| L4-018 | 4 RL 的 Q4 Evidence 汇总由用户/总集成线执行；**禁止**进入本源码交付包。|

用户审核点 HR-L4-1～HR-L4-4 涉及冻结环境一致性、Replay 和 normalization 安全、逐算法迁移、四算法独立 train/save/resume/evaluate 的真实审计。没有这些记录和最终签核，L4 整体只能作为**待集成的生产源码候选**。

## 6. 本线配置及依赖

- `RLConfig.algorithm`: `PPO | SAC | TD3 | DDPG`；`device`: `cpu | cuda`（由 L6 在 run 前决策；本线不处理 `auto`）；`normalize_observation`: bool；`normalize_reward`: 固定 false；`noise_sigma`: TD3/DDPG normal action noise 标准差；`hyperparameters`: 按四算法独立白名单严格解析。
- 通用超参数：`learning_rate`, `gamma`。PPO：`n_steps`, `batch_size`, `n_epochs`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`, `target_kl`。SAC：`buffer_size`, `learning_starts`, `batch_size`, `tau`, `train_freq`, `gradient_steps`, `ent_coef`, `target_entropy`。TD3：`buffer_size`, `learning_starts`, `batch_size`, `tau`, `train_freq`, `gradient_steps`, `policy_delay`, `target_policy_noise`, `target_noise_clip`。DDPG：同 Off-policy 公共字段，但没有 TD3 专属字段。
- 依赖候选：CPython 3.14.x、Stable-Baselines3 2.9.0、Gymnasium 1.2～1.3、PyTorch 2.14.x、NumPy 2.x；研究文档要求的 3.14.7 只允许受控无密钥环境。实际各 OS/CPU/GPU 的 wheel/ABI 需要 G0 核验。
- 可选：`psutil` 提供 RSS；不存在时 `cpu_rss_bytes=None`，不得填假值。若需报告网络外部 scaler 文件，由 L2 原所有者/ArtifactStore 提供，L4 不复制其他开发线源码。

## 7. Interface Change Requests（必须由用户批准）

- **ICR-L4-001 / L3**：确认 `rles_transition_hash` 动态、只读、按当前环境真实 action transform / execution / risk 等字段实时计算；L4 经 `VecEnv.get_attr` 在开始/每次 step/评估/保存及恢复前验证，不可复用静态旧值。
- **ICR-L4-002 / L3 + L1**：确认 `rles_boundary_token`、stage step cursor、终止结算、还原后 `reset()` 起点、无在途订单的相互关系；证明分阶段 resume 不携带其他 fold replay、position、挂单，也不会重复执行上次订单。此 ICR 未签前 L1 恢复 Gate 阻断。
- **ICR-L4-003 / L1**：确认 `SeedTree` 的输出结构（algorithm / n 个 Env / noise / replay）及 `gate_authorizer` 的可信验签契约；正式适配必须由 L1/总集成提供，不允许 L4 私自定义 L1 公共 DTO。
- **ICR-L4-004 / L5**：冻结 `CanonicalActor` source signature、layer ordering、float32 weight namespace、activation/output squash/scaling、固定 observations parity 领域（raw 还是 normalized），尤其 PPO clip/tanh 数学差异；不一致必须拒绝 Hybrid 或批准独立蒸馏路径。
- **ICR-L4-005 / L6**：定义 BenchmarkRunner 读取 `Performance`、CPU RSS 未知值、GPU 内存峰值以及 stage event 接口；当前只有生产 metric hook。

## 8. 已知限制与变更文件

- 目标系统的 L1/L3/L5 实际源代码和签核证据未提供；本包没有代造它们，也不能宣称实际 Q4-RL 验证成功。
- 执行环境无 SB3/Gymnasium 且无法联网安装，无法做四算法原生训练、训练-恢复 parity 或 cp314 CPU/CUDA 系统验证；这些必须在用户的批准运行时另行完成。静态源码审查无法替代上述实验。
- Stage Resume 在加载时显式调用一次 `VecEnv.reset()`，要求注入的 L3 factory 把对应 `EpisodeReplay` 用于重建原生边界；随后严格核对 token、原始观测和 SB3 last obs。stage.v3 额外保存并恢复每个 Env 的确定性 episode seed 游标。L3 仍须证明 Engine token 能精确还原且不跨 fold；未解决不得宣称完整端到端 exact resume。
- 在 v0.4.0 现有扁平观测 MLP 范围外的 Dict/CNN、历史多币组合、Hedge Mode、L2 mid-step exact resume、全流程自动实盘均不在本线承诺范围。
- `src/rles_rl_sb3/{__init__,errors,config,envs,normalize,noise,checkpoint,actor,provider}.py`：L4 生产源码；`pyproject.toml`：独立包的依赖及 entry point；设计契约保持 v0.4.0，发布包版本升级为 v0.4.1。本说明是唯一附属文档。

**用户下一步**：提供已签核 Q0/Q3、L1/L3/L5 冻结接口或批准上述 ICR 后，开展 L4-003～L4-018 的真实环境验收；用户决定最终合并、发布与 tag。


## 9. v0.4.1 加固与 ICR（后续开发）

本次修改均在 `rles-rl-sb3` 所有权内，不修改 L1/L2/L3/L5/L6 源码、不修改 SB3 算法更新公式。

| 生产文件 | 功能变更 |
|---|---|
| `config.py` | 固定 DTO 类型及 tuple 字段，封闭四算法未知参数；不可变超参数映射、严格 dtype/布尔检查、PPO/SAC 无效噪声改值阻断 |
| `envs.py` | 只读 L3 stage 屏障观测证明接口；spawn VecEnv 与按 episode index 派生的可恢复 seed schedule |
| `provider.py` | CPython/SB3/Torch/Gymnasium runtime gate；恢复前显式触发 L3 replay reset；阶段恢复后不许重新开始；失败 attempt 污染阻断、环境清理与输出边界校验 |
| `normalize.py` | 拒绝 NaN/Inf/非正归一化 epsilon 或 clip_obs |
| `checkpoint.py` | stage.v3：SB3 与 L3 连续性证据、`load(force_reset=False)`、设备/依赖数值 profile、replay/RNG/环境 seed 游标恢复 |
| `actor.py` | float32 parity 输入硬校验；精确权重迁移报告含数量/签名/方法；蒸馏单独标注；L0 policy/schema/scaler 哈希一致性 |
| `pyproject.toml` | 生产包 v0.4.1，CPython 3.14 小版本基线 |

**新增强制接口验收**：ICR-L4-002 扩展为三个同时成立的条件：（a）L3 当前/恢复 `rles_boundary_observation` 原始观测不可变、shape/dtype 确定；（b）`rles_boundary_token` 与 obs 对应同一已结算决策屏障且可无条件准确恢复 L3 引擎及订单队列；（c）stage.step_cursor 是累计**向量决策次数**而非单环境 episode 内的步数。未经用户签核，不允许使用旧 replay 宣称恢复通过。

**显式版本门禁**：训练/载入需要 G0 验证的 CPython 3.14.x + SB3 2.9.0 + PyTorch 2.14.x + Gymnasium 1.2.x/1.3.x。模型 checkpoint 的 Python patch、OS、CPU 架构、NumPy、Gymnasium、Torch/CUDA runtime 与 GPU 设备名必须匹配；旧 stage.v1/v2 / 新 stage.v3 不互相兼容，不能通过 `strict=False` 降级。

**尚未通过的 Gate**：Q0/Q3 尚无用户签核与冻结的 L3 Engine/Gym 代码；L1/SeedTree 与 L5 E1 `CanonicalActor` 真实格式亦需批准 ICR。没有真实 SB3 2.9.0 和 CPython 3.14.7 环境，当前仅可声明实现候选及隔离验证，不可声明四算法实际训练、断点恢复、跨设备数值复现或 Q4-RL PASS。

# Windows CUDA 环境验证记录

日期：2026-10-16。范围：项目 Windows `.venv`、RTX 5070 Laptop GPU、项目 RL learner CUDA 路径。

## 发现与处理

机器安装了 NVIDIA RTX 5070 Laptop GPU，驱动 582.05，驱动报告 CUDA 13.0。项目虚拟环境之前装的是 `torch 2.14.1+cpu`，所以 `torch.cuda.is_available()` 为 false，4 个 `gpu` 标记用例都跳过。仓库已经提供 Windows CUDA 锁 `requirements/cuda.lock`，其中 PyTorch 是 `2.14.1+cu130`。问题是当前虚拟环境选用了 CPU 构建，不是显卡或驱动不可用。

已在当前 `.venv` 执行仓库锁定的安装命令：

```powershell
uv pip install --python .venv/Scripts/python.exe --require-hashes --torch-backend cu130 -r requirements/cuda.lock
```

锁文件没有改动。`.venv` 被 Git 忽略，因此新环境重建后需按 CUDA 锁重复安装；如果运行 CPU 专用测试环境，仍可使用 `requirements/cpu.lock`。

## 验证结果

- `torch 2.14.1+cu130`，PyTorch CUDA 构建 13.0，`torch.cuda.is_available() == True`，检测到 1 张卡。
- 设备为 `NVIDIA GeForce RTX 5070 Laptop GPU`，compute capability **12.0**，显存约 **8 GiB**。PyTorch 支持架构列表包含 `sm_120`。
- 在 GPU 上完成 1024×1024 CUDA 张量矩阵乘法并同步，证明 CUDA kernel 可以实际执行。
- 项目应用入口以 `rl_device="cuda"` 跑完 PPO 4 步合成行情训练；结果明确记录 `rl_device=cuda`，没有静默回退 CPU。
- `tests/test_cuda_smoke.py`：**4 passed**，PPO、SAC、TD3、DDPG 均完成 GPU 训练、保存、加载及推理。
- 依赖锁安装输出：PyTorch CPU 构建卸载，CUDA 构建安装；锁定依赖安装成功。

首次 pytest 尝试的 CUDA 运算和训练均已成功，但保存模型时 Python `TemporaryDirectory` 采用了 Codex 沙箱无写权限的 Windows 系统 Temp，造成 4 项保存失败。将 `TEMP`/`TMP` 临时指向仓库 `.tmp` 后四项全过。这是本次受限执行环境的临时目录权限限制，不是 CUDA 或模型保存代码问题。复验命令：

```powershell
New-Item -ItemType Directory -Force .tmp/cuda-test-tmp | Out-Null
$env:TEMP = (Resolve-Path .tmp/cuda-test-tmp).Path
$env:TMP = $env:TEMP
.venv/Scripts/python.exe -m pytest tests/test_cuda_smoke.py -q `
  --basetemp=.tmp/cuda-smoke-NEW-RUN --tb=short
```

PPO 的小型 MLP 训练会提示 SB3：这类策略通常不适合用 GPU，显卡虽可工作，但未必比 CPU 快。当前 Nautilus 行情和账户环境仍在 CPU；本次只证明 learner 可用 GPU，不代表整条历史回放会因此提速。选择运行拓扑应通过项目已有端到端基准，而不是只看 GPU 使用率。

## 日常 CUDA 诊断

```powershell
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
.venv/Scripts/python.exe qa/smoke/imports.py
$env:UV_CACHE_DIR = "$PWD/.tmp/uv-cache"
uv pip check --python .venv/Scripts/python.exe
```

项目训练配置用 `"rl_device": "cuda"` 显式请求 CUDA；缺少 CUDA 构建或设备时会在设备解析阶段失败，不会偷偷改用 CPU。训练结果中应检查 `rl_device` 字段。

本记录证明本机当前环境能够使用 CUDA，不构成跨机器驱动验证、CUDA 并行吞吐基准或盈利能力结论。

## 2026-10-17 continuation verification

独立环境 `.venv-cuda` 当前安装 `torch 2.14.1+cu130`，RTX 5070 可见。`tests/test_cuda_smoke.py` 本轮 **4 passed**（PPO、SAC、TD3、DDPG，训练/保存/加载/推理），保留 Stable-Baselines3 对 MLP PPO 在 GPU 上效率较低的提示。另将实际 RLES A0 `train --train-only` 64 步命令显式设为 CUDA，并在 Windows Job 6 GiB 进程树限制下运行成功；仅使用现有 CI 测试夹具，不执行回测或策略评价。完整监督器 JSON、独立验证摘要与 bundle 在 `.artifacts/audit-development-20261017/n04-supervised-train/`。该结果证明当前本机端到端训练可选用 CUDA，不代表 GPU 提速、VRAM 限额或其他机器认证。

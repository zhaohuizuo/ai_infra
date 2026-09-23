# AI Infra 学习实验室

这是一套面向转岗的 16 周学习工程，重点是 GPU 算子与 LLM 推理。默认学习环境：

- 本地：Apple Silicon macOS，用于编辑代码和整理报告
- 远程：Linux + NVIDIA RTX A6000 48GB，用于 CUDA/Triton 实验
- 基础：Python 熟悉，PyTorch/C++ 入门
- 投入：每周约 10 小时

## 最终产出

1. `gpu-kernel-lab`：CUDA reduction、transpose、softmax、GEMM 与 profiling 报告
2. `pytorch-fused-ops`：Triton/CUDA fused RMSNorm 和 PyTorch Custom Op
3. `mini-llm-engine`：KV Cache、动态 batching、推理指标及 vLLM 对比

完整安排见 [ROADMAP.md](ROADMAP.md)，每周状态记录在 [PROGRESS.md](PROGRESS.md)。

## 目录

```text
.
├── common/                         # 所有实验共用的测量工具
├── labs/
│   └── week01_pytorch_basics/      # 第一周实验
├── scripts/
│   └── check_gpu_env.py            # A6000 环境验收
├── ROADMAP.md
└── PROGRESS.md
```

后续周的目录在开始对应阶段时创建，避免一次生成大量没有做过的模板。

## A6000 环境初始化

建议使用 Python 3.11。PyTorch 应根据服务器驱动和 CUDA 环境，从 PyTorch 官方安装页面选择对应命令，不在仓库里固定某个 CUDA wheel。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 在这里安装匹配服务器 CUDA 环境的 PyTorch
python -m pip install -r requirements-dev.txt
python scripts/check_gpu_env.py
```

环境检查通过后运行第一周实验：

```bash
python -m labs.week01_pytorch_basics.tensor_layout --device cuda
python -m labs.week01_pytorch_basics.layout_benchmark
python -m labs.week01_pytorch_basics.gemm_benchmark
```

## 实验纪律

每个性能结论都必须同时具备：

- 正确性对照
- 预热和 GPU 同步
- 明确的 shape、dtype、设备与软件版本
- 至少 100 次测量的中位数和 P90
- 延迟之外的有效带宽或 TFLOPS
- profiler 证据或可以验证的性能假设

不要只记录“快了多少”，还要解释减少了哪些访存、同步、launch 或计算开销。

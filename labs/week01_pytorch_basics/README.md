# 第 1 周：PyTorch Tensor 与可信测量

本周目标不是记 API，而是建立后面所有算子实验共用的测量基础。

## 每日计划

### Day 1：远程环境，约 1 小时

```bash
python scripts/check_gpu_env.py
```

把输出复制到 [NOTES.md](NOTES.md)。确认 GPU 是 A6000、CUDA smoke test 通过，并记录 PyTorch、CUDA、Triton 和驱动版本。

### Day 2：Tensor 语义，约 1.5 小时

```bash
python -m labs.week01_pytorch_basics.tensor_layout --device cuda
```

逐行解释 shape、stride、storage offset、contiguous。修改 transpose 维度并预测输出，再运行验证。

### Day 3：CUDA 测量，约 1 小时

阅读 `common/benchmark.py`，回答：

1. 为什么需要 warmup？
2. 为什么调用 `torch.cuda.synchronize()`？
3. 为什么中位数通常比平均数更稳健？
4. CUDA Event 和 CPU wall clock 分别测量什么？

### Day 4：Layout 性能，约 1.5 小时

```bash
python -m labs.week01_pytorch_basics.layout_benchmark
python -m labs.week01_pytorch_basics.layout_benchmark --rows 8192 --cols 2048
```

先写下性能预测，再运行。解释 stride 如何改变相邻线程的访存位置。

### Day 5：dtype 与 GEMM，约 1.5 小时

```bash
python -m labs.week01_pytorch_basics.gemm_benchmark
python -m labs.week01_pytorch_basics.gemm_benchmark --sizes 1024 2048 4096
```

比较 FP32、FP16、BF16。不要只看延迟，同时比较 TFLOPS。

### Day 6：第一次 timeline，约 2.5 小时

使用服务器上的 Nsight Systems：

```bash
mkdir -p profiles
nsys profile -o profiles/week01_gemm \
  python -m labs.week01_pytorch_basics.gemm_benchmark --sizes 2048
```

找到 warmup、矩阵乘 kernel 和同步位置。记录 kernel 名称与大致持续时间。

### Day 7：复盘，约 1 小时

完成 [NOTES.md](NOTES.md)，更新仓库根目录的 `PROGRESS.md`。

## 验收标准

- [ ] 环境检查全部通过
- [ ] 能根据 shape 手算 stride
- [ ] 能解释 `view` 与 `reshape` 在非连续 Tensor 上的差异
- [ ] benchmark 包含 warmup、CUDA Event 和同步
- [ ] 报告 FP32/FP16/BF16 GEMM 的延迟与 TFLOPS
- [ ] 有一张 Nsight Systems timeline 或对应文字记录
- [ ] `NOTES.md` 中不存在没有尝试回答的问题

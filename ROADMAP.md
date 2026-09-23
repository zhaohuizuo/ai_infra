# 16 周路线

默认每周 10 小时：概念 2 小时、编码 5 小时、profiling 与报告 3 小时。

| 周 | 主题 | 必做实验 | 阶段验收 |
|---|---|---|---|
| 1 | PyTorch Tensor 与测量 | stride/storage、layout、FP32/FP16/BF16 GEMM | 能写可信的 CUDA benchmark |
| 2 | 推理数学 | Softmax、RMSNorm、SwiGLU、Attention、RoPE | 能标注 shape、FLOPs 和访存量 |
| 3 | C++ 最小基础 | 指针、引用、RAII、模板、CMake | 能编译和调试多文件 C++ 项目 |
| 4 | CUDA 编程模型 | Vector Add、ReLU、二维 indexing | 理解 grid/block/warp 和异步执行 |
| 5 | GPU 访存 | naive/coalesced/shared transpose | 能解释合并访问与 bank conflict |
| 6 | Reduction/Softmax | shared-memory、warp-shuffle 版本 | 会用 Nsight Compute 定位瓶颈 |
| 7 | GEMM | naive、tiled、shared-memory GEMM | 理解 tiling、复用与计算强度 |
| 8 | Triton | Vector Add、Softmax、RMSNorm、Matmul | 理解 block-based 编程模型 |
| 9 | 算子融合 | residual + RMSNorm、可选 fused SwiGLU | 完成第一个作品集项目 |
| 10 | PyTorch Custom Op | schema、CUDA、FakeTensor、opcheck、compile | 自定义算子可接入真实模型 |
| 11 | Transformer Decoder | GQA、RoPE、causal attention、sampling | 从零跑通自回归生成 |
| 12 | KV Cache | prefill/decode、带/不带 cache 对比 | 会计算 KV 容量与 TTFT/TPOT |
| 13 | 迷你推理引擎 | 请求队列、scheduler、动态 batch、streaming | 完成第二个作品集项目 |
| 14 | vLLM 源码 | 请求到 scheduler/runner/kernel 的调用链 | 输出一次 decode step 分析文档 |
| 15 | A6000 性能实验 | HF eager/compile/vLLM、并发和长度矩阵 | 形成完整性能报告 |
| 16 | 转岗交付 | 整理三个项目、简历、面试题 | 项目可复现、结论有数据支撑 |

## 阶段门槛

### 第 1～3 周：基础门槛

- 能解释 shape、stride、storage 和 contiguous
- 不使用 `time.time()` 直接测异步 CUDA kernel
- 能阅读简单 C++/CUDA 代码并独立编译
- 能手写数值稳定的 Softmax 和 RMSNorm

### 第 4～7 周：CUDA 门槛

- 能从 Tensor shape 推导 grid/block
- 能判断访存是否合并
- 能解释 occupancy、register pressure、warp stall
- 能用 Roofline 判断 memory-bound 或 compute-bound

### 第 8～10 周：算子工程门槛

- 同一算子具有 PyTorch、Triton/CUDA、测试、benchmark
- 覆盖多种 shape、FP16/BF16 和边界条件
- 自定义算子可被 PyTorch dispatcher 和 `torch.compile` 正确处理

### 第 11～15 周：推理门槛

- 能解释 prefill 与 decode 的性能特征
- 能手算模型权重与 KV Cache 显存
- 能解释 continuous batching、PagedAttention、prefix cache
- 能用 TTFT、TPOT、吞吐量、P50/P99 描述服务性能

## 暂缓内容

在完成第 12 周前，不把时间投入到：手写 PTX/SASS、CUTLASS/CuTe 深层模板、RDMA、大规模训练、复杂多机调度。A6000 不支持 Hopper FP8 路线，因此 FP8 只学习概念，不作为主实验。


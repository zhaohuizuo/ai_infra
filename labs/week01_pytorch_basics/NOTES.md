# 第 1 周实验记录

## 环境

- 日期：
- GPU：
- Driver：
- CUDA runtime/toolkit：
- PyTorch：
- Triton：

## Tensor 语义

1. shape、stride 和 storage 分别描述什么？
2. 为什么 transpose 通常不复制数据？
3. `view` 为什么可能在 transpose 之后失败？
4. `reshape` 在什么情况下会产生拷贝？如何验证？

## 测量方法

1. 第一次执行为什么通常更慢？
2. CUDA 是异步的，对计时有什么影响？
3. 为什么记录 median、P90 和 minimum？它们分别能反映什么？

## Layout 实验

| Shape | dtype | contiguous GB/s | strided GB/s | 比值 |
|---|---|---:|---:|---:|
|  |  |  |  |  |

性能预测：

实际结果：

解释：

## GEMM 实验

| M=N=K | dtype | median ms | P90 ms | TFLOPS |
|---:|---|---:|---:|---:|
|  |  |  |  |  |

FP16/BF16 为什么可能比 FP32 更快？

shape 对吞吐率有什么影响？

## Profiler 观察

- kernel 名称：
- kernel 持续时间：
- CPU launch 与 GPU execution 是否重叠：
- 一个没有解释清楚的现象：

## 本周结论

- 最重要的三个认识：
- 一个被实验推翻的猜测：
- 下周需要补充的知识：


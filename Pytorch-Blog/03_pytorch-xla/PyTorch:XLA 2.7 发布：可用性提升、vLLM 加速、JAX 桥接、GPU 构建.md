>标题：PyTorch/XLA 2.7 发布：可用性提升、vLLM 加速、JAX 桥接、GPU 构建
>
>作者：Pei Zhang, Chris Jones
>
>日期：2025-05-13

# PyTorch/XLA 2.7 发布：可用性提升、vLLM 加速、JAX 桥接、GPU 构建

PyTorch/XLA 是一个 Python 包，它利用 XLA 深度学习编译器，使 PyTorch 的深度学习任务能够在多种硬件后端上运行，包括 Google Cloud TPU、GPU 以及 AWS Inferentia/Trainium。PyTorch/XLA 团队一直在努力为使用 TPU/GPU 和 XLA 后端的研究人员与开发者带来新的功能。在本次更新中，我们对框架进行了大量新增和改进。

其中一些亮点包括：

- 可用性改进
- 与 JAX 操作的实验性桥接
- 基于 Pallas 的 Ragged Paged Attention 新内核，为 [vLLM TPU](https://docs.vllm.ai/en/v0.5.5/getting_started/tpu-installation.html) 上的进一步优化铺平道路

这些功能、Bug 修复以及其他细节已经在[发布说明](https://github.com/pytorch/xla/releases)中进行了详细列出。下面让我们深入了解这些重点更新内容！

## 可用性改进

开发者现在可以通过标记希望分析的精确代码区域，更有针对性地对关键代码段进行性能分析。例如：

```python
server = xp.start_server(8001)
xp.start_trace(profiling_dir)
# Run some computation
...
xp.stop_trace()
```

PyTorch/XLA 2.7 还引入了一个新的 API，用于查询已缓存的编译图数量，有助于在生产环境的推理或训练中检测意外的编译行为。另一个改进是优化了主机到设备的数据传输，通过避免不必要的张量复制来提升整体性能。

## PyTorch/XLA 中的 JAX 桥接（原型）

我们正在尝试将 JAX 操作**直接集成到 PyTorch/XLA 的计算图**中，作为在两个框架之间建立桥接的一种方式——这种方法允许用户在使用 XLA 运行的 PyTorch 模型中调用 JAX 函数。

在一个应用场景中，我们探索了从 PyTorch/XLA 调用 `jax.experimental.shard_alike` 函数。该函数在某些代码模式（例如 `scan`）中能改善张量的分片传播（sharding propagation），我们已将其作为编译器中 GSPMD（Generalized SPMD）工作流的一部分进行了集成。这个功能也已在 [torchprime](https://github.com/AI-Hypercomputer/torchprime) 项目中用于支持 [SplashAttention Pallas kernel](https://github.com/AI-Hypercomputer/torchprime/blob/b123c0cc157c28f32a0f6588f19e2d352d2a3617/torchprime/torch_xla_models/experimental/custom_kernel.py)。

```python
import torch_xla.core.xla_builder as xb
# Native function written in JAX
def jax_function(...):
  import jax
  ...
  return ...
res = xb.call_jax(...) </pre?
```

## 基于 Pallas 的 Ragged Paged Attention 内核

对于大语言模型而言，高效处理变长序列的注意力机制至关重要。此次新增的 **Ragged Paged Attention Pallas 内核**为 [vLLM TPU](https://docs.vllm.ai/en/v0.5.5/getting_started/tpu-installation.html) 上的推理带来了显著的性能和可用性提升。

本次更新引入了一个使用 [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) 自定义内核语言实现的专用内核，最终会被降阶（lower）为适用于 TPU 的 Mosaic 格式。它支持 **ragged（变长）**输入序列，并实现了**分页注意力（paged attention）模式**。以下是其主要特性：

- 支持预填充（prefill）与解码（decode）操作的混合执行：显著提升推理吞吐量。例如，对于 llama-3-8b 模型，相较于使用填充的多查询分页注意力实现，速度可提升最多达 5 倍。
- 无需 GMM（Grouped Matmul）元数据！元数据在内核内部即时动态生成，无需提前提供。这一优化可带来约 10% 的性能提升。
- 提供 CUDA Flash Attention 的等效实现：接口设计相似，同时支持分页注意力，便于用户无缝迁移。

我们正在与 **vLLM 社区**持续合作，进一步优化性能、扩展内核覆盖范围，并简化大规模 TPU 推理流程。

## GPU 构建功能回归

在 PyTorch/XLA 2.6 版本中，GPU 构建功能曾被暂停，但在 2.7 版本中我们重新启用了 GPU 持续集成（CI）。当前发布版本支持基于 CUDA 12.6 的 GPU 构建，这标志着 GPU 支持迈出了重要一步。

尽管本次版本中的 CUDA 支持仍处于实验阶段，我们计划在未来的版本中扩展对更多 CUDA 版本的支持。

## 参与贡献

欢迎大家访问 [GitHub](https://github.com/pytorch/xla) 查看最新更新。我们始终欢迎社区的反馈和贡献，期待你的积极参与！


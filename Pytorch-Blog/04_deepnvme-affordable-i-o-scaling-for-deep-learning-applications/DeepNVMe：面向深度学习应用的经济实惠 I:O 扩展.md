>标题：DeepNVMe：面向深度学习应用的经济实惠 I/O 扩展
>
>作者：Joe Mayer, Logan Adams, Olatunji Ruwase
>
>日期：2025-06-17

# 介绍

我们在2024年夏季推出了 [DeepNVMe](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/08-2024/README.md)，这是一套用于解决深度学习（DL）中 I/O 瓶颈的优化方案。DeepNVMe 通过利用本地 NVMe SSD、NVIDIA Magnum IO™️ GPUDirect® Storage（GDS）以及 Linux 异步 I/O（AIO）等存储创新技术，为受 I/O 限制的深度学习工作负载带来了显著的加速效果。在本次更新中，我们很高兴地宣布 DeepNVMe 在多个方面的改进：

* 将应用范围扩展到 FastPersist 模型检查点和 SGLang 推理，
* 通过升级 PCIe Gen4 到 Gen5 NVMe SSD 实现 I/O 性能的扩展
* 扩展对仅 CPU 环境、基于偏移的 I/O 操作和张量数据类型转换的支持。本文报告的结果适用于 DeepSpeed 版本 ≥ [0.17.1](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.17.1)。/

# 评估环境

我们的实验在 Azure [ND-H200-v5](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-h200-v5-series?tabs=sizebasic) 虚拟机上进行。主要的软件配置总结如下表。

| 软件    | 版本        |
| ------- | ----------- |
| Ubuntu  | 24.04.2     |
| PyTorch | 2.6.0       |
| CUDA    | 12.6        |
| SGLang  | 0.4.4.post4 |

# 解决深度学习的 I/O 瓶颈

我们使用 DeepNVMe 开发了 FastPersist 和 ZeRO-Inference，分别针对深度学习训练和推理中的 I/O 瓶颈。实验在单台虚拟机上进行，将可用的 NVMe SSD 合并为一个 RAID-0（即磁盘条带）卷，以利用聚合的读写带宽。由于 DeepNVMe 支持通过 CPU 缓冲区（即异步 I/O，AIO）或 NVIDIA GPUDirect Storage（即 GDS）卸载张量，我们分别报告了这两种模式下的结果。

## FastPersist：更快的模型检查点创建

虽然将模型检查点保存到持久存储对模型训练至关重要，但现有方法效率低下，成为主要瓶颈。我们开发了 FastPersist 来解决检查点性能挑战。[FastPersist](https://arxiv.org/abs/2406.13768) 通过三大关键技术使训练中的检查点开销几乎可以忽略不计：（i）DeepNVMe，（ii）数据并行，以及（iii）I/O 与计算重叠。

我们的目标是通过单进程微基准测试（可点击[这里](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/deepnvme/model_checkpoint)查看），展示 DeepNVMe 在 FastPersist 中的作用，该测试将模型检查点状态从 HBM 序列化到本地 NVMe。实验中，我们以流行的 PyTorch `torch.save()` 作为基线，并将 FastPersist 集成到 `torch.save()`，以简化使用和性能对比。

## 更快地将 PyTorch 模型保存到本地 NVMe 存储

我们测量了将 Phi-3-Mini 检查点状态从 HBM 序列化到本地 NVMe 存储的吞吐量。下图总结了结果。相比基线，FastPersist 的检查点速度显著提升。在 8x Gen5 NVMe 配置下，速度提升超过 20 倍。同时，FastPersist 在 8x Gen5 相比 4x Gen5 的 NVMe 带宽提升下也展现出良好的扩展性。

![img](images/01-image.avif)

FastPersist 显著加快了模型检查点保存到本地 NVMe 的速度。

# ZeRO-Inference：让生成式 AI 更加普及

[ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/README.md) 是一项技术，通过降低模型推理的 GPU 成本，实现了先进模型的普及化。ZeRO-Inference 通过将模型权重卸载到 DRAM 和 NVMe 存储，使得数百亿参数的大型模型能够在仅一块 GPU 上进行推理计算。ZeRO-Inference 设计用于离线或吞吐量导向的推理场景。在本文中，我们分享了 ZeRO-Inference 的两项更新：首先，我们将 ZeRO-Inference 集成到了 SGLang —— 一个先进的模型服务框架；其次，我们观察到 ZeRO-Inference 的性能随着最新 Azure SKU 中更快的 NVMe SSD 进行了提升。

## 通过 ZeRO-Inference 集成实现 SGLang 的普及

[SGLang](https://docs.sglang.ai/) 是一个面向大型语言模型（LLM）和视觉语言模型（VLM）的先进服务框架。我们将 ZeRO-Inference 集成到 SGLang，使得预算有限的用户也能使用 SGLang，并为现有用户提供了降低成本的选项。我们使用 SGLang 的[离线基准测试工具](https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_offline_throughput.py)，测量了在单台配备 NVMe 卸载的 H200 上运行 LLAMA3-70B 的生成吞吐量（LLAMA3-70B 无法在 141GB VRAM 中完整加载，必须卸载）。实验配置为提示长度 512，生成长度 32，批量大小 128。下图总结了 AIO 和 GDS 卸载两种模式下的结果。

![img](images/02-image.avif)

ZeRO-Inference 通过 NVMe 卸载提升了 SGLang 的推理性能，从而降低了硬件成本。

## 利用更快的 NVMe SSD 扩展 HF Transformer 生成能力

ZeRO-Inference 通过高效的模型卸载到 DRAM 或 NVMe，增强了 HF Transformer 的推理能力。我们此前在 Azure [NC_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic) 虚拟机上使用单 GPU 和四块 Gen4 NVMe [评估](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/08-2024/README.md#high-performance-offloading-via-nvme-scaling)了 LLAMA-3-70B 的生成性能，测试配置为提示长度512，输出32个 token，批量大小96。由于 NVMe 带宽成为主要瓶颈，我们在提供 Gen5 NVMe 的 Azure ND-H200-v5 上重复了实验。下图总结的结果显示，ZeRO-Inference 利用更高的 NVMe 带宽提升了生成速度。例如，使用 GDS 时，生成速度从四块 Gen4 NVMe 的每秒 7 个 token 提升至四块 Gen5 NVMe 的每秒 17 个 token，进一步提升至八块 Gen5 NVMe 的每秒 26 个 token。未使用 GDS 时也观察到了类似的提升。这些结果表明，通过增加 NVMe 带宽，ZeRO-Inference 的性能可以以经济高效的方式得到提升。

![img](images/03-image.avif)

ZeRO-Inference 利用可用的 NVMe 带宽，实现了 LLAMA-3-70B 生成性能的扩展。

# I/O 性能扩展

我们使用 `ds_io` 基准测试工具展示了 DeepNVMe 随可用 NVMe 带宽成比例地扩展 I/O 性能。这使用户能够通过增加数量或使用更快的 NVMe SSD，以较低成本加速受 I/O 限制的深度学习应用。在实验中，我们测量了 1GB 数据在 HBM 与 NVMe 之间传输时的读写带宽。评估内容包括从 PCIe Gen4 升级到 Gen5 以及从 4 块 SSD 扩展到 8 块 SSD。所有 SSD 被合并为一个 RAID-0（磁盘条带）卷。下图总结的结果显示，DeepNVMe 在这两个维度上均能扩展 I/O 性能。从 4x Gen4 SSD 升级到 4x Gen5 SSD，读取带宽从 10GB/s 提升至 27GB/s，写入带宽从 5GB/s 提升至 11GB/s；从 4x Gen5 扩展到 8x Gen5，读取带宽进一步提升至 48GB/s，写入带宽提升至 26GB/s。

![img](images/04-image.avif)

微基准测试显示 DeepNVMe 能够随着可用 NVMe 带宽扩展 I/O 性能。

# 拓宽适用范围

我们通过解除对硬件环境和 I/O 操作的限制，扩大了 DeepNVMe 的使用场景，具体说明如下。

## CPU-Only 环境

尽管 GPU（及类似加速器）在深度学习中占主导地位，但 CPU 仍用于推荐系统等重要机器学习任务中。然而，DeepNVMe 之前在纯 CPU 环境中无法使用，这是因为 DeepNVMe 依赖于 `torch.pin_memory()` 来分配页面锁定的 CPU 张量，而 `torch.pin_memory()` 在 CPU 版本的 `torch` 中不起作用，示例如下：

```python
>>> import torch
>>> torch.__version__
'2.6.0+cpu'
>>> x = torch.empty(1024).pin_memory()
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
RuntimeError: Cannot access accelerator device when none is available.
```

我们通过新增分配（`new_cpu_locked_tensor()`）和释放（`free_cpu_locked_tensor()`）页面锁定 CPU 张量的机制，使 DeepNVMe 在 CPU 环境中可用。以下代码演示了如何分配一个 pinned CPU 张量（`x`）：

```python
>>> import torch
>>> torch.__version__
'2.6.0+cpu'
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle()
>>> x = h.new_cpu_locked_tensor(1024, torch.Tensor())
>>> x.shape
torch.Size([1024])
>>> x.dtype
torch.float32
```

## 基于偏移的 I/O 操作

 此前，DeepNVMe 只能读写整个文件内容。现在我们改进了 DeepNVMe，支持从用户指定的偏移位置读写文件的部分内容。具体地，扩展了现有的读/写接口，允许传入用户指定的 `file offset` 参数（默认为 0），示例如下：

```pythpn
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> help(AsyncIOBuilder().load().aio_handle().pread)
Help on method pread in module async_io:

pread(...) method of async_io.aio_handle instance
pread(self: async_io.aio_handle, buffer: torch.Tensor, filename: str, validate: bool, async: bool, file_offset: int = 0) -> int
```

# 张量数据类型转换

在开发 FastPersist 时，我们需要以字节格式操作模型张量（通常是浮点数据类型），以提高 I/O 操作的性能和便利性。然而，我们找不到一种零拷贝（zero-copy）机制将任意数据类型的张量转换为字节类型（即 `torch.uint8`），于是决定自行实现该功能。该功能通过 `UtilsBuilder` 操作提供，下面的示例演示了如何将 `torch.bfloat16` 张量转换为 `torch.uint8`。需要注意的是，由于该功能是零拷贝的，`bf16_tensor` 和 `byte_tensor` 是同一数据的不同别名。

```python
>>> import torch
>>> from deepspeed.ops.op_builder import UtilsBuilder
>>> util_ops = UtilsBuilder().load()
>>> bf16_tensor = torch.zeros(1024, dtype=torch.bfloat16, device='cuda')
>>> bf16_tensor
tensor([0., 0., 0., ..., 0., 0., 0.], device='cuda:0', dtype=torch.bfloat16)
>>> byte_tensor = util_ops.cast_to_byte_tensor(bf16_tensor)
>>> byte_tensor
tensor([0, 0, 0, ..., 0, 0, 0], device='cuda:0', dtype=torch.uint8)
>>> bf16_tensor += 1.0
>>> bf16_tensor
tensor([1., 1., 1., ..., 1., 1., 1.], device='cuda:0', dtype=torch.bfloat16)
>>> byte_tensor
tensor([128, 63, 128, ..., 63, 128, 63], device='cuda:0', dtype=torch.uint8)
```

# 总结

本文介绍了我们对 DeepNVMe —— 一种加速深度学习应用的 I/O 优化技术 —— 的持续改进。我们公布了 DeepNVMe 在应用范围、I/O 性能扩展以及易用性等多个方面的提升。

# 致谢

本文所述工作由微软 DeepSpeed 团队的 Joe Mayer、Logan Adams 和 Olatunji Ruwase 完成。

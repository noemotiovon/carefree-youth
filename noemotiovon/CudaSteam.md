# Cuda Stream 概念

CUDA Stream（CUDA 流）是 **NVIDIA CUDA 编程模型**中的一个非常核心的概念，它代表了一组 **按顺序执行的 CUDA 操作队列**，也可以理解为 GPU 上的一个任务“通道”或“流水线”。

CUDA Stream 就是 GPU 上的任务调度通道，允许你更灵活地控制指令执行顺序，实现并发和异步。

# Cuda Stream 特点

* 每个 CUDA Stream 就像一个 **命令队列**，你可以把内核执行（kernel launches）、内存拷贝（memory transfers）等命令提交到某个 stream。

* 同一个 stream 中的操作是 **按顺序执行** 的。

* 不同 stream 中的操作是 **可能并发执行** 的（如果硬件支持，并且操作之间没有依赖）。

# Cuda Stream 用途

* **提升并发性**：内核执行、内存拷贝等可以重叠，充分利用 GPU 的计算和传输能力。

* **任务隔离**：多个任务可以通过独立 stream 隔离，互不干扰。

* **异步操作**：你可以使用 stream 让任务异步进行，而不是 CPU 等待 GPU 完成。

# 默认 stream：

- 不指定 stream 时，CUDA 使用默认 stream（又称为 **legacy default stream**）。
- 默认 stream 中的操作会与其他所有 stream 同步（取决于设备行为）。


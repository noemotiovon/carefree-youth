>论文地址：https://arxiv.org/pdf/2504.17577



TileLang 的目标是在**高性能（performance）**与**可组合性（composability）**之间取得平衡，从而为开发者提供既能灵活表达复杂数据流逻辑，又能接近手写 CUDA 代码性能的统一编程模型。

TileLang 的设计围绕以下三大目标展开：

1. **解耦数据流与调度（Decoupling Dataflow and Schedule）**
    传统编译器通常将数据流逻辑与优化调度逻辑紧密绑定，使得在不同硬件上复用 kernel 变得困难。TileLang 通过一组轻量化注解与可插拔调度原语，将数据流（计算逻辑）与执行策略（线程映射、流水线、tensorize 等）分离。
    这种解耦使得同一计算逻辑可以在多种硬件平台上（如 NVIDIA GPU、AMD GPU、Ascend NPU 等）通过不同调度策略实现高效执行。
2. **统一的 block-thread 范式（Unified Block-Thread Model）**
    TileLang 提供一种统一的块-线程编程模型，将硬件的线程层次结构抽象为可组合的语义单元。
    与 CUDA 或 Triton 的固定层次不同，TileLang 的 block、warp、thread 均可通过配置灵活定义，允许用户以声明式方式指定计算单元的映射关系。
3. **灵活的编译后端（Extensible Backend System）**
    TileLang 的中间表示（IR）可以映射到不同的后端，包括 CUDA、HIP、Metal、以及自定义硬件 DSL（如 TVM Script）。这种设计使其能够作为上层框架（如 PyTorch、vLLM）与底层硬件加速库之间的中间桥梁，方便 AI 系统开发与 kernel 优化。

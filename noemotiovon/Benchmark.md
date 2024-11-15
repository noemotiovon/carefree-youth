### 0 参考资料：

[vLLM官方文档](https://docs.vllm.ai/en/v0.6.0/performance_benchmark/benchmarks.html)

[vLLM性能测试结果](https://simon-mo-workspace.observablehq.cloud/vllm-dashboard-v0/perf)

[how-to-benchmark-vllm](https://www.substratus.ai/blog/how-to-benchmark-vllm)

### 1 基础概念

基准测试，也称之为性能测试，是一种用于衡量计算机系统，软件应用或硬件组件性能的测试方法。基准测试旨在通过运行一系列标准化的任务场景来测量系统的性能表现，从而帮助评估系统的各种指标，如响应时间、吞吐量、延迟、资源利用率等。

核心由3部分组成：数据集、工作负载、度量指标。

vLLM项目的benchenmark有三类：分别是Latency Test、Throughput Test、Serving Test。

#### 1.1 Latency Test（延迟测试）

**延迟测试**用来测量模型或系统从接收到请求到产生响应所需要的时间。这对于评估模型在实时推理场景下的响应速度尤为重要。

**测试目标：**

主要用于评估单个请求的响应时间，尤其在生成任务或推理任务中。

测试结果通常以 **毫秒 (ms)** 为单位，显示模型从输入到输出的总时间。

#### **2. Throughput Test（吞吐量测试）**

**吞吐量测试**用来测量系统在单位时间内可以处理的请求数量。这通常用于评估系统的处理能力和效率，特别是在大规模并发请求场景下。

**测试目标：**

评估在高负载下，系统每秒可以处理的推理请求或生成的 token 数量。

测试结果通常以 **requests per second (RPS)** 或 **tokens per second** 为单位。

#### **3. Serving Test（服务测试）**

**服务测试**主要用来测试系统在实际部署环境下的表现，评估系统在处理真实用户请求时的性能。通常结合 REST API 或其他在线推理接口进行测试。

**测试目标：**

模拟真实的用户请求，测试模型在生产环境中的表现。

评估系统的稳定性、响应时间，以及在持续请求负载下的性能表现。

#### 4. 常见指标

**Tput**: 吞吐量，表示模型在单位时间内能够处理的请求数量，表示模型在单位时间内能够处理的请求数量。

**Mean**：平均值。

**Medium**： 中位数。

**P99**：99%的数据都在哪个范围。

**Std**：标准差。

**TTFT**: 生成第一个 token 的时间，反应模型初始化/输入预处理的效率。

**Mean ITL**: 是生成连续两个 token 之间的延迟时间，表示模型在生成 token 的过程中每步处理速度较快，意味着流畅的文本生成。

**Input Tput**: 输入 token 的吞吐量，表示每秒处理的输入 token 数量，反应模型能够快速处理输入数据的能力。

**Output Tput**: 输出 token 的吞吐量，表示每秒生成的输出 token 数量，反应模型生成输出的能力。

---



### 2 vLLM性能测试指标

- Latency of vllm.
  - Metric: median end-to-end latency (ms). We use median as it is more stable than mean when outliers occur.
  - Input length: 32 tokens.
  - Output length: 128 tokens.
- Throughput of vllm.
  - Metric: throughput (request per second)
  - Input length: 200 prompts from ShareGPT.
  - Output length: the corresponding output length of these 200 prompts.
- Serving test of vllm
  - Metrics: median TTFT (time-to-first-token, unit: ms) & median ITL (inter-token latency, unit: ms). We use median as it is more stable than mean when outliers occur.
  - Input length: 200 prompts from ShareGPT.
  - Output length: the corresponding output length of these 200 prompts.
  - Average QPS: 1, 4, 16 and inf. QPS = inf means all requests come at once.


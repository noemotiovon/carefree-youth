### 1 服务启动

```bash
python3 -m vllm.entrypoints.openai.api_server 
				-tp 1 
				--model /home/lcg/.cache/modelscope/hub/Qwen/Qwen2-0___5B-Instruct 
				--port 8000 
				--disable-log-stats 
				--disable-log-requests
				# 可选--enable-prefix-caching \
#####同下
python3 -m vllm.entrypoints.openai.api_server -tp 1 --model /home/lcg/.cache/modelscope/hub/Qwen/Qwen2-0___5B-Instruct --port 8000 --disable-log-stats --disable-log-requests
```



### 2 访问服务

**使用 curl：**

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  	"model":"/home/lcg/.cache/modelscope/hub/Qwen/Qwen2-0___5B-Instruct",
    "prompt": "给我讲个英雄联盟的背景故事：",
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 1
  }'
```

**使用 requests 库（Python）：**

```python
import requests

url = "http://localhost:8000/v1/completions"
data = {
  	"model":"/home/lcg/.cache/modelscope/hub/Qwen/Qwen2-0___5B-Instruct",
    "prompt": "给我讲个英雄联盟的背景故事",
    "max_tokens": 1000,
    "temperature": 0.7
}

response = requests.post(url, json=data)
print(response.json())
```

可选参数：

"top_p": 1



### 3 代码片段

```python
if get_pp_group().is_last_rank:
		# Sampling metadata is only required for the final pp group
```

pp group 通常指的是 **Pipeline Parallel Group**

**1. 分组的作用**：

每个 pp group 包含执行流水线中一个阶段的设备。

不同的分组对应于模型的不同部分（如前几层、中间层、后几层）。

**2. 通信的范围**：

同一个 pp group 内的设备可能需要通信，负责将计算结果传递给下一个阶段的设备。

不同的 pp group 通常只通过边界传递数据，减少了全局同步的开销。

**3. 示例**：

假设一个模型分为 4 部分，使用 4 个 GPU。Pipeline 分组可能如下：

Group 1: GPU 0（前几层）

Group 2: GPU 1（中间层 1）

Group 3: GPU 2（中间层 2）

Group 4: GPU 3（后几层）



### 4 模型加载

**1. DummyModelLoader**

**对应场景**: 用于模拟加载过程，通常在测试和开发阶段使用。

**特点**:

不实际加载任何模型。

提供伪装接口以模拟模型加载行为。

**典型用途**:

单元测试时验证模型加载器的流程是否正确。

调试时减少不必要的资源消耗。

**2. TensorizerLoader**

**对应场景**: 加载经过 Tensorizer 工具预处理的模型。

**特点**:

Tensorizer 是一个用于将模型序列化为紧凑格式的工具。

能快速加载和反序列化模型到内存中。

**典型用途**:

部署需要高效加载模型的场景。

需要支持序列化和反序列化的在线推理。

**3. ShardedStateLoader**

**对应场景**: 加载分片模型（Sharded State Dict）。

**特点**:

模型的权重分散存储在多个文件中，每个分片只包含一部分参数。

通常用于大模型的加载，避免一次性加载整个模型导致内存不足。

**典型用途**:

分布式训练或推理场景。

超大模型（如 GPT-3、LLAMA）分片存储的权重加载。

**4. BitsAndBytesModelLoader**

**对应场景**: 加载使用 bitsandbytes 库量化的模型。

**特点**:

bitsandbytes 是一种高效的量化库，可以将模型权重压缩为低精度格式（如 8-bit 或 4-bit）。

显著降低模型的显存占用，同时维持较高精度。

**典型用途**:

推理时需要运行大模型，但硬件资源受限（如显存不足）。

部署需要显存优化的量化模型。

**5. GGUFModelLoader**

**对应场景**: 加载 GGUF 格式的模型。

**特点**:

GGUF 是一种针对高效推理优化的模型存储格式。

提供高效的内存映射和硬件兼容性。

**典型用途**:

部署专为推理优化的模型（如量化和裁剪后的模型）。

快速启动并适配多种硬件。

**6. DefaultModelLoader**

**对应场景**: 加载默认格式的模型，通常是 PyTorch 的 state_dict。

**特点**:

直接使用 PyTorch 的标准加载接口加载模型权重。

支持未被特殊处理或优化的模型文件。

**典型用途**:

处理常见的模型格式。

使用标准 PyTorch 模型加载流程。



### 5 kv-cache初始化

total_npu_memory：设备总内存

init_npu_memory：设备初始化后设备的总内存

执行profile_run，按照最大的seqs和tokens加载

此时的

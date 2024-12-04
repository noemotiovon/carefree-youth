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


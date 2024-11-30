### 1 服务启动

```bash
python3 -m vllm.entrypoints.openai.api_server 
				-tp 1 
				--model /home/lcg/.cache/modelscope/hub/Qwen/Qwen2-0___5B-Instruct 
				--port 8000 
				--disable-log-stats 
				--disable-log-requests
#####同下
python3 -m vllm.entrypoints.openai.api_server -tp 1 --model /home/lcg/.cache/modelscope/hub/Qwen/Qwen2-0___5B-Instruct --port 8000 --disable-log-stats --disable-log-requests
```


## 一、前置变量定义和初始化

```python
num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS
```

从环境变量中获取每个 worker 使用的 GPU 数量，通常是 1。

```python
self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
self.workers: List[RayWorkerWrapper] = []
self.pp_tp_workers: List[List[RayWorkerWrapper]] = []
```

初始化 driver dummy worker（仅占资源，不运行任务），以及实际用于运行任务的 workers 和基于 PP/TP 分组的 `pp_tp_workers`。

------

## 二、解析是否需要配置 Nsight profiling

```python
if self.parallel_config.ray_workers_use_nsight:
    ray_remote_kwargs = self._configure_ray_workers_use_nsight(ray_remote_kwargs)
```

如果启用了 Nsight profiling，会修改 `ray_remote_kwargs` 来支持 profiling 工具。

------

## 三、解析 bundle indices（用于资源绑定）

```python
if envs.VLLM_RAY_BUNDLE_INDICES:
    ...
else:
    for bundle_id, bundle in enumerate(placement_group.bundle_specs):
        if bundle.get(current_platform.ray_device_key, 0):
            bundle_indices.append(bundle_id)
    bundle_indices = bundle_indices[:self.parallel_config.world_size]
```

如果设置了 `VLLM_RAY_BUNDLE_INDICES` 环境变量，使用用户指定的 bundle 索引，否则从 `placement_group.bundle_specs` 中选出有 GPU 的前 N 个 bundle，N = world size。

你的设置是：

- world_size = TP×PP = 4
- `bundle_indices = [0, 1, 2, 3]`

------

## 四、启动 worker 并记录元信息

```python
for rank, bundle_id in enumerate(bundle_indices):
    ...
    worker = ray.remote(...)(RayWorkerWrapper).remote(...)
    worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=rank))
```

通过 Ray 启动 16 个 `RayWorkerWrapper` actor，每个 worker 绑定对应的 bundle。

- `rpc_rank=0~3`
- 这些 worker actor 会立即启动，但尚未初始化内部模型和资源。

------

## 五、收集每个 worker 的 IP 地址

```python
worker_ips = ray.get([
    each.worker.get_node_ip.remote() for each in worker_metadata
])
for each, ip in zip(worker_metadata, worker_ips):
    each.ip = ip
```

并行收集每个 worker 所在的物理节点 IP。

------

## 六、如果不是 Ray SPMD 模式，则需要从中挑选出 driver dummy worker

```python
if not self.use_ray_spmd_worker:
    for i, each in enumerate(worker_metadata):
        ...
        if self.driver_dummy_worker is None and worker_ip == driver_ip:
            self.driver_dummy_worker = worker
            self.driver_worker = RayWorkerWrapper(...)
            worker_metadata.pop(i)
            break
```

- 如果 `use_ray_spmd_worker` 为 False，会从与 driver 节点同一 IP 的 worker 中挑出一个 dummy worker。
- 用作资源占用，不实际执行任务。

之后总 worker 数就变成 15 个。

------

## 七、排序 worker 以便进行重新编号

```python
sorted_worker_metadata = sorted(worker_metadata, key=sort_by_driver_then_worker_ip)
...
for i, item in enumerate(sorted_worker_metadata):
    item.adjusted_rank = i + start_rank
```

排序规则：

1. 优先放 driver 节点上的 worker；
2. 节点上的 worker 越少，越靠前；
3. IP 小的排前。

然后：

- 为每个 worker 重新分配 `adjusted_rank`
- 若 SPMD 模式，rank 从 0 开始；否则从 1 开始（因为 rank=0 是 driver）

------

## 八、重新设定 `self.workers`，并通知 worker 自己的 rank

```python
self.workers = [item.worker for item in sorted_worker_metadata]
self._run_workers("adjust_rank", rerank_mapping)
```

将排序后的 worker 加入 `self.workers`，并通过 `adjust_rank` 告诉每个 worker 其新的 rank。

------

## 九、获取每个 worker 的 GPU 信息，并构建分布图

```python
for worker in [self.driver_dummy_worker] + self.workers:
    ...
    worker_node_and_gpu_ids.append(ray.get(worker.get_node_and_gpu_ids.remote()))

node_workers = defaultdict(list)
node_gpus = defaultdict(list)

for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
    node_workers[node_id].append(i)
    node_gpus[node_id].extend([int(x) for x in gpu_ids])
```

- 每个 worker 报告其所在 node 和 GPU ID。
- 构建 node -> worker rank 映射，和 node -> GPU 列表映射。

例：

```python
node_workers = {
  "node1": [0,1,...,7],
  "node2": [8,9,...,15]
}
```

------

## 十、准备传递给 worker 的环境变量

```python
all_args_to_update_environment_variables = [{
    current_platform.device_control_env_var:
    ",".join(map(str, node_gpus[node_id])),
} for (node_id, _) in worker_node_and_gpu_ids]
```

每个 worker 被传入一个参数字典，告诉它可见的 GPU。

```python
for args in all_args_to_update_environment_variables:
    for name in env_vars_to_copy:
        if name in os.environ:
            args[name] = os.environ[name]
```

从 driver 环境复制相关环境变量给每个 worker。

然后执行：

```python
self._run_workers("update_environment_variables", ...)
```

让每个 worker 设置其环境变量。

------

## 十一、构造初始化通信地址

```python
if len(node_gpus) == 1:
    driver_ip = "127.0.0.1"
distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())
```

构建用于 torch distributed 初始化用的 TCP 地址。

------

## 十二、初始化每个 worker 的 distributed 设置

```python
for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
    local_rank = node_workers[node_id].index(rank)
    kwargs = dict(...)
    all_kwargs.append(kwargs)

self._run_workers("init_worker", all_kwargs)
```

调用每个 worker 的 `init_worker`，设置：

- rank
- local_rank（在该节点内的 index）
- distributed 初始化地址
- 是否为 driver

------

## 十三、加载模型并初始化设备

```python
self._run_workers("init_device")
self._run_workers("load_model", ...)
```

分别初始化设备、加载模型（可配置并发数量）。

------

## 十四、根据 PP/TP 构建 `pp_tp_workers`

```python
if self.use_ray_spmd_worker:
    for pp_rank in range(self.parallel_config.pipeline_parallel_size):
        self.pp_tp_workers.append([])
        for tp_rank in range(self.parallel_config.tensor_parallel_size):
            rank = pp_rank * TP + tp_rank
            self.pp_tp_workers[pp_rank].append(self.workers[rank])
```

构建：

```python
self.pp_tp_workers = [
  [worker0, worker1],  # PP=0, TP=0,1
  [worker2, worker3],  # PP=1, TP=0,1
  ...
]
```

每一组 `pp_tp_workers[i]` 是 pipeline stage `i` 的一组 TP 并行 worker。

------

## 十五、划分 driver worker 和非 driver worker

```python
self.tp_driver_workers: List[RayWorkerWrapper] = []
self.non_driver_workers: List[RayWorkerWrapper] = []

for index, worker in enumerate(self.workers):
    rank = index + 1
    if rank % TP == 0:
        self.tp_driver_workers.append(worker)
    else:
        self.non_driver_workers.append(worker)
```

将 rank 为 TP 组首的 worker 归入 `tp_driver_workers`，其余归入 `non_driver_workers`，用于后续通信优化（如 broadcast）。






# （一）架构

## Ray 核心目标：

* **高性能**：强化学习中每次 observe 都需要与 Environment 进行交互，因此作者的目标是实现毫秒级计算延迟，每秒处理超过百万个任务。即使到了2024年，能达到这种性能的计算编排系统仍然很少。作者通过分布式调度器和分布式、容错的存储系统来管理控制状态，以满足AI应用的高性能需求。
* **统一的编程模型**：提供统一接口来同时支持任务并行和基于actor的计算，让开发者更容易构建和管理复杂的AI应用。这对提升开发和运维效率至关重要。通过一套框架灵活支持调度层和抽象层，可以将机器学习的模型训练、推理服务和数据处理都整合到同一个系统中，大幅降低开发复杂性和成本。
* **灵活性和异构支持**：支持时间和资源使用上的异构性，以及对不同类型硬件资源（如CPU、GPU、TPU）的支持。
* **动态执行**：支持动态执行模型，以适应AI应用中因环境互动而变化的计算需求。这是 Argo/Airflow 等任务编排框架所不具备的，或实现起来极其复杂的特性。
* **提高效率和降低成本**：通过优化资源利用和支持使用廉价资源（如AWS的spot实例），降低运行AI应用的成本。

## 实现方案

**1.统一的编程模型**：

- Ray提供统一接口来表达任务并行和基于actor的计算，使其能同时处理无状态并行任务和有状态持久化服务。
- 通过`@ray.remote`装饰器，用户可以定义远程函数（tasks）和远程对象（actors），并在Ray集群中实现并行分布式执行。

**2.动态任务图执行引擎**：

- Ray的核心是动态任务图执行引擎，能在运行时根据数据和控制流变化动态构建并执行任务图。
- 该引擎支持任务即时调度和执行，允许任务按需动态生成和建立依赖关系。

**3.分布式调度器**：

- Ray实现的分布式调度器能以毫秒级延迟调度数百万任务。
- 调度器利用全局控制存储（[GCS](https://zhida.zhihu.com/search?content_id=251837785&content_type=Article&match_order=1&q=GCS&zhida_source=entity)）记录任务信息和状态，实现透明调度和负载均衡。

**4.分布式和容错的存储系统**：

- Ray采用分布式对象存储管理任务的输入输出，确保数据一致性和容错性。
- 通过内存存储和LRU策略，Ray将任务执行延迟降至最低。

**5.全局控制存储（GCS）**：

- Ray引入GCS集中管理控制平面状态，包括任务、对象和actor的状态信息。
- GCS的设计支持Ray水平扩展，使系统组件能够独立扩展并保持容错能力。

**6.故障容忍**：

- Ray通过基于血统（lineage）的故障恢复机制，确保失败任务能透明地重新执行。
- 对于基于actor的计算，Ray采用基于检查点的恢复机制，缩短故障恢复时间。

**7.资源管理**：

- Ray支持为任务和actors指定包括CPU、GPU在内的资源需求，实现高效的资源管理和利用。
- 这种资源感知的调度机制让Ray能更好地支持异构计算资源。

**8. 性能优化**：

- Ray通过多线程网络传输和优化的序列化/反序列化机制减少任务执行延迟。
- Ray利用共享内存和零拷贝数据传输提升数据访问效率。

**9.易用性和集成：**

- Ray提供简洁API，能轻松与现有Python环境和AI/ML库集成。
- Ray的设计支持与模拟器和深度学习框架无缝集成，简化AI应用的开发和部署流程。



# （二）编译

>背景：Ray是一个使用Bazel构建的，基于gRPC上层打造的开源分布式计算框架，旨在简化分布式应用的开发和运行。它支持无缝地将 Python 代码扩展到多核、多节点环境，适合构建高性能的分布式系统。Ray 提供灵活的任务调度和状态管理，支持多种编程模型，包括任务并行和 actor 模式，并通过自动化的资源管理和容错机制简化复杂分布式工作的部署。它还拥有丰富的生态系统，包含机器学习库（如 Ray Tune、Ray Serve 和 RLlib），适用于模型训练、超参数调优、在线服务等场景，是云原生应用和大规模计算的理想选择。

1. 下载代码仓库

   ```bash
   git clone https://github.com/ray-project/ray.git
   ```

2. 创建虚拟环境

   ```bash
   # 创建虚拟环境 ray
   conda create -n ray python=3.9 
   # 激活虚拟环境
   conda activate myenv
   ```

3. 安装编译依赖

   ```bash
   # 安装基础依赖
   sudo apt-get update
   sudo apt-get install -y build-essential curl clang pkg-config psmisc unzip
   
   # 安装 Bazelisk.
   ci/env/install-bazel.sh
   
   # 安装dashboard编译需要的 node version manager and node 14
   $(curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh)
   # 在执行上面的命令时报错 bash: #!/usr/bin/env: No such file or directory，使用下面的脚本下载并安装
   curl -O https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh
   bash install.sh
   
   nvm install 14
   nvm use 14
   ```

4. 构建 dashboard

   ```bash
   cd ray/python/ray/dashboard/client
   npm ci
   npm run build
   ```

5. 构建 ray

   ```bash
   cd ../../..
   pip install -r requirements.txt
   
   ##如果构建机器的内存小于32G，需要限制内存使用，避免oom
   export BAZEL_ARGS="--local_ram_resources=8"
   ##debug编译，保留符号表供调试
   export RAY_DEBUG_BUILD=debug
   pip install -e . --verbose
   ```

   - 可以通过以下环境变量来调整构建过程（在运行 `pip install -e .` 或 `python setup.py install` 时）：
     - **RAY_INSTALL_JAVA**：如果设置为 1，将执行额外的构建步骤来构建代码库中的 Java 部分。
     - **RAY_INSTALL_CPP**：如果设置为 1，将安装 [ray-cpp](https://zhida.zhihu.com/search?content_id=251883928&content_type=Article&match_order=1&q=ray-cpp&zhida_source=entity)。
     - **RAY_DISABLE_EXTRA_CPP**：如果设置为 1，常规（非 cpp）构建将不提供某些 cpp 接口。
     - **SKIP_BAZEL_BUILD**：如果设置为 1，则不会执行 Bazel 构建步骤。
     - **SKIP_THIRDPARTY_INSTALL**：如果设置，将跳过安装第三方 Python 包。
     - **RAY_DEBUG_BUILD**：可以设置为 `debug`、`asan` 或 `tsan`。其他值将被忽略。
     - **BAZEL_ARGS**：如果设置，传递一组空格分隔的参数给 Bazel。这对于限制构建过程中资源的使用非常有用。例如，查看 [Bazel 用户手册](https://link.zhihu.com/?target=https%3A//bazel.build/docs/user-manual) 了解更多有效参数。
     - **IS_AUTOMATED_BUILD**：用于 CI，以调整 CI 机器的构建配置。
     - **SRC_DIR**：可以设置为源代码目录的根路径，默认值为 `None`（即当前工作目录）。
     - **BAZEL_SH**：在 Windows 上用于查找 `bash.exe`。
     - **BAZEL_PATH**：在 Windows 上用于查找 `bazel.exe`。
     - **MINGW_DIR**：在 Windows 上用于查找 `bazel.exe`，如果在 `BAZEL_PATH` 中没有找到的话。
   - 建议开启小梯子，不止加速效果很明显，中间有很多包下载会失败

【问题汇总】

**1 当你对系统进行 Bazel 不可能知道的操作时，最好运行`bazel clean --expunge`。这只是一种更礼貌的做法`rm -rf ~/.cache/bazel`:)**

**2 ubuntu 系统版本：24.04，gcc 版本：13.3.0，clang 版本：18.1.3**

**3 在Ascend环境上，有可能构建环境会错误的选择到gcc和lld，会导致一些奇怪的编译错误（例如，你的环境变量中存在lld，被识别，但是构建过程中并不会使用非系统路径下的lld等）。这时可以通过指定LD来解决：在 ~/.bazelrc 中加入：build --linkopt=-fuse-ld=gold**

**4 通过下面的脚本对PATH中的值去重**

```bash
export PATH=$(echo $PATH | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':')
```

**5 **





# （三）GCS

## 核心组件

**1.节点管理 (GcsNodeManager)**

- 负责管理集群中的节点
- 处理节点注册、心跳和故障检测

**2.Actor 管理 (GcsActorManager)**

- 管理 Actor 的创建、销毁和重建
- 处理 Actor 调度和故障恢复

**3.资源管理 (GcsResourceManager)**

- 管理集群资源
- 追踪节点资源使用情况

**4.任务管理 (GcsTaskManager)**

管理任务的调度和执行

**5.存储实现**

```c++
enum class StorageType {
  IN_MEMORY,      // 内存存储
  REDIS_PERSIST,  // Redis 持久化存储
  UNKNOWN
};
```

## 核心功能

**1. 服务启动流程**

```c++
void GcsServer::Start() {
  // 1. 初始化 KV 管理器
  InitKVManager();

  // 2. 异步加载 GCS 表数据
  auto gcs_init_data = std::make_shared<GcsInitData>();
  gcs_init_data->AsyncLoad();

  // 3. 初始化各个管理器
  InitGcsNodeManager();
  InitGcsActorManager(); 
  InitGcsResourceManager();
  // ...

  // 4. 启动 RPC 服务
  rpc_server_.Run();
}
```

在 **单机模式** 下，`ray.init()` 会触发 GCS Server 启动。

在 **集群模式** 下，GCS Server 是 `ray start --head` 时启动的，而 `ray.init()` 只会连接到已存在的 GCS Server。

**2.事件监听机制**

```c++
void GcsServer::InstallEventListeners() {
  // 节点事件
  gcs_node_manager_->AddNodeAddedListener();
  gcs_node_manager_->AddNodeRemovedListener();

  // Worker 事件  
  gcs_worker_manager_->AddWorkerDeadListener();

  // Job 事件
  gcs_job_manager_->AddJobFinishedListener();
}
```

**3. 调度机制**

```c++
// 资源变化时触发调度
gcs_resource_manager_->AddResourcesChangedListener([this] {
  // 调度待处理的 placement groups
  gcs_placement_group_manager_->SchedulePendingPlacementGroups();
  // 调度待处理的任务
  cluster_task_manager_->ScheduleAndDispatchTasks();
});
```

## 核心代码

### gcs_client

**1. accessor**

- accessor访问器主要用于**访问 GCS 中存储**的不同类型的数据
- 每个访问器类都提供了一系列异步和同步方法来：增删改查+订阅
- 大量使用异步操作,通过回调函数处理结果
- 支持超时机制
- 提供本地缓存功能

**2. global_state_accessor**

`GlobalStateAccessor` 是用来为语言前端(如 Python 的 state.py)提供同步接口来访问 GCS 中的数据。

`GlobalStateAccessor` 的 C++ 实现是通过 **Python C 扩展**（也叫 Python C API）导入到 `ray._raylet` 模块中的，并不是直接在 `ray/_raylet.py` 这个 Python 文件里定义的。**`ray._raylet` 不是 Python 代码，而是一个 C++ 编写的 Python C 扩展模块**（共享库 `.so/.pyd`）。

Ray 使用 **pybind11**（一个 C++ 绑定 Python 的库）来暴露 `GlobalStateAccessor` 给 Python。

* 同步接口
* 序列化处理：数据以序列化字符串形式返回,使用时需要用 protobuf 反序列化
* 线程安全，多重所保护
* 连接管理，Job 相关，Node 相关，Actor 相关，Placement Group 相关

### gcs_server

| 文件名                        | 主要功能             | 核心组件/类              | 关键特性                                                     |
| ----------------------------- | -------------------- | ------------------------ | ------------------------------------------------------------ |
| gcs_server.h/cc               | GCS 服务器的主要实现 | GcsServer                | - 管理所有 GCS 服务 <br />- 处理 RPC 请求 <br />- 协调各个组件 |
| gcs_resource_manager.h        | 资源管理器           | GcsResourceManager       | - 集群资源管理 <br />- 资源分配 - 资源追踪                   |
| gcs_resource_scheduler.h      | 资源调度器           | GcsResourceScheduler     | - 资源调度策略 <br />- 负载均衡 - 调度优化                   |
| gcs_actor_manager.h           | Actor 管理器         | GcsActorManager          | - Actor 生命周期管理 <br />- Actor 调度 - 故障恢复           |
| gcs_placement_group_manager.h | 放置组管理器         | GcsPlacementGroupManager | - 放置组创建/删除 <br />- 资源捆绑管理 <br />- 调度策略      |
| gcs_node_manager.h            | 节点管理器           | GcsNodeManager           | - 节点注册/注销 <br />- 心跳监控 <br />- 节点状态管理        |
| gcs_worker_manager.h          | Worker 管理器        | GcsWorkerManager         | - Worker 生命周期 <br />- Worker 分配 - 状态追踪             |
| gcs_job_manager.h             | Job 管理器           | GcsJobManager            | - Job 提交/完成 <br />- Job 状态管理 <br />- 资源分配        |
| gcs_table_storage.h           | 表存储接口           | GcsTableStorage          | - 元数据存储 <br />- 状态持久化 <br />- 数据访问接口         |
| gcs_redis_failure_detector.h  | Redis 故障检测器     | GcsRedisFailureDetector  | - Redis 健康检查 <br />- 故障检测 <br />- 自动恢复           |
| gcs_init_data.h               | 初始化数据管理       | GcsInitData              | - 系统初始化数据 <br />- 启动配置 <br />- 状态恢复           |
| gcs_function_manager.h        | 函数管理器           | GcsFunctionManager       | - 函数注册 <br />- 版本管理 <br />- 函数元数据               |
| gcs_kv_manager.h              | KV 存储管理器        | GcsKVManager             | - 键值存储 <br />- 元数据管理 <br />- 数据同步               |

### pubsub

**1. GcsPublisher类/GcsSubscriber/PythonGcsPublisher/PythonGcsSubscriber**

* 状态同步，支持各种消息类型（Actor/Job/NodeInfo/Error）
* 事件通知
* 资源更新

### store_client

**1. in_memory_store_client**

* 实现了所有必要的存储操作
* 内存管理，使用智能指针管理表对象，自动清理不再使用的资源
* 并发控制：细粒度锁控制，表级别的互斥访问
* 支持异步操作完成通知
* 高性能访问/查找

**2. observable_store_client**

* 跟踪系统状态：Actor 状态监控，资源状态追踪，任务状态更新
* 提供了一种机制来监控和响应系统状态变化
* 支持组件间的松耦合通信
* 实现了高效的状态同步机制
* 便于实现复杂的依赖关系和状态管理

**3. redis_store_client**

RedisStoreClient 提供了一个可靠的分布式存储实现，适合作为 Ray 系统的持久化存储后端。



# （四）Object Manager

| 文件名                                | 主要功能             | 核心组件                      | 关键特性                                                     |
| ------------------------------------- | -------------------- | ----------------------------- | ------------------------------------------------------------ |
| object_manager.h/cc                   | 对象管理器的核心实现 | ObjectManager                 | - 对象的推送/拉取 <br />- 对象生命周期管理 <br />- RPC 服务处理 <br />- 内存管理 |
| object_buffer_pool.h/cc               | 对象缓冲池管理       | ObjectBufferPool              | - 内存缓冲区管理 <br />- 对象数据传输 <br />- 内存复用       |
| chunk_object_reader.h/cc              | 分块对象读取器       | ChunkObjectReader             | - 大对象分块读取 <br />- 流式传输 <br />- 内存优化           |
| pull_manager.h/cc                     | 对象拉取管理         | PullManager                   | - 对象拉取请求管理 <br />- 重试机制 <br />- 优先级调度       |
| push_manager.h/cc                     | 对象推送管理         | PushManager                   | - 对象推送请求管理 <br />- 流量控制 <br />- 失败重试         |
| object_directory.h/cc                 | 对象目录服务         | ObjectDirectory               | - 对象位置跟踪 <br />- 对象元数据管理<br />- 位置更新通知    |
| ownership_based_object_directory.h/cc | 基于所有权的对象目录 | OwnershipBasedObjectDirectory | - 对象所有权管理 <br />- 生命周期控制 <br />- 垃圾回收       |
| plasma/store_runner.h/cc              | Plasma 存储运行器    | PlasmaStoreRunner             | - 共享内存管理 <br />- 对象持久化<br />- 存储服务            |
| spilled_object_reader.h/cc            | 内存溢出管理         | SpilledObjectReader           | - 提供高效可靠的溢出对象读取机制 <br />- 管理磁盘IO操作<br />- 支持大对象的高效处理 |
| common.h                              | 通用定义和工具       | -                             | - 常量定义 <br />- 工具函数<br />- 类型定义                  |


























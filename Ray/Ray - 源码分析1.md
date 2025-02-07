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
**1 编译redis时，报错 /bin/sh: 1: pkg-config: not found**

解决方法：

```bash
sudo apt-get install pkg-config
```



# （三）GCS












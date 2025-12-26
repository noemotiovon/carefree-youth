# ACL GRAPH 接入可视化设计图

本文档包含 ACL GRAPH 接入方案的详细可视化图表。

## 1. 系统交互时序图

```
┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│Application  │  │CANN Backend  │  │Graph Cache  │  │Graph Object │  │ACL Runtime   │
└──────┬──────┘  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
       │                 │                  │                │                │
       │ compute(graph)  │                  │                │                │
       ├────────────────>│                  │                │                │
       │                 │                  │                │                │
       │                 │ find_match()     │                │                │
       │                 ├─────────────────>│                │                │
       │                 │                  │                │                │
       │                 │                  │ matches?       │                │
       │                 │                  ├───────────────>│                │
       │                 │                  │                │                │
       │                 │                  │<───────────────┤                │
       │                 │                  │                │                │
       │                 │<─────────────────┤                │                │
       │                 │ found/not found  │                │                │
       │                 │                  │                │                │
       │                 │                  │                │                │
       │                 │ evaluate_and_capture_cann_graph()  │                │
       │                 │                  │                │                │
       │                 │                  │                │                │
       │                 │ ┌────────────────────────────────────────────────┐ │
       │                 │ │ if (use_cann_graph && capture_required)      │ │
       │                 │ │   aclmdlRICaptureBegin()                     │ │
       │                 │ │   └─────────────────────────────────────────>│
       │                 │ │                                              │ │
       │                 │ │ for each node:                               │ │
       │                 │ │   compute_forward(node)                      │ │
       │                 │ │   └─────────────────────────────────────────>│
       │                 │ │                                              │ │
       │                 │ │ if (use_cann_graph && capture_required)      │ │
       │                 │ │   aclmdlRICaptureEnd()                       │ │
       │                 │ │   <──────────────────────────────────────────┘
       │                 │ │                                              │ │
       │                 │ │ if (use_cann_graph)                          │ │
       │                 │ │   aclmdlRIExecuteAsync(graph)                │ │
       │                 │ │   └─────────────────────────────────────────>│
       │                 │ └──────────────────────────────────────────────┘ │
       │                 │                  │                │                │
       │<────────────────┤                  │                │                │
       │ success         │                  │                │                │
       │                 │                  │                │                │
```

## 2. 数据流向图

```
GGML Computation Graph (cgraph)
           │
           │ create_from_cgraph()
           ▼
┌──────────────────────────────┐
│   ggml_cann_graph            │
│   ────────────────────────   │
│   • aclmdlRI graph           │
│   • ggml_graph_properties[]  │
│     │                        │
│     ├─> node_properties[0]   │
│     │   • node_address       │
│     │   • node_op            │
│     │   • ne[], nb[]         │
│     │   • src_address[]      │
│     │   • src_ne[], src_nb[] │
│     │   • op_params[]        │
│     │                        │
│     ├─> node_properties[1]   │
│     │   ...                  │
│     │                        │
│     └─> node_properties[N]   │
└──────────────┬───────────────┘
               │
               │ push()
               ▼
┌──────────────────────────────┐
│  graph_lru_cache             │
│  ────────────────────────    │
│  • capacity: 12              │
│  • cache_list:               │
│    ┌─────────────────────┐   │
│    │ graph_0 (front)     │   │ ← Most Recently Used
│    ├─────────────────────┤   │
│    │ graph_1             │   │
│    ├─────────────────────┤   │
│    │ ...                 │   │
│    ├─────────────────────┤   │
│    │ graph_11 (back)     │   │ ← Least Recently Used
│    └─────────────────────┘   │
└──────────────────────────────┘
```

## 3. 状态转换图

```
                    ┌─────────────┐
                    │ Graph Mode  │
                    │ Disabled    │
                    └──────┬──────┘
                           │
                           │ acl_graph_mode = false
                           │
                           ▼
                ┌──────────────────────┐
                │  Eager Mode          │
                │  ────────────        │
                │  • 逐算子执行         │
                │  • 无图捕获           │
                │  • 无缓存管理         │
                └──────────────────────┘


                    ┌─────────────┐
                    │ Graph Mode  │
                    │  Enabled    │
                    └──────┬──────┘
                           │
                           │ acl_graph_mode = true
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Graph Capture Mode                  │
        │  ─────────────────                   │
        │  • 开始捕获: CaptureBegin            │
        │  • 执行算子: compute_forward         │
        │  • 结束捕获: CaptureEnd              │
        │  • 存储图: 加入 LRU Cache            │
        └───────────┬──────────────────────────┘
                    │
                    │ 图已捕获并缓存
                    │
                    ▼
        ┌──────────────────────────────────────┐
        │  Graph Execution Mode                │
        │  ─────────────────                   │
        │  • 查找匹配: find_and_move_to_front  │
        │  • 执行图: ExecuteAsync              │
        │  • 跳过逐算子执行                     │
        └───────────┬──────────────────────────┘
                    │
                    │ 图结构发生变化
                    │
                    └──────────────────────────┐
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  Graph Capture Mode  │
                                    │  (重新捕获)          │
                                    └──────────────────────┘
```

## 4. 决策树图

```
开始执行计算图
      │
      ▼
┌─────────────────┐
│ acl_graph_mode? │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   No        Yes
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────────────┐
│ Eager   │ │ prefill 模式检测     │
│ Mode    │ │ (FLASH_ATTN_EXT)     │
└─────────┘ └──────────┬───────────┘
                       │
                  ┌────┴────┐
                  │         │
              prefill    decode
                  │         │
                  ▼         ▼
            ┌─────────┐ ┌──────────────┐
            │ Eager   │ │ 使用 Graph   │
            │ Mode    │ │ 模式         │
            └─────────┘ └──────┬───────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 在 Cache 中查找     │
                    │ find_and_move_to_   │
                    │ front(cgraph)       │
                    └──────┬──────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                  找到          未找到
                    │             │
                    ▼             ▼
         ┌─────────────────┐ ┌─────────────────┐
         │ capture_req=    │ │ capture_req=    │
         │ false           │ │ true            │
         │                 │ │                 │
         │ 复用已有图        │ │ 创建新图并捕获    │
         └─────────────────┘ └─────────────────┘
```

## 5. 核心方法调用栈

```
ggml_backend_cann_graph_compute(backend, cgraph)
│
├─> 设置设备: ggml_cann_set_device(device)
│
├─> 清理工作区: g_nz_workspaces[device].clear()
│
├─> 判断是否使用 Graph 模式
│   │
│   ├─> 检查 acl_graph_mode 配置
│   │
│   ├─> 检查 prefill 模式 (可选)
│   │   └─> 遍历节点查找 FLASH_ATTN_EXT
│   │       └─> 检查序列长度 (ne[1])
│   │
│   └─> 决定 use_cann_graph 值
│
├─> Graph Cache 操作
│   │
│   ├─> 如果 use_cann_graph == true
│   │   │
│   │   └─> graph_lru_cache.find_and_move_to_front(cgraph)
│   │       │
│   │       ├─> 遍历 cache_list
│   │       │   │
│   │       │   └─> graph->matches_cgraph(cgraph)
│   │       │       │
│   │       │       ├─> 比较节点数量
│   │       │       │
│   │       │       └─> 遍历每个节点
│   │       │           │
│   │       │           └─> prop->has_matching_properties(node)
│   │       │               │
│   │       │               ├─> 比较地址
│   │       │               ├─> 比较算子类型
│   │       │               ├─> 比较维度/步长
│   │       │               └─> 比较算子参数
│   │       │
│   │       └─> 如果找到: 移动到前端, 返回 true
│   │           如果未找到: 返回 false
│   │
│   └─> 如果未找到匹配
│       │
│       └─> ggml_cann_graph::create_from_cgraph(cgraph)
│           │
│           ├─> 创建新 graph 对象
│           │
│           └─> 遍历 cgraph->nodes
│               │
│               └─> 提取并存储节点属性
│                   │
│                   ├─> node_address, node_op
│                   ├─> ne[], nb[]
│                   ├─> src_address[], src_ne[], src_nb[]
│                   └─> op_params[]
│
└─> evaluate_and_capture_cann_graph(ctx, cgraph, use_cann_graph, capture_required)
    │
    ├─> 如果 use_cann_graph && capture_required
    │   │
    │   └─> aclmdlRICaptureBegin(stream, GLOBAL_MODE)
    │
    ├─> 如果 !use_cann_graph || capture_required
    │   │
    │   └─> 遍历 cgraph->nodes
    │       │
    │       ├─> 跳过空节点/VIEW/RESHAPE 等
    │       │
    │       └─> ggml_cann_compute_forward(ctx, node)
    │           │
    │           └─> 根据节点类型调用对应的 ACL 算子
    │               │
    │               ├─> aclnnAdd, aclnnMul, etc.
    │               ├─> aclnnMatmul, aclnnRmsNorm, etc.
    │               └─> ... (其他算子)
    │
    └─> 如果 use_cann_graph
        │
        ├─> 获取 cache_list.front() (匹配的 graph)
        │
        ├─> 如果 capture_required
        │   │
        │   └─> aclmdlRICaptureEnd(stream, &graph->graph)
        │
        └─> aclmdlRIExecuteAsync(graph->graph, stream)
```

## 6. 内存与资源管理图

```
┌─────────────────────────────────────────────────────────┐
│              ggml_backend_cann_context                   │
│  ───────────────────────────────────────────            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  graph_lru_cache                                 │   │
│  │  ─────────────                                   │   │
│  │                                                   │   │
│  │  cache_list (std::list<ggml_cann_graph*>)        │   │
│  │  ┌────────────────────────────────────────────┐  │   │
│  │  │ graph_0                                    │  │   │
│  │  │ ┌──────────────────────────────────────┐  │  │   │
│  │  │ │ aclmdlRI graph ─────┐                │  │  │   │
│  │  │ │                     │                │  │  │   │
│  │  │ │ properties[]        │                │  │  │   │
│  │  │ │  ├─> node_prop_0    │                │  │  │   │
│  │  │ │  ├─> node_prop_1    │                │  │  │   │
│  │  │ │  └─> node_prop_N    │                │  │  │   │
│  │  │ └─────────────────────┘                │  │  │   │
│  │  │                                        │  │  │   │
│  │  │ ┌───────────────────────────────────┐ │  │  │   │
│  │  │ │ 析构时自动释放:                   │ │  │  │   │
│  │  │ │ aclmdlRIDestroy(graph)            │ │  │  │   │
│  │  │ └───────────────────────────────────┘ │  │  │   │
│  │  └───────────────────────────────────────┘  │  │   │
│  │  ... (最多 12 个)                           │  │   │
│  │  └───────────────────────────────────────┘  │  │   │
│  │                                               │  │   │
│  │  当缓存满时:                                  │  │   │
│  │  └─> 删除 cache_list.back()                  │  │   │
│  │      └─> 调用析构函数                         │  │   │
│  │          └─> 释放 aclmdlRI 资源              │  │   │
│  └──────────────────────────────────────────────┘  │   │
│                                                      │   │
│  ┌──────────────────────────────────────────────┐   │   │
│  │  streams[GGML_CANN_MAX_STREAMS]              │   │   │
│  │  ─────────────────────────────────           │   │   │
│  │  • stream[0] ────────────> aclrtStream      │   │   │
│  │  • stream[1] ────────────> aclrtStream      │   │   │
│  │  • ...                                       │   │   │
│  └──────────────────────────────────────────────┘   │   │
│                                                      │   │
│  ┌──────────────────────────────────────────────┐   │   │
│  │  mem_pool                                    │   │   │
│  │  ─────────────────────────────────           │   │   │
│  │  • 内存池管理                                 │   │   │
│  └──────────────────────────────────────────────┘   │   │
└──────────────────────────────────────────────────────┘

资源生命周期:
  ┌─────────────────────────────────────────────┐
  │ ggml_backend_cann_context 创建              │
  │   └─> graph_lru_cache 初始化                │
  │       └─> capacity = 12 (可配置)            │
  └─────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │ 执行计算图                                  │
  │   └─> 创建 ggml_cann_graph                  │
  │       └─> aclmdlRI graph = nullptr          │
  └─────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │ 捕获阶段                                    │
  │   └─> aclmdlRICaptureBegin                  │
  │       └─> 执行算子 (compute_forward)        │
  │           └─> aclmdlRICaptureEnd            │
  │               └─> graph->graph 被赋值       │
  └─────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │ 缓存管理                                    │
  │   └─> cache_list.push_front(graph)          │
  │       └─> 如果满，删除 back()               │
  └─────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │ 后续执行                                    │
  │   └─> 复用已捕获的 graph                    │
  │       └─> aclmdlRIExecuteAsync              │
  └─────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │ 上下文销毁                                  │
  │   └─> graph_lru_cache.~destructor()         │
  │       └─> 遍历所有 graph                    │
  │           └─> delete graph                  │
  │               └─> graph->~destructor()      │
  │                   └─> aclmdlRIDestroy()     │
  └─────────────────────────────────────────────┘
```

## 7. 性能优化策略图

```
性能优化维度
    │
    ├─> 减少图捕获开销
    │   │
    │   ├─> LRU 缓存机制
    │   │   └─> 捕获一次，多次复用
    │   │
    │   └─> 精确匹配策略
    │       └─> 避免不必要的重新捕获
    │
    ├─> 减少调度开销
    │   │
    │   └─> Graph 模式
    │       └─> 一次性执行整个图
    │           └─> 减少 host-device 交互
    │
    ├─> 内存优化
    │   │
    │   ├─> LRU 容量限制
    │   │   └─> 默认 12 个图
    │   │
    │   └─> 自动资源释放
    │       └─> 析构函数保证清理
    │
    └─> 执行模式优化
        │
        ├─> 异步执行
        │   └─> aclmdlRIExecuteAsync
        │
        └─> 场景自适应
            ├─> Prefill: 可选择 eager mode
            └─> Decode: 使用 graph mode
```

## 8. 错误处理流程图

```
ACL API 调用
    │
    ▼
┌──────────────────┐
│ ACL_CHECK(stmt)  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│ 执行 stmt               │
│ err_code = (stmt)       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ err_code == 0?          │
└────┬────────────┬───────┘
     │            │
    Yes          No
     │            │
     │            ▼
     │    ┌──────────────────────┐
     │    │ aclGetRecentErrMsg() │
     │    │ 获取错误信息          │
     │    └──────────┬───────────┘
     │               │
     │               ▼
     │    ┌──────────────────────┐
     │    │ ggml_cann_error()    │
     │    │ • 打印错误信息        │
     │    │ • 打印文件/行号       │
     │    │ • 终止程序            │
     │    └──────────────────────┘
     │
     ▼
继续执行
```

## 9. 配置与初始化流程图

```
程序启动
    │
    ▼
┌──────────────────────────┐
│ 读取环境变量             │
│ ─────────────────────    │
│ • GGML_CANN_ACL_GRAPH    │
│ • GGML_CANN_PREFILL_...  │
│ • GGML_CANN_GRAPH_...    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 创建 CANN Context        │
│ ─────────────────────    │
│ ggml_backend_cann_       │
│ context(device)          │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 解析配置                 │
│ ─────────────────────    │
│ acl_graph_mode =         │
│   parse_bool(env)        │
│                          │
│ graph_lru_cache.capacity │
│   = parse_integer(env)   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 初始化完成               │
│ ─────────────────────    │
│ • device 已设置          │
│ • streams 未创建 (延迟)   │
│ • graph_lru_cache 为空    │
│ • acl_graph_mode 已设置   │
└──────────────────────────┘
```

## 10. 典型执行场景示例

### 场景 1: 首次执行 (需要捕获)

```
时间线: 
─────────────────────────────────────────────────────────>
        
步骤 1: find_and_move_to_front(cgraph)
        └─> cache 为空，返回 false
        └─> graph_capture_required = true
        
步骤 2: create_from_cgraph(cgraph)
        └─> 创建新的 ggml_cann_graph
        └─> 提取所有节点属性
        └─> graph.graph = nullptr
        
步骤 3: push(new_graph)
        └─> 添加到 cache_list 前端
        
步骤 4: evaluate_and_capture_cann_graph(...)
        │
        ├─> aclmdlRICaptureBegin() [开始捕获]
        │
        ├─> for each node:
        │   └─> compute_forward(node) [执行并捕获]
        │
        ├─> aclmdlRICaptureEnd() [结束捕获]
        │   └─> graph.graph 被赋值
        │
        └─> aclmdlRIExecuteAsync(graph.graph) [执行图]
        
结果: 图已捕获并缓存，可以复用
```

### 场景 2: 重复执行 (直接复用)

```
时间线: 
─────────────────────────────────────────────────────────>
        
步骤 1: find_and_move_to_front(cgraph)
        └─> 遍历 cache_list
        └─> 找到匹配的 graph (graph_0)
        └─> 移动到前端
        └─> 返回 true
        └─> graph_capture_required = false
        
步骤 2: evaluate_and_capture_cann_graph(...)
        │
        ├─> 跳过 CaptureBegin (不需要捕获)
        │
        ├─> 跳过逐算子执行 (使用 graph)
        │
        └─> aclmdlRIExecuteAsync(graph.graph) [直接执行]
        
结果: 跳过捕获和逐算子执行，直接执行已缓存的图
```

### 场景 3: 图结构变化 (重新捕获)

```
时间线: 
─────────────────────────────────────────────────────────>
        
步骤 1: find_and_move_to_front(cgraph)
        └─> 遍历所有缓存的 graph
        └─> 都不匹配 (节点数量或属性不同)
        └─> 返回 false
        └─> graph_capture_required = true
        
步骤 2: create_from_cgraph(cgraph)
        └─> 创建新的 ggml_cann_graph
        └─> 提取新图的节点属性
        
步骤 3: push(new_graph)
        └─> 添加到 cache_list 前端
        └─> 如果 cache 满，删除最旧的 graph
        
步骤 4: evaluate_and_capture_cann_graph(...)
        │
        └─> [同场景 1 的捕获和执行流程]
        
结果: 新图已捕获并缓存，旧图被淘汰（如果 cache 满）
```


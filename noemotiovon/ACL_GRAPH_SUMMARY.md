# ACL GRAPH 接入设计方案 - 快速总结

## 核心方法: evaluate_and_capture_cann_graph

这是 ACL GRAPH 接入的核心实现方法，负责整个图的捕获和执行流程。

### 方法签名

```cpp
static void evaluate_and_capture_cann_graph(
    ggml_backend_cann_context * cann_ctx,           // CANN 后端上下文
    ggml_cgraph *               cgraph,             // GGML 计算图
    bool                        use_cann_graph,     // 是否使用 CANN graph
    bool                        cann_graph_capture_required // 是否需要捕获图
)
```

### 核心功能

1. **图捕获启动** (条件: `use_cann_graph && cann_graph_capture_required`)
   - 调用 `aclmdlRICaptureBegin()` 开始图捕获

2. **计算图执行** (条件: `!use_cann_graph || cann_graph_capture_required`)
   - 遍历所有节点
   - 跳过 VIEW、RESHAPE 等无需执行的节点
   - 调用 `ggml_cann_compute_forward()` 执行每个算子

3. **图捕获结束** (条件: `use_cann_graph && cann_graph_capture_required`)
   - 调用 `aclmdlRICaptureEnd()` 获取捕获的图对象
   - 将图对象存储到 `ggml_cann_graph` 中

4. **图执行** (条件: `use_cann_graph`)
   - 从 LRU Cache 前端获取匹配的图
   - 调用 `aclmdlRIExecuteAsync()` 异步执行图

### 关键设计点

#### 1. 图捕获与执行的分离

- **首次执行**: 需要捕获 (`capture_required = true`)
  - 执行流程: CaptureBegin → 执行算子 → CaptureEnd → ExecuteAsync
  
- **重复执行**: 直接复用 (`capture_required = false`)
  - 执行流程: ExecuteAsync (跳过捕获和逐算子执行)

#### 2. 图匹配机制

通过 `ggml_graph_node_properties` 进行精确匹配:
- 节点数量
- 算子类型
- 张量维度、步长
- 张量地址 (VIEW 节点除外)
- 算子参数 (特定算子)

#### 3. LRU 缓存管理

- 默认容量: 12 个图
- 新图添加到前端
- 匹配的图移动到前端
- 缓存满时删除最旧的图

### 调用关系

```
ggml_backend_cann_graph_compute()
    │
    ├─> 判断是否使用 Graph 模式
    │   ├─> 检查 acl_graph_mode 配置
    │   ├─> 检查 prefill 模式
    │   └─> 决定 use_cann_graph
    │
    ├─> Graph Cache 查找/创建
    │   ├─> find_and_move_to_front() → 查找匹配
    │   └─> create_from_cgraph() → 创建新图
    │
    └─> evaluate_and_capture_cann_graph()
            │
            ├─> CaptureBegin (如果需要)
            ├─> 逐算子执行 (如果需要)
            ├─> CaptureEnd (如果需要)
            └─> ExecuteAsync (如果使用 graph)
```

### 性能优化

1. **减少捕获开销**: 通过 LRU 缓存复用已捕获的图
2. **精确匹配**: 避免不必要的重新捕获
3. **异步执行**: 使用 `aclmdlRIExecuteAsync` 不阻塞主线程
4. **场景自适应**: Prefill 可禁用 Graph，Decode 使用 Graph

### 配置选项

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `GGML_CANN_ACL_GRAPH` | 启用 ACL Graph | `on` |
| `GGML_CANN_PREFILL_USE_GRAPH` | Prefill 使用 Graph | `false` |
| `GGML_CANN_GRAPH_CACHE_CAPACITY` | 缓存容量 | `12` |

### 相关文件

- **实现文件**: `ggml/src/ggml-cann/ggml-cann.cpp`
  - `evaluate_and_capture_cann_graph()` (行 2093-2134)
  - `ggml_backend_cann_graph_compute()` (行 2148-2194)

- **头文件**: `ggml/src/ggml-cann/common.h`
  - `ggml_backend_cann_context` (行 549-640)
  - `ggml_cann_graph` (行 288-371)
  - `ggml_cann_graph_lru_cache` (行 380-438)
  - `ggml_graph_node_properties` (行 218-286)

### 详细文档

- **完整设计方案**: `ACL_GRAPH_DESIGN.md`
- **可视化图表**: `ACL_GRAPH_VISUALIZATION.md`


**小结(重要)**：

可以理解为一个seq，经过生成prompt，prompt包含token，vllm根据block_size，将prompt分成不同的token group，token group包含多个token，每个token会被放在某个槽位上，槽位信息在slot_mapping中存储。每个槽位可以存放一个token所需要的所有的kv cache的存储。一个block占用的空间 = 2 * block_size * num_head * head_size * num_layers * dtype_size。

### 7 BlockSpaceMange

vLLM对物理块的操作都由BlockSpaceMange块管理器实现。

块管理器类是在调度类**Scheduler**中初始化的。

分配物理块的动作是在调度的过程中执行的。

物理块和设备强相关。

#### 7.1 PhysicalTokenBlock

调度系统中的物理块**并不执行存储kv值的操作**，它的用途是**记录物理block的状态**。
**调度系统和块管理器中使用的是真实设备的物理块的编号、状态等信息**。

```python
class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        self.device = device
        # 该物理块在对应设备上的全局block索引号
        self.block_number = block_number
        # 每个block槽位数量(默认16)
        self.block_size = block_size
        # 在prefix caching场景下使用，其他场景值为-1
        self.block_hash = block_hash
        # 该物理块的hash值是由多少个前置token计算而来的，非prefix caching场景值为0
        self.num_hashed_tokens = num_hashed_tokens
        # 该物理块被引用次数
        self.ref_count = 0
        # 物理块最后一个被访问时间，非prefix caching场景值为-1
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME
        # 该物理块是否被计算过，只在prefix caching场景下启用
        self.computed = False

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed})')


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]
```

---

#### 7.2 BlockSpaceMangeV1

代码：

```python
class BlockSpaceManagerV1(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
            self,
            block_size: int,
            num_gpu_blocks: int,
            num_cpu_blocks: int,
            watermark: float = 0.01,
            sliding_window: Optional[int] = None,
            enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
		...
        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching
        # 水位线，是一个数量阈值，设置它的目的是避免gpu上物理块全部使用完。
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # 根据是否做了prefix caching限制，来选择不同的allocator
        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            self.gpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                    Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                    Device.CPU, block_size, num_cpu_blocks)
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                    Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator = UncachedBlockAllocator(
                    Device.CPU, block_size, num_cpu_blocks)
         
        # Mapping: seq_id -> BlockTable.
        # 记录每个seq对应的BlockTable(这是一个包含物理块索引号的list)
        self.block_tables: Dict[int, BlockTable] = {}
        
        # Mapping: req_id -> BlockTable. Note that each SequenceGroup has a unique equest ID
        # 功能同上，但cross_block_tables记录的是encoder-decode类型的模型，暂时混略
        self.cross_block_tables: Dict[str, BlockTable] = {}
```

* BlockAllocator：物理块分配者，负责实际为seq做物理块的分配、释放、拷贝等操作。我们推理时使用gpu_allocator，和 cpu_allocator用于gpu资源不足时临时存储kv-cache，对应的swapped队列。

  其中，BlockAllocator又分成两种类型：

  * CachedBlockAllocator：按照prefix caching的思想（prompt共享）来分配和管理物理块。带有这些相同prefix信息（如"提示词 你是一个助手"）的prompt完全可以共享用于存放prefix的物理块，这样既节省显存，也不用再对prefix做推理。
  * UncachedBlockAllocator：正常分配和管理物理块，没有额外实现prefix caching的功能。

* **block_tables**：负责维护每个seq下的物理块列表，本质上它是一个字典，因为调度器是全局的，所以它下面的的BlockManager自然也是全局的。因为seq_id也是全局唯一，所以这个字典维护着调度系统中所有待推理的seq（即使它们在不同的seq_group中）的物理块。

```python
class UncachedBlockAllocator(BlockAllocatorBase):
    def __init__(
            self,
            device: Device,
            block_size: int,
            num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: BlockTable = []
        # 假设系统GPU可用显存能容纳256个block，那就在这里直接
        # 初始化256个block，用时从free_blocks中取就好。
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_blocks.append(block)

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        """分配block: 从自由态block列表中取出一个block，并将引用计数设为1"""
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        """释放block，引用计数置为0"""
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        """获得当前gpu上可用block数量"""
        return len(self.free_blocks)

    def get_num_total_blocks(self) -> int:
        """获得当前gpu所有block总数"""
        return self.num_blocks
	...
```

---

#### 7.3 can_allocate

是否可为seq_group分配足够物理块用于prefill（**_schedule_prefills中有使用**）

**_num_required_blocks**是当前seq_group需要的block数量

```python
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        # 只对encoder-decode模型有效，忽略
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # 计算当前seq序列需要的物理block数量
        # 这是seq的一个属性，对于waiting状态的seq，n_blocks=len(prompt)/16, 向上取整
        self_num_required_blocks = self._get_seq_num_required_blocks(
                seq_group.get_seqs(status=SequenceStatus.WAITING)[0])
        # 又是encoder-decode相关，忽略
        cross_num_required_blocks = self._get_seq_num_required_blocks(seq_group.get_encoder_seq())
        num_required_blocks = self_num_required_blocks + cross_num_required_blocks
		
		    # 滑窗，忽略
        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks, self.block_sliding_window)
        # 当前gpu空闲的blocks数量
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        # 如果设备中所有的物理块数量 - 该seq实际需要的物理块数量 < 水位线block数量，则不分配
        # 说明当前seq太长了，标记为NEVER，以后也不处理这个seq_group了
        if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks:
            return AllocStatus.NEVER
        # 如果设备中可用的物理块数量 - 该seq实际需要的block数量 >= 水位线block数量，则分配
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        # 否则，现在不能分配(暂时没足够的blocks)，但可以延迟分配
        else:
            return AllocStatus.LATER
```

---

#### 7.4 allocate

```python
    def allocate(self, seq_group: SequenceGroup) -> None:
        is_encoder_decoder = seq_group.is_encoder_decoder()
        # 只对encoder-decode模型有效，忽略
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        # 对于WAITING装的seq_group，seq只有1条，就是prompt
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        # block_table:list,存储的是当前seq用到的物理块的索引号
        block_table: BlockTable = self._allocate_sequence(seq,
                                                          seq_group.num_seqs(),
                                                          is_encoder_decoder)

        # Assign the self-attention block tables for each sequence.
        # 记录每一个seq序列使用的block_table，block_tables是一个全局变量，记录这所有
        # seq_group的seq，根据add_request()中代码可知，不同seq_group的seq.id也不会重复，没有相互覆盖的风险
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.block_tables[seq.seq_id] = block_table.copy()

        # Allocate encoder sequence
        # 忽略
        if is_encoder_decoder:
            # A SequenceGroup has only a single encoder sequence (at most),
            # thus allocate with a ref count of 1
            block_table = self._allocate_sequence(seq_group.get_encoder_seq(),
                                                  1, is_encoder_decoder)
            # Assign the cross-attention block table for the SequenceGroup.
            self.cross_block_tables[seq_group.request_id] = block_table

```

---

#### 7.5 _allocate_sequence

allocate中分配物理的方法是:_allocate_sequence，从以下代码可以看出，vllm删除了logical block，取而代之的关系在这里呈现
。从空闲的物理blocks中取出 num_prompt_blocks 个block，映射给当前seq_group中的seq。

```python
    def _allocate_sequence(self, \
                           seq: Sequence, \
                           ref_count: int, \
                           is_encoder_decoder: bool = True) -> BlockTable:
        # Allocate new physical token blocks that will store the prompt tokens.
        # 当前seq需要的物理块数量
        num_prompt_blocks = seq.n_blocks

        block_table: BlockTable = []
        for logical_idx in range(num_prompt_blocks):
            # 滑窗，忽略
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
                # Set the reference counts of the token blocks.
                block.ref_count = ref_count
            elif not is_encoder_decoder and self.enable_caching:
                block = self.gpu_allocator.allocate(
                        seq.hash_of_block(logical_idx),
                        seq.num_hashed_tokens_of_block(logical_idx))
            # 默认情况下走下面的分支
            else:
                block = self.gpu_allocator.allocate()
                # Set the reference counts of the token blocks.
                # 由于seq_group下的所有seq共享一个prompt，所以有ref_count = num_seqs
                # 表示这些seqs的逻辑块都引用它了
                block.ref_count = ref_count
            block_table.append(block)

        return block_table

```

---

#### 7.6 can_append_slots

```python
    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerV1"

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

```

can_allocate和 can_append_slots 的区别：

* can_allocate：对处于waiting状态的seq_group，首先要给他分配block，做prefill，即prompt的token产生的kv-cache存放在block中。此时占用block数量根据prompt长度而定。假设prompt长度为20，block_size为16，则需要2个block。
* can_append_slots：对处于running状态的seq_group，处于解码状态，每个seq每次推理会产生1个tokens，有num_seqs个seq则会产生num_seqs个token，最好的情况是：每个seq对应的last block都没满，不需要新增block就能完成新kv-cache的存储，此时需要的blocks为0， 最坏的情况是：每个seq last block都满了，再进来的token只能开辟新的block，此时需要的blocks数量为num_seqs，所有当可用blocks数量多于或等于num_seqs，当前seq_group就能继续做推理。

### 8 模型初始化

在vllm初始化时，主要初始化4个模块：**tokenizer（分词器），model_executor（tf模型转换到vllm模型），self._initialize_kv_caches（kv block初始化），scheduler （调度器）**

#### 8.1 model_executor

模型的初始化在model_executor的构造方法中调用，通过load_model加载。

self.driver_worker是work（vllm/worker/worker.py）的一个实例对象，**每个gpu上的都维护着自己的Worker实例**，负责维护 KV-cache，并在 GPU 上执行模型。在分布式推理的情况下，每个work都会被分配模型的一部分（不同的head并行计算，然后汇总计算结果）。

_initialize_model函数的功能为**通过hf模型的config参数，获得模型名**，然后根据这个名称去加载vllm**改造后的该模型模型结构**

每个模型有自己加载权重的方式，下面是llama模型加载权重的代码vllm/model_executor/models/llama.py：

```python
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # vllm与hf两种模型实现方式之间的名称映射
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # vllm, hf,share_id
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        # 获得当前vllm改造后llama模型的参数和对应的权重(此时的权重应是随机生成的)
        params_dict = dict(self.named_parameters())
        # 遍历hf模型每层参数的名称和权重
        for name, loaded_weight in weights:
			...
            # vllm, hf,share_id
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # 将hf模型的层名，替换为vllm中的层名
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                # 获得vllm改造后llama权重参数
                param = params_dict[name]
                weight_loader = param.weight_loader
                # 将hf模型参数更新到对应的vllm模型参数中,完成权重参数的映射工作
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
				...

```

通过上述vllm中llama的load_weights方法(经过观察， 所有decode-only模型的load_weights几乎都一样)，**将vllm模型和hf模型不同参数名之间做映射，之后将hf类型的权重赋值给vllm模型中**（通过参数名联系），至此，完成模型转换工作。

---

#### 8.2 _initialize_kv_caches

作用是计算当前blocks总量，可用blocks数量。

* **Batch size (**B**)**：批次的数量。

* **Num heads (**H**)**：注意力头的数量；通常用于多头注意力，以便在不同子空间中进行并行的注意力计算。

* **Sequence length (**T**)**：输入序列的token数量。

* **Head dimension (**d_k**)**：单个头的key或value的维度大小，即每个注意力头的投影维度。

在transformer中，QKV的矩阵形状如下：

* **query(q)的shape**：(B, H, T, d_k)

* **key (k) 的shape**：(B, H, T, d_k)

* **value (v) 的shape**：(B, H, T, d_k)

一个block占用的空间 = 2 * block_size * num_head * head_size * num_layers * dtype_size

每个token占用的KV Cache = 2 * num_head * head_size * num_layers * dtype_size

每个token要保存计算过的所有层的kv值，这样才算一个完整的kv-cache。

_initialize_kv_caches方法的目的是**计算出GPU/CPU block数量，然后对这些block进行初始化**。

```python
    def _initialize_kv_caches(self) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        num_gpu_blocks, num_cpu_blocks = self.model_executor.determine_num_available_blocks()
		...
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self.model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)

```

计算block数量的方法为self.model_executor.**determine_num_available_blocks**()

```python
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        # 构建推理允许的最大seq和tokens 数量组成的推理数据，进行不使用kv-cache的模型推理
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        # 记录此时可用的GPU和总GPU数量，此时模型运行占用的GPU显存还没释放
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # peak_memory就是当前模型占用的显存
        peak_memory = self.init_gpu_memory - free_gpu_memory
		...
		# 获得一个block占用的GPU显存
        cache_block_size = self.get_cache_block_size_bytes()
        # 计算总的可用GPU block数量
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -peak_memory) // cache_block_size)
        # 计算CPU数量,对于CPU，不需要额外计算，因为是固定大小的内存。
        num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks
```

vllm如何计算KV Cache的显存？

* 根据设置的最大seq和最大的tokens，构造一组“最大”数据。
* 用这一组数据，去跑一下不用KV-Cache，做一遍推理。
* 记录此时的可用GPU内存和总GPU内存，free_gpu_memory，total_gpu_memory。
* 模型初始化后的内存：self.init_gpu_memory
* 模型占用的内存：peak_memory = self.init_gpu_memory - free_gpu_memory
* 可分配给KV Cache的内存：total_gpu_memory * self.cache_config.gpu_memory_utilization（利用率） - peak_memory
* 计算cpu可以分配的内存块：self.cache_config.swap_space_bytes // cache_block_size

计算出KV Cache后，开始完成CacheEngine的init。

```python
    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        # shape=[num_blocks, block_size，num_kv_heads，head_size]
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        # 遍历每一层，一个token的完整kv-cache包含所有层的子kv
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fb499fc239264688868eca2fdbdc8be1.png)

kv-cache.shape每个维度代表含义如下：

list 28：当前模型有28层，每层都要保持当前层计算的kv

内部元素的shape含义：

2：分别存储k和v的计算结果
2760：当前GPU有2760个block
16：每个block有16个槽位，即可以放16个k或v的值
8：当前模型head数量
128：每个head的head_size
这个kv-cache就是推理过程中用于存储kv值的容器，这里一次性初始好，全部填充为0，所以在实际推理过程中会发现，vllm会直接把显存占用爆涨到一个很大值，就是因为初始化了很多预填充kv-cache的block。

---

### 9 模型推理step()

#### 9.1 使用模型

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e6a2f38b7a9e4eb182c6a950e72ccbd1.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/10f13e1fb563455db0c67dcc45a4b082.png)

在vLLM中，

input_ids.shape=[num_tokens, ] 假如输入的3条prompt长度分别为48，44，43，那么num_tokens=135

在 vllm 中，将输入的多个 prompt 序列的 token 合并在一起，形成一个一维 input_ids（形状为 [num_tokens, ]），从而避免了传统的 padding 操作。这样做的原因是为了：

1. **节省内存和计算**：padding 会引入多余的计算开销和显存占用，而不合并输入序列时，每个序列都需要 pad 到最大长度（如 48）。

2. **简化去 padding 的操作**：在生成过程中，去掉填充内容（去 padding）可能会带来额外的复杂度。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/83aac0afde8248efa7577d67e031f1d7.png)

在进行推理前，我们还需要把准备prefill的prompt的每个token（就是上面的input_ids, 这时还没做embedding操作）映射到block中，如seq_id=0的prompt长度为48，由于block_size=16, 所以他刚好能填充3个block（编号为2759,2758,2757）。映射关系会写入到slot_mapping列表中，那么这个操作如何来做呢？
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/92758f8019f541ff825a551b4f2312bd.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/37c5d2eb62224d578089ccdf1066e425.png)

slot_mapping存放的是槽位号，以第一个seq为例，他分到了三个block，分别是2759，2758，2757。那么他对应的槽位号分别是的起始位置分别是2759 * 16（block_size），2758 * 16，2757 * 16；分别是44144，44128，44112。

#### 9.2 模型推理

model_input:

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/18f614aa9c1e49a19ee13638bd9a2ad7.png)

```python
    def execute_model(
            self,
            model_input: ModelInputForGPUWithSamplingMetadata,
            kv_caches: List[torch.Tensor],
            intermediate_tensors: Optional[IntermediateTensors] = None,
            num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
		...
        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][graph_batch_size]
        else:
            model_executable = self.model
		...
        hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **MultiModalInputs.as_kwargs(multi_modal_kwargs, device=self.device),
                **seqlen_agnostic_kwargs)
		...

```

我们使用第四篇文章用过的llama3.1来剖析剩余代码，**model_executable**最终执行llama模型的forward代码。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6b08a33727b24038942205798f664a20.png)

llama结构类型的大模型的推理，可分为两个阶段：prompt和generate, 在使用kv-cache的情况下，二者的区别仅是输入数据维度的差异，即generate阶段seq序列长度始终为1， 不过在vllm中却有不一样的处理，prefill之后，会把模型构建为cuda计算图，这样计算会更加高效。

##### **第一次推理 prefill**

入参：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fbb80b0d587c40e78655143d55a2d124.png)

```python
def forward(
    self,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
    intermediate_tensors: Optional[IntermediateTensors],
    inputs_embeds: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, IntermediateTensors]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
        	# 输入的通常都是未embedding的token，在这里进行词嵌入
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    for i in range(self.start_layer, self.end_layer):
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions,	# shape=[num_tokens,]
            hidden_states,	# shape=[num_tokens,embed_size]
            kv_caches[i - self.start_layer],	# 当前layer对应的kv-cache
            attn_metadata,	# 保存着slot_mapping, 通过这个map向kv-cache中填值
            residual,
        )
	...
    return hidden_states
```

**小结(重要)**：

可以理解为一个seq，经过生成prompt，prompt包含token，vllm根据block_size，将prompt分成不同的token group，token group包含多个token，每个token会被放在某个槽位上，槽位信息在slot_mapping中存储。每个槽位可以存放一个token所需要的所有的kv cache的存储。一个block占用的空间 = 2 * block_size * num_head * head_size * num_layers * dtype_size。

```python
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output

```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/878ab59278a746bcba540da692e594d5.png)

k,v的shape为[135,1024], q的shape为[135,4096], 说明使用了GQA技术，即4个q共享一个kv

GQA（Grouped Query Attention，分组查询注意力）技术是一种通过分组查询来优化注意力机制的技术，主要用于大型语言模型的多头注意力机制中。它的主要思想是让多个 Query 共享一个 Key-Value 对，这样可以有效减少计算和存储需求。

计算模块：vllm_module/attention/backends/flash_attn.py class FlashAttentionImpl(AttentionImpl)

**该模块主要完成两个功能：缓存kv值和计算attention。**

```python
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
		...
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        # query.shape=[135, 32, 128]
        query = query.view(-1, self.num_heads, self.head_size)
        # key.shape=[135, 8, 128]
        key = key.view(-1, self.num_kv_heads, self.head_size)
        # value.shape=[135, 8, 128]
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
        	# 取出该层缓存key的block，key_cache.shape=[1756, 16, 8, 128]
        	# 关于这个shape的维度含义，再第四篇文章中已经讲过了
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
			# 调用cuda核函数缓存kv值
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )
		...
        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache is None or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
				# 计算attention值
                out = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
				...

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

```

```python
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, key_cache,
                                                   value_cache, slot_mapping,
                                                   kv_cache_dtype, k_scale,
                                                   v_scale)

```

```c++
void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, const double k_scale,
    const double v_scale) {
	...
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}

```

```
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)         \
  vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>       \
      <<<grid, block, 0, stream>>>(                                   \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), block_stride, key_stride, \
          value_stride, num_heads, head_size, block_size, k_scale, v_scale);

```

```c++
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads,
                                         // head_size]
    cache_t* __restrict__ value_cache,   // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride, const int key_stride, const int value_stride,
    const int num_heads, const int head_size, const int block_size,
    const float k_scale, const float v_scale) {
  // 每个cuda block处理一个token
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  
  // 如果槽索引小于 0，表示 token 被填充（padding），则直接返回
  if (slot_idx < 0) {
    return;
  }
   // 计算 block 索引和 block 内的偏移量
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  
  // 计算每个注意力头和每个头的总数据量
  const int n = num_heads * head_size;
  
  // 每个线程处理数据中的一个元素
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    // 计算当前线程处理的 key 和 value 数据在输入数组中的索引
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;
    // 计算当前元素对应的注意力头索引和头内的偏移量
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    // 计算在缓存中目标位置的索引
    const int64_t tgt_key_value_idx = block_idx * block_stride +
                                      block_offset * num_heads * head_size +
                                      head_idx * head_size + head_offset;
    
    // 从输入数组中加载当前的 key 和 value 数据
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    
    // 缓存kv值
    // 如果使用自动类型，不进行额外的缩放和转换，直接存储
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_value_idx] = tgt_key;
      value_cache[tgt_key_value_idx] = tgt_value;
    } else {	// 否则，使用指定的缩放因子对数据进行转换后存储
      key_cache[tgt_key_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, k_scale);
      value_cache[tgt_key_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, v_scale);
    }
  }
}

```

##### **非第一次推理 decode**

经过预填充阶段后，vllm会把模型本身及推理过程处理成cuda计算图，正式的解码阶段，会直接使用计算图获得推理结果。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/62ffc6ee88c541ef8ead142716a6dd76.png)

在decode推理前，我们先来看下输入参数与prefill有什么不同：
在初始阶段我们设定每个seq生成4条output，关于拼接原理，在第一篇文章由详细讲过了。
从model_input数据结构看，此时的模型输入只有一个token（这是prefill后生成的第一个token）。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/70f94e1be782492194ea3f476efafc6e.png)

我们输入的prompt数量为3，设定每个prompt生成4条output，为什么这里是16个token？ 这是因为decode使用的是cuda计算图，图需要固定大小的张量。

计算图执行的推理流程如下：vllm/worker/model_runner.py class CUDAGraphRunner

```python
def forward(
        self,
        input_ids: torch.Tensor,                       # 输入的 token IDs 张量
        positions: torch.Tensor,                       # 输入的位置信息张量
        kv_caches: List[torch.Tensor],                 # KV cache 列表（这里被删除，不再使用）
        attn_metadata: AttentionMetadata,              # 注意力元数据，包含 slot_mapping 和其他解码元数据
        intermediate_tensors: Optional[IntermediateTensors],  # 中间张量，可能包含中间结果的数据
        **kwargs,                                      # 其他关键字参数，用于额外的自定义操作
) -> torch.Tensor:
    # KV caches 是固定的张量，因此在后续操作中不需要复制它们
    del kv_caches  # 删除 kv_caches，因为它们不再需要

    # 将输入张量复制到模型的输入缓冲区
    self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)  # 复制输入 token IDs
    self.input_buffers["positions"].copy_(positions, non_blocking=True)  # 复制位置信息
    self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping, non_blocking=True)  # 复制 slot_mapping
    
    # 根据后端的不同，处理额外的输入数据
    if self.backend_name != "flashinfer":
        # 如果后端不是 "flashinfer"，复制解码元数据中的序列长度和块表
        self.input_buffers["seq_lens_tensor"].copy_(
                attn_metadata.decode_metadata.seq_lens_tensor,
                non_blocking=True)
        self.input_buffers["block_tables"].copy_(attn_metadata.decode_metadata.block_tables, non_blocking=True)
    
    # 如果 input_buffers 包含 "seqlen_agnostic_capture_inputs"，在 CUDA 图之前复制输入
    if "seqlen_agnostic_capture_inputs" in self.input_buffers:
        self.model.copy_inputs_before_cuda_graphs(self.input_buffers, **kwargs)

    # 如果提供了 intermediate_tensors，复制这些中间张量到输入缓冲区
    if intermediate_tensors is not None:
        for key in intermediate_tensors.tensors:
            self.input_buffers[key].copy_(intermediate_tensors[key], non_blocking=True)
    
    # 执行计算图，计算存储在self的各个属性中
    # 这个计算图是核心代码，可惜这里看不到。
    self.graph.replay()
    
    # 如果 input_buffers 包含 "seqlen_agnostic_capture_inputs"，在 CUDA 图之后复制输出
    if "seqlen_agnostic_capture_inputs" in self.input_buffers:
        self.model.copy_outputs_after_cuda_graphs(self.input_buffers, **kwargs)
    
    # 返回输出张量
    if get_pp_group().is_last_rank:
        return self.output_buffers["hidden_states"]  # 如果是最后一个进程，返回隐藏状态张量

    return self.output_buffers  # 否则返回输出缓冲区

```


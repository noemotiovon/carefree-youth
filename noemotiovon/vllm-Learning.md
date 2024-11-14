## vLLM源码解析

> 参考文章：[CSDN-弈秋001](https://blog.csdn.net/weixin_42479327?type=blog)
>
> vllm版本：0.5.4

### 1 大模型推理流程

1. Prefill：预填充阶段，把整段prompt喂给大模型做推理，获得kv-cache并保存。
2. Decode：自回归生成阶段，根据prompt开始自回归生成Token序列。由于Decode阶段是逐一生成token，因此不能像Prefill阶段那样能做大段prompt的并行计算，所以在LLM推理过程中，Decode阶段的耗时一般是更大的，单步生成token的耗时约占总推理时长的90%。

---

### 2 PagedAttention

参考文章：[图解大模型计算加速系列之：vLLM核心技术PagedAttention原理](https://zhuanlan.zhihu.com/p/691038809)

PagedAttention参考实现：虚拟内存分页管理

概念梳理：

* 对于一个序列，生成的token串是连续的，连续的token串组成一个block块（vLLM默认值为16个token组成一个block）
* block那的token是连续的token串，block间的token串不一定连续

三张表：

1. Logical KV blocks：存储每个token在哪个块中，记录Physical Block Number，多个token对应某一Physical Block Number
2. Block Table：存储每个块的信息，如Physical Block Number、对应物理内存的位置信息以及当前填充到了哪个位置
3. Physical KV blocks：每个token实际存储在物理内存的哪个位置

由于一个prompt可能对应多个输出，当块内在Decode过程中，产生多个结果时，就对当前一个block进行复制，之后再分别进行推理（Paraller Sampling）。这种做法其实也是Automatic Prefix Caching，一种类似于前缀树的架构，通过这种方式减少内存的占用。

多输出有两种情况：

1. Parallel Sampling: 如果指定了n个输出，就把prompt复制n份，拼成一个batch喂给模型做推理。这时会产生prompt 的kv-cache重复存储，对这个重复的优化是另外问题，这里不展开了。
2. Beam Search：集束搜索，每个decode阶段，产生top k个token（k也被称为束宽），对应着当前时刻的top k个序列。它们的前置token也会有大量的kv-cache重复。

---

### 3  vLLM数据结构

1个请求batchsize包含多个prompts，每一个prompt被认定为vLLM中的一个请求，每一个prompt请求对应多个seq组成的output输出。每个seq都有单独的推理状态。

- WAITING：正在waiting队列中。waiting队列中的序列都没有做过prefill。
- RUNNING：正在running队列中，即已经开始做推理。
- SWAPPED：正在swapped队列中，表示此时gpu资源不足，相关的seq_group被抢占。
- FINISHED_STOPPED：正常执行完毕，例如碰到符号，该seq的推理正常结束了
  FINISHED_LENGTH_CAPPED：因为seq的长度达到最大长度限制，而结束推理
  FINISHED_ABORTED：因不正常状态，而被终止的推理。例如客户端断开连接，则服务器会终止相关seq的推理
  FINISHED_IGNORED：因prompt过长而被终止执行的推理。本质上也是受到长度限制

一个seqenceGroup对应唯一的一个prompt、一个requestId，包含多个Sequence。

---

### 4 调度原则

Scheduler维护者三个双端队列**waiting，running，swapped。**

* waiting：入口，初始只有prompt，后续推理会成为新的数据。
* running：存储着上一次被送去做推理的seq_groups，下一次推理前，需要先检查系统是否有足够资源让他们留在队列中继续做下一次推理。如果资源不足，就把seq一条一条pop出去，放到waiting或者swapped队列。
* swapped：不满足条件的，放到swapped。

调度原则：FCFS，资源不足需要抢占时，后来的请求被先抢占（preemption），对应钢材running中的pop操作。

关于抢占 preemption 的因素：

1. gpu blocks 数量是否充足
2. 当前调度能处理的seqs和tokens是否超过数量阈值（指的是每次允许推理的最大seqs数量和tokens数量）

关于抢占 preemption 的处理方式：

1. **如果parallel sampling=1**，直接释放所有physical blocks，将任务重新放回wait队列（放到队列头部，下一次最先取它），**重新**从prefill阶段开始做推理。
2. **如果parallel sampling>1**，先把处理的好的blocks交换到CPU上，等gpu显存充足，再把这些blocks从CPU加载回来。

![image-20241112104538789](/Users/lichenguang/Library/Application Support/typora-user-images/image-20241112104538789.png)

* **每条prompt处理成Sequence对象，然后Sequence包装成seq_group**，这条seq_group会存入waiting队列。此时只有一条seq，就是prompt，**连预填充prefill都没做**。status为waiting
* 调度器选中这条seq_group做推理，图中我们展示两种情况，4输出和1输出，因此会产生4条seq和1条seq, 其中4 seq共享prompt，status为running
* 推理一段时间后，gpu blocks资源不足或tokens或seqs数量超出阈值，发生抢占现象。多输出的seq_group相关kv blocks会被swap out到CPU上；而单输出的seq_group则会把相关的（prefill和decode）kv blocks释放，将seq_group重新放回waiting队列，就像什么都没发生过，幸运的是会被放到waiting队列最前面。
* 系统资源充足了，被swapped的seq_group会从CPU上**swap_in** gpu继续推理；单输出seq_group则**从prefill开始重新奋斗**。
* 多输出**必定会出现某条seq先推理结束**，此时还活跃的seq数减1， 变为3个，当某条seq推理完结，会被标记为finished, 以后不再调用资源处理它，只有等seq_group中所有seq都推理结束，该seq_group才算推理完成。

---

### 5 推理代码

入口：llm.generate()

关键函数：**_validate_and_add_requests（数据预处理），_run_engine（实际推理）**。

#### **5.1 _validate_and_add_requests：**

主要函数链路：

* _validate_and_add_requests

* llm._add_request

  * 赋值request_id

* llm_engine.add_request

  * 设置请求到达时间

* llm_engine.process_model_inputs

  * 拿到tokenizer，并将文字tokenizer编码成token_id

  * 将结果转换为字典形式返回

    ```json
    llm_inputs = {‘prompts’:xxx,‘prompts_token_ids’:xxx,‘multi_modal_data’:None}
    ```

  * 用户输入的prompt经过_validate_and_add_requests处理后，**会封装为seq_group**，然后将seq_group加入合适gpu维护的scheduler的waiting队列, 等待处理。

* llm_engine._add_processed_request

  * 添加seq_id，与request_id是两个完全不同的变量
  * 整合生成Sequence对象，包含当前prompt的各种信息:token_id,status(waiting,...), 占用blocks数量(逻辑,物理数量相同)
  * 根据Sequence对象，整合生成seq_group
  * 获取当前每个调度器（GPU）上未推理结束的seq_group数量：len(self.waiting) + len(self.running) + len(self.swapped)
  * 找出工作量最少的调度器（GPU）并将seq加入

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e6c15c6cffff4edaa56cc6d307eb8f25.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54251d83029e407a9167c605e477e4bd.png)

#### 5.2  _run_engine: 

整个推理engine中，最重要的是**self.llm_engine.step()**，封装了所有的调度，推理和后处理代码。

主要函数链路：

* 初始化tqdm（python进度条工具）

* ```
  				# 如果当前调度器中还有没完成推理的请求（调度器中waiting/running/swapped任一队列非空）
          while self.llm_engine.has_unfinished_requests():
              # 执行1次推理调度（step），决定哪些请求的数据可以参与到这次推理中，step输出本次推理结果
              step_outputs = self.llm_engine.step()
              # 一次step推理后，如果有请求已经完成了推理，将推理结果装进outputs中，
              for output in step_outputs:
                  if output.finished:
                      outputs.append(output)
                      if use_tqdm:
                          if isinstance(output, RequestOutput):
                              # Calculate tokens only for RequestOutput
                              total_in_toks += len(output.prompt_token_ids)
                              in_spd = total_in_toks / pbar.format_dict["elapsed"]
                              total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                              out_spd = total_out_toks / pbar.format_dict["elapsed"]
                              pbar.postfix = (
                                  f"est. speed input: {in_spd:.2f} toks/s, output: {out_spd:.2f} toks/s"
                              )
                          pbar.update(1)
  ```

* llm_engine.step()

---

### 6 Schedule

#### 6.1 _schedule_default

Schedule的入口在操作swapped队列，目的在于构造running队列。

![image-20241113105511107](/Users/lichenguang/Library/Application Support/typora-user-images/image-20241113105511107.png)

**调度方法**：

* schedule_prefill 处理Waiting队列，从Waiting队列中获取seq_group，类比领导安排的工作
* schedule_running处理Running队列，从Running队列中获取seq_group，类比手头正在处理的工作
* schedule_wrapped处理Wrapped队列，从Wrapped队列中获取seq_group，类比暂时搁置的工作

**调度顺序**：

1. 优先查看是否有阻塞的任务（check Swapped队列），如果没有阻塞的任务，则工作不饱和，调用schedule_prefill处理Waiting队列的任务。
2. 如果有阻塞的任务（check Swapped队列），则工作已经饱和，调用schedule_running处理Running队列的任务，并检查当前执行的任务是否过载，是否需要抢占。
3. 如果需要抢占，说明工作量饱和，将被抢占的seq_groups根据不同的抢占类型放入waiting队列和Swapped队列。
4. 如果不需要抢占，则说明当前处理的任务不饱和，将之前阻塞的任务选出部分来执行，调用schedule_warpped处理Wrapped队列的任务。
5. 将所有搜集到的任务，校验是否超过了Budget（check tokens and seqs），注意是seqs而不是seq_groups。
6. 将搜集到的任务加入Running队列，并执行推理。

**代码**：

```python
def _schedule_default(self) -> SchedulerOutputs:
     """Schedule queued requests.
     The current policy is designed to optimize the throughput. First,
     it batches as many prefill requests as possible. And it schedules
     decodes. If there's a pressure on GPU memory, decode requests can
     be swapped or preempted.

     当前策调度略旨在优化吞吐量。首先，它会批量处理尽可能多的预填充请求。然后它会安排解码。
     如果 GPU 内存有压力，则可以交换或抢占解码请求。因此会优先从swapped进行判断。
     """
     # Include running requests to the budget.
     # 每次step都要重新初始化一个budget来管理本次调度的的tokens和seqs数量, 根据数量是否超过阈值，决定将本次
     # seq_groups放入哪个队列。(一个seq_groups会包含多个seqs)
     budget = SchedulingBudget(
             token_budget=self.scheduler_config.max_num_batched_tokens,
             max_num_seqs=self.scheduler_config.max_num_seqs,
     )

     # Make sure we include num running seqs before scheduling prefill,
     # so that we don't schedule beyond max_num_seqs for prefill.
     # 先统计正在执行推理的seq_groups中seq的数量
     for seq_group in self.running:
         budget.add_num_seqs(seq_group.request_id, seq_group.get_max_num_running_seqs())
     # lora推理相关，可忽略
     curr_loras = set(
             seq_group.lora_int_id for seq_group in self.running
             if seq_group.lora_int_id > 0) if self.lora_enabled else None

     # 以下三个变量，类似于C++中的结构体。将多个变量合在一起，通过.属性访问
     # 各自保存处于不同活跃状态(wait,run,swap)的seq_groups具有的属性
     prefills = SchedulerPrefillOutputs.create_empty()
     running_scheduled = SchedulerRunningOutputs.create_empty()
     swapped_in = SchedulerSwappedInOutputs.create_empty()

     # If any requests are swapped, prioritized swapped requests.
     # 为什么要从swap开始判断？
     # 调度的任务是优化吞吐量，即保证处于running状态的seqs最多。running从wait和swap队列
     # 获得，首先积压的任务可能要比wait的优先级高，因为swap队列中的任务始终占据着系统资源，当
     # running可添加时，应该首先处理swap。
     if not self.swapped:  # 如果swapped队列为空
         # 既然不能从swap想running转移，那就只能从wait队列拿任务了。
         # wait队列中的都是原始任务，第一步要预填充
         # prefills是一个伪结构体：可以.出以下属性
         #     seq_groups: List[SequenceGroup]
         #     ignored_seq_groups: List[SequenceGroup]
         #     num_lookahead_slots: int
         prefills = self._schedule_prefills(budget, curr_loras, enable_chunking=False)

     # Don't schedule decodes if prefills are scheduled.
     # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
     # only contains decode requests, not chunked prefills.

     # self.waiting空,或 self.swapped非空,都会导致prefills.seq_groups数量为0
     # # 这个判断的意思是,prefills.seq_groups==0,说明本次调度没有安排预填充任务,那么就安排解码任务.
     # 执行推理任务的seq_group都在running队列，因此需要对这个队列进行调度。
     # 调度什么呢？
     # 是看running队列中的seq_group是否可以继续做推理任务。因为vllm动态管理，最大限度优化吞吐量，会导致blocks资源紧张
     # 上次推理生成的tokens的kv-cache需要GPU blocks去存储，导致资源消耗。那么这次准备推理时blocks数量不一定能够它完成
     # 推理，所以要对running队列中每个seq_group进行检查，看是否可以进行做推理。
     if len(prefills.seq_groups) == 0:
         running_scheduled = self._schedule_running(budget, curr_loras, enable_chunking=False)

         # If any sequence group is preempted, do not swap in any sequence
         # group. because it means there's no slot for new running requests.
         # 在对running队列调度后(从self.running队列取seq_group,准备进行推理),如果没有seq_group被
         # 抢占(退回wait队列),也没有seq_group被转移到CPU上, 说明blocks资源充足,可以把以前
         # self.swapped队列中积压的seq_group转移到gpu blocks做推理.

         # 注意这几个队列的判断逻辑. 如果self.swapped原本就非空,会进入上面的if判断分支进行self.running队列
         # 调度取值.然后根据这个过程中是否有seq_group被preempted和swapped_out获知blocks资源使用情况.
         # 如果没有被preempted和swapped_out.说明工作不饱和，就把self.swapped内容取出来，加入running队列进行推理
         # 如果有被preempted和swapped_out，说明资源紧张. self.swapped积压的任务暂时还处理不了
         # 如果下面if为False,意思是就不要再从self.swapped转移seq_group会gpu做推理了
         if len(running_scheduled.preempted) + len(running_scheduled.swapped_out) == 0:
             swapped_in = self._schedule_swapped(budget, curr_loras)

     # 最后一次判断本次推理的tokens和seqs数量是否超过阈值
     assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
     assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

     # Update waiting requests.
     # 这个类型被抢占的seq_group，打回原型，重新加入waiting队列。
     # 幸运的是添加到了队列头部，当再次从waiting队列取数据时，会优先处理它
     self.waiting.extendleft(running_scheduled.preempted)
     # Update new running requests.
     # 将以上通过层层筛选的seq_group加入到running队列(真·running)，这些seq_group才是下一步的推理对象
     self.running.extend([s.seq_group for s in prefills.seq_groups])
     self.running.extend([s.seq_group for s in running_scheduled.decode_seq_groups])
     self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])
     # Update swapped requests.
     # 没有足够资源做推理的seq_group会从running转移到swap队列(swap队列是路径之一，另一个是加入到wait队列)
     self.swapped.extend(running_scheduled.swapped_out)
     # 统计被抢占的seq_group数量
     preempted = len(running_scheduled.preempted) + len(running_scheduled.swapped_out)

     # There should be no prefill from running queue because this policy
     # doesn't allow chunked prefills.
     assert len(running_scheduled.prefill_seq_groups) == 0
     assert len(swapped_in.prefill_seq_groups) == 0

     return SchedulerOutputs(
             scheduled_seq_groups=(prefills.seq_groups +
                                   running_scheduled.decode_seq_groups +
                                   swapped_in.decode_seq_groups),
             num_prefill_groups=len(prefills.seq_groups),
             num_batched_tokens=budget.num_batched_tokens,
             blocks_to_swap_in=swapped_in.blocks_to_swap_in,
             blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
             blocks_to_copy=running_scheduled.blocks_to_copy +
                            swapped_in.blocks_to_copy,
             ignored_seq_groups=prefills.ignored_seq_groups +
                                swapped_in.infeasible_seq_groups,
             num_lookahead_slots=running_scheduled.num_lookahead_slots,
             running_queue_size=len(self.running),
             preempted=preempted,
     )
```

---

#### 6.2 _schedule_prefills

是否可非配物理块有三种状态：OK/NEVER/LATER，分别代表立即分配/永不分配/延迟分配。

ignored_seq_groups：存放因太长，导致所需的blocks和总blocks的差值超过了某个阈值。

![image-20241113142537919](/Users/lichenguang/Library/Application Support/typora-user-images/image-20241113142537919.png)

prompt limit：限制用户在输入 prompt（即指令或文本）时可以包含的最大 token 数量，防止过长的输入导致资源占用过高。（单个输入）

Token limit：限制模型在输入和输出之和上的 token 数量，确保不会超出模型的实际处理能力。（全部的输出和输出）

**调度顺序**：

1. 因为加入任务也需要占用时间，需要通过self._passed_delay(time.time())判断是否是合理的获取任务时机，平衡调度任务与推理任务的时机，同时在waiting队列不为空的时候去获取seq_group。
2. seq_group参数的有效性校验，此时seq_gorup中只有一条还未产生推力的seq。
3. 如果seq的长度（tokens长度） > 每次调度能处理的最大序列长度，标记FINISHED_IGNORED，装入ignored_seq_groups，从Waiting队列移除。
4. can_allocate = self.block_manager.can_allocate(seq_group)，根据可分配状态，如果为NEVER，标记FINISHED_IGNORED，装入ignored_seq_groups，从Waiting队列移除，继续处理下一个seq_group；如果为LATER，则跳出循环，直接返回；如果为ok，则开始分配物理块。

**代码**：

```python
    def _schedule_prefills(
            self,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        # ignored_seq_groups：记录因太长（所需的blocks和总blocks之间的差值超过阈值了），
        # 而无法继续做生成的seq_group，这些seq_group中的seq状态都会被标记为
        # FINISHED_IGNORED，表示直接不处理他们
        ignored_seq_groups: List[SequenceGroup] = []
        # 用于装载从wait队列转移出来的seq_group
        seq_groups: List[SequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        # self._passed_delay：通过比较当前请求到达时间来确定是否要从wait队列拿任务
        # 因为拿任务也要占用时间，需要平衡调度任务与推理任务的调用时机
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            # list,从当前seq_group取出准备推理的seq序列. 1个prompt可能有多个seq(多输出)，但wait队列中连预填充
            # 都没进行，因此这时的seq(仅是prompt)数量必定==1
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, "Waiting sequence group should have only one prompt sequence."

            # 当前待推理的seq_group需要处理,或者说准备返回的tokens数量,
            # 对于WAITING状态，只有1个seq，tokens数量为prompt长度
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                # 从waiting取出的seq_group，连预填充都没做，更不会有output token，
                # 若计算出的tokens数量不等与prompt数量，一定有问题，抛出异常吧！
                assert num_new_tokens == num_prompt_tokens

            # 如果这条seq的长度 > 每次调度能处理的最大序列长度，那么把这条seq的状态置为FINISHED_IGNORED，
            # 并将对应seq_group装入ignored_seq_groups中，然后将其从waiting列表中移除，永不再处理，完结撒花~
            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                        "Input prompt (%d tokens) is too long"
                        " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)  # 加入失败者联盟
                waiting_queue.popleft()  # 从 waiting 队列中踢出去
                continue  # 继续从waiting拿数据处理

            # If the sequence group cannot be allocated, stop.
            # 比较当前seq需要的物理块,gpu可用物理块之间的数量关系. 决定是否能给当前seq_group分配物理块
            # can_allocate返回值可能有三种： NEVER：不分配；OK：可以分配；LATER：延迟分配
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            # 当前seq需要的blocks数量,超过gpu能提供的最大数量.加入失败者联盟,永不再处理，
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                        f"Input prompt ({num_new_tokens} tokens) is too long"
                        " and exceeds the capacity of block_manager")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # lora相关，忽略
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            # 当前seq_group中状态为 未执行完 的序列的数量，即seq还没推理完成的数量. 刚从wait中取出时，
            # seq数量是1,但推理生成阶段,这个seq_group中会有n个seq在并行.n是外部传入的output数量. 因此这里num_new_seqs==n
            num_new_seqs = seq_group.get_max_num_running_seqs()
            # budget.can_schedule同时判断tokens和seqs数量是否超过阈值，任一个超过单次调度能执行的总数的阈值
            # 说明这step可推理的seqs数量已经马上趋于饱和，不能再加入seq到running队列。跳出while, 结束本次waiting向running的调度
            if num_new_tokens == 0 or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                              num_new_seqs=num_new_seqs):
                break

            # Can schedule this request.
            # lora相关，忽略
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)

            # 走到这一步时，说明当前seq_group已经通过上述种种验证，可以被加入running队列进行推理
            # 先将其从waiting队列中移出
            waiting_queue.popleft()
            # 为当前seq_group分配物理块,并将该seq_group中每条seq的status从waiting改为running
            self._allocate_and_set_running(seq_group)

            # ScheduledSequenceGroup类似于C++结构体。仅包含seq_group和token_chunk_size两个变量
            # 搞不懂vllm为什么总喜欢这种包裹操作，在各处代码中随处可见。用基本的list,或dict不好吗！
            seq_groups.append(
                    ScheduledSequenceGroup(seq_group=seq_group,
                                           token_chunk_size=num_new_tokens))

            # 当前seq_group的tokens和seqs数量增加到预算budget中
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        # 和lora相关的操作，忽略
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
                seq_groups=seq_groups,
                ignored_seq_groups=ignored_seq_groups,
                num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

```

---

#### 6.3 _schedule_running

代码：

```python	
    def _schedule_running(
            self,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> SchedulerRunningOutputs:
        # Blocks that need to be swapped or copied before model execution.
        # todo 类型变了
        # blocks_to_swap_out：{gpu物理块id: cpu物理块id}
        # blocks_to_copy: {旧物理块id：[由旧物理块copy-on-write而来的新物理块id]}
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.

        running_queue = self.running

        while running_queue:
            seq_group = running_queue[0]
            # 当前待推理的seq_group需要处理,或者说准备返回的tokens数量,对于RUNNING状态，每个seq返回1个token
            num_running_tokens = self._get_num_new_tokens(
                    seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            # todo 觉得这个判断有点多余,因为处于RUNNING状态的seq,必定有tokens返回,prompt总不能为空吧，num_running_tokens
            # todo 不可能为0, 再说,如果为0, 方法self._get_num_new_tokens内部就会抛出异常,因为做了assert断言
            if num_running_tokens == 0:
                break

            # 经过num_running_tokens检验没问题后, 将该seq_group从running_queue中取出来
            running_queue.popleft()

            # 对于这个seq_group，检查对于其中的每一个seq，是否能至少分配一个物理块给它，如果不能的话
            # （说明要执行抢占操作了，否则马上会没有资源让这个最早到达的seq_group做完推理）：
            # 这里用了while...else，如果while条件正常结束，则进入else内容；如果被break，则不会执行else
            while not self._can_append_slots(seq_group):  # 如果不能为当前seq_group的每个seq都分配一个block
                # 这个seq_group本来是要送去做推理的,但没有足够的gpu物理blocks分配给它
                # 根据vllm的调度原则，这个seq_group要被优先处理，没有足够资源，就把running队列最后位置的
                # seq_group踢出去，释放gpu blocks给当前seq_group使用。

                # seq_group准备返回的tokens数量已经加到budget属性上,现在不处理它, 要把数量再减回来
                # budget会记录每次+-数量的seq_group.request_id,如果以前没被+过，现在就不会被-，就像下面的调用一样
                budget.subtract_num_batched_tokens(seq_group.request_id, num_running_tokens)
                # 在外层总调度中，已经在budget汇总了所有正在活跃的seqs数量，现在要减去属于该seq_group的seqs数量
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id, num_running_seqs)

                # lora相关,忽略
                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # ------------------------------------------------------------------------------------------------------
                # 经过以下释放gpu blocks工作后,再次进入while循环判断gpu blocks数量是否够用,
                # 如果够用,进入到与while对应的else分支,如果不够用,继续释放gpu blocks,直到够用或running_queue全部取完.
                # ------------------------------------------------------------------------------------------------------

                # 如果此时running_queue队列不为空,把最后一个seq_group踢出去放入swap队列,给
                # 上面这个seq_group腾位置(释放最后一个seq_group对应的gpu blocks)
                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()

                    # 有两种swap方式,RECOMPUTE:删除所有,回炉到waiting队列. SWAP:将blocks全部转移到CPU blocks上
                    preempted_mode = self._preempt(victim_seq_group, blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                # 如果running_queue队列已经空了,没有替罪的羊,只能把自己放入swap队列了.
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group, blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    # 此时running_queue队列已空,已经没有seq_group可处理了,使用break中断
                    # while循环, 不走后面的else分支,直接return,而且本次调度没有指定任何待推理的seq_group
                    break
            else:
                # 为当前seq_group分配gpu 物理blocks. 这里只分配了逻辑blocks与物理blocks的映射关系
                # blocks_to_copy:[旧物理块id, copy - on - write而来的新物理块id]
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(ScheduledSequenceGroup(seq_group=seq_group,
                                                                     token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(ScheduledSequenceGroup(seq_group=seq_group,
                                                                    token_chunk_size=1))
                # todo 这似乎是个bug, 如果_can_append_slots为True，会跳过while直接走当前else分支
                # todo seqs在外层_schedule_default已经更新过，所以这里只更新tokens就好
                # todo 但是，如果_can_append_slots为False，budget会同时减去seq_group的tokens和seqs数量
                # todo 下面把tokens再加回来，逻辑没问题，但没有更新seqs！
                budget.add_num_batched_tokens(seq_group.request_id, num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                # 默认情况下, 以下两个if都走不到
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        return SchedulerRunningOutputs(
                decode_seq_groups=decode_seq_groups,
                prefill_seq_groups=prefill_seq_groups,
                preempted=preempted,
                swapped_out=swapped_out,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=False))

```



#### 6.4 _schedule_swapped

**代码**：

```python
def _schedule_swapped(
            self,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        # Blocks that need to be swapped or copied before model execution.
        # [(cpu物理块id, gpu物理块id)]
        blocks_to_swap_in: List[Tuple[int, int]] = []
        # [(旧物理块,copy - on - write而来的新物理块id)]
        blocks_to_copy: List[Tuple[int, int]] = []
        # 准备解码的seq_group
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        # 准备预填充的seq_group
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        # 因各种原因，被标记为不再处理的seq_group，如预填充序列太长了...
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            # 取出swap队列中最早被抢占的seq_group
            seq_group = swapped_queue[0]

            # ----------------------------------------------------------------------------------------------------------
            # If the sequence group cannot be swapped in, stop.
            # 对被抢占seq_group有两种处理方式，1. 清空放入waiting队列，这时is_prefill为True
            # 2.blocks全部转移到CPU上，这时is_prefill为False
            # self._get_num_lookahead_slots(is_prefill)必定为0，否则抛出异常，block_manager_v1不支持非0情况
            is_prefill = seq_group.is_prefill()
            # 根据需要的，与可用的物理blocks数量判断，是否可以把当前seq_group从swap队列转移到running队列
            alloc_status = self.block_manager.can_swap_in(
                    seq_group, self._get_num_lookahead_slots(is_prefill))
            if alloc_status == AllocStatus.LATER:    # 稍后，资源多时再处理
                break
            elif alloc_status == AllocStatus.NEVER:  # 不合格，永不再处理
                logger.warning(
                        "Failing the request %s because there's not enough kv "
                        "cache blocks to run the entire sequence.",
                        seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue
            # ----------------------------------------------------------------------------------------------------------

            # lora相关，忽略
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            # 取出这个seq_group在剩余生命周期内将并行运行的最大seq数量
            num_new_seqs = seq_group.get_max_num_running_seqs()
            # 当前准备转移的seq_group,需要处理,或者说准备返回的tokens数量，
            # decode模式：每个seq num_token=1,其他模式则遵循各自的状态
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)
            # 感觉num_new_tokens==0的判断有点多余，基本不可能为0
            # budget.can_schedule 会判断加上当前seq_group的num_new_tokens和num_new_seqs后
            # 总数是否会超标，如果超标，说明不能再添加任何seq_group到running队列，直接结束本次调度
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens, num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)

            # 如果能走到这步，说明可向running队列转移了。先把当前seq_group从swap队列踢出来
            # 再把CPU上的blocks转移到GPU block上
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            # 判断是不是预填充，将这个seq_group加入不同的分组
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(ScheduledSequenceGroup(seq_group, token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(ScheduledSequenceGroup(seq_group, token_chunk_size=1))

            # 将这个马上上岸的seq_group的tokens和seqs数量更新到budget中
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
                decode_seq_groups=decode_seq_groups,
                prefill_seq_groups=prefill_seq_groups,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_copy=blocks_to_copy,
                num_lookahead_slots=self._get_num_lookahead_slots(
                        is_prefill=False),
                infeasible_seq_groups=infeasible_seq_groups,
        )
```

---

#### 






















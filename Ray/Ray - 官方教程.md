> 参考文档：
>
> [Ray: A Distributed Framework for Emerging AI Applications](./PDF/Ray-Paper.pdf)
>
> [Ray - doc](https://docs.ray.io/en/latest/)



## 1 安装

| 命令                               | 已安装组件                                              |
| ---------------------------------- | ------------------------------------------------------- |
| `pip install -U "ray"`             | 核心                                                    |
| `pip install -U "ray[default]"`    | 核心, 仪表盘, 集群启动器                                |
| `pip install -U "ray[data]"`       | 核心, 数据                                              |
| `pip install -U "ray[train]"`      | 核心, 训练                                              |
| `pip install -U "ray[tune]"`       | 核心, 调优                                              |
| `pip install -U "ray[serve]"`      | 核心, 仪表盘, 集群启动器, 服务                          |
| `pip install -U "ray[serve-grpc]"` | 核心, 仪表盘, 集群启动器, 支持 gRPC 的服务              |
| `pip install -U "ray[rllib]"`      | 核心, 调优, RLlib                                       |
| `pip install -U "ray[all]"`        | 核心, 仪表盘, 集群启动器, 数据, 训练, 调优, 服务, RLlib |

你可以组合安装额外功能。例如，要安装带有仪表板、集群启动器和训练支持的 Ray，你可以运行：

```bash
pip install -U "ray[default,train]"
```



## 2 Ray Core

首先通过命令启动本地集群

```bash
ray start --head
```

### 2.1 TASK（任务）

Ray 使得任意函数能够在独立的 Python 工作者上异步执行。这些函数被称为 **Ray 远程函数**，它们的异步调用被称为 **Ray 任务**。

```python
import ray
import time


# A regular Python function.
def normal_function():
    return 1


# By adding the `@ray.remote` decorator, a regular Python function
# becomes a Ray remote function.
@ray.remote
def my_function():
    return 1


# To invoke this remote function, use the `remote` method.
# This will immediately return an object ref (a future) and then create
# a task that will be executed on a worker process.
obj_ref = my_function.remote()

# The result can be retrieved with ``ray.get``.
assert ray.get(obj_ref) == 1


@ray.remote
def slow_function():
    time.sleep(10)
    return 1


# Ray tasks are executed in parallel.
# All computation is performed in the background, driven by Ray's internal event loop.
for _ in range(4):
    # This doesn't block.
    slow_function.remote()
```

**【注意】**：

* 当slow_function.remote()远程调用后，如果不执行ray.get()来拿这个值，发现这些任务是执行失败的。

通过命令行查看正在运行和已完成的任务及其数量：

```bash
# This API is only available when you download Ray via `pip install "ray[default]"`
ray summary tasks
```

**指定所需资源：**

```python
import ray


# Specify required resources.
@ray.remote(num_cpus=4, num_gpus=2)
def my_function():
    return 1


# Override the default resource requirements.
obj_ref = my_function.options(num_cpus=3).remote()
value = ray.get(obj_ref)
```

**【注意】**：此时只会重置num_cpus的数量，num_gpus还是为2个。

如果资源不足：

```bash
❯ python test.py
2025-01-09 09:51:42,702 INFO worker.py:1636 -- Connecting to existing Ray cluster at address: 127.0.0.1:6379...
2025-01-09 09:51:42,706 INFO worker.py:1812 -- Connected to Ray cluster. View the dashboard at http://127.0.0.1:8265 
(autoscaler +4s) Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.
(autoscaler +4s) Error: No available node types can fulfill resource request {'CPU': 3.0, 'GPU': 2.0}. Add suitable node types to this cluster to resolve this issue.
(autoscaler +39s) Error: No available node types can fulfill resource request {'CPU': 3.0, 'GPU': 2.0}. Add suitable node types to this cluster to resolve this issue.
(autoscaler +1m14s) Error: No available node types can fulfill resource request {'CPU': 3.0, 'GPU': 2.0}. Add suitable node types to this cluster to resolve this issue.
(autoscaler +1m50s) Error: No available node types can fulfill resource request {'CPU': 3.0, 'GPU': 2.0}. Add suitable node types to this cluster to resolve this issue.
```

在Mac上改成如下代码可以正常运行：

```python
import ray


# Specify required resources.
@ray.remote(num_cpus=4, num_gpus=2)
def my_function():
    return 1


# Override the default resource requirements.
obj_ref = my_function.options(num_cpus=2, num_gpus=0).remote()
value = ray.get(obj_ref)
print(a)

```

【注意】：

* 这里的num_cpus限制的是CPU核心数，而非物理CPU个数。

* 这里的num_gpus限制的是物理GPU个数，而非GPU核心数。

**将对象引用传递给Ray的Task：**

```python
import ray


# Specify required resources.
@ray.remote(num_cpus=4)
def my_function():
    return 1


@ray.remote
def function_with_an_argument(value):
    return value + 1


obj_ref1 = my_function.remote()
assert ray.get(obj_ref1) == 1

# You can pass an object ref as an argument to another Ray task.
obj_ref2 = function_with_an_argument.remote(obj_ref1)
assert ray.get(obj_ref2) == 2
```

【注意】

* 由于第二个任务依赖于第一个任务的输出，Ray 将不会执行第二个任务，直到第一个任务完成。
* 如果这两个任务被安排在不同的机器上，第一个任务的输出（对应于 `obj_ref1/objRef1` 的值）将通过网络发送到第二个任务被安排的机器上。

**等待部分结果：**

```python
object_refs = [slow_function.remote() for _ in range(2)]
# Return as soon as one of the tasks finished execution.
ready_refs, remaining_refs = ray.wait(object_refs, num_returns=1, timeout=None)
```

**多重返回：**

```python
# By default, a Ray task only returns a single Object Ref.
@ray.remote
def return_single():
    return 0, 1, 2


object_ref = return_single.remote()
assert ray.get(object_ref) == (0, 1, 2)


# However, you can configure Ray tasks to return multiple Object Refs.
@ray.remote(num_returns=3)
def return_multiple():
    return 0, 1, 2


object_ref0, object_ref1, object_ref2 = return_multiple.remote()
assert ray.get(object_ref0) == 0
assert ray.get(object_ref1) == 1
assert ray.get(object_ref2) == 2
```

对于返回多个对象的任务，Ray 还支持远程生成器，允许任务一次返回一个对象以减少工作线程的内存使用。Ray 还支持动态设置返回值数量的选项，这在任务调用者不知道预期有多少返回值时非常有用。

```python
@ray.remote(num_returns=3)
def return_multiple_as_generator():
    for i in range(3):
        yield i


# NOTE: Similar to normal functions, these objects will not be available
# until the full task is complete and all returns have been generated.
a, b, c = return_multiple_as_generator.remote()
```

**取消任务ray.cancel()：**

```python
@ray.remote
def blocking_operation():
    time.sleep(10e6)


obj_ref = blocking_operation.remote()
ray.cancel(obj_ref)

try:
    ray.get(obj_ref)
except ray.exceptions.TaskCancelledError:
    print("Object reference was cancelled.")
```

**嵌套远程函数：**

```python
import ray


@ray.remote
def f():
    return 1


@ray.remote
def g():
    # Call f 4 times and return the resulting object refs.
    return [f.remote() for _ in range(4)]


@ray.remote
def h():
    # Call f 4 times, block until those 4 tasks finish,
    # retrieve the results, and return the values.
    return ray.get([f.remote() for _ in range(4)])
  
print(ray.get(g.remote()))
print(ray.get(h.remote()))
```

**【注意】**：

* `f` 的定义必须出现在 `g` 和 `h` 的定义之前，因为一旦 `g` 被定义，它将被序列化并发送给工作进程，因此如果 `f` 尚未定义，定义将是不完整的。

Ray 在阻塞时会释放 CPU 资源。这可以防止死锁情况，即嵌套任务等待父任务持有的 CPU 资源。考虑以下远程函数。

```python
@ray.remote(num_cpus=1, num_gpus=1)
def g():
    return ray.get(f.remote())
```

当一个 `g` 任务正在执行时，它会在调用 `ray.get` 时释放其CPU资源。当 `ray.get` 返回时，它将重新获取CPU资源。在整个任务的生命周期内，它将保留其GPU资源，因为任务很可能会继续使用GPU内存。

**动态生成器实现：当返回值的数量由远程函数动态设置，而不是由调用者设置时。**

**场景1：num_returns由任务调用者设置**

在可能的情况下，调用者应使用 `@ray.remote(num_returns=x)` 或 `foo.options(num_returns=x).remote()` 设置远程函数的返回值数量。Ray 将向调用者返回这些数量的 `ObjectRefs`。远程任务应返回相同数量的值，通常作为元组或列表。与动态设置返回值数量相比，这减少了用户代码的复杂性和性能开销，因为 Ray 会提前确切知道需要向调用者返回多少个 `ObjectRefs`。

在不改变调用者的语法的情况下，我们也可以使用一个远程生成器函数来迭代地生成值。生成器应生成与调用者指定的返回值数量相同的值，这些值将逐一存储在Ray的对象存储中。对于生成与调用者指定数量不同的值的生成器，将会引发错误。

例如，我们可以交换以下返回返回值列表的代码：

```python
import numpy as np


@ray.remote
def large_values(num_returns):
    return [
        np.random.randint(np.iinfo(np.int8).max, size=(100_000_000, 1), dtype=np.int8)
        for _ in range(num_returns)
    ]
```

对于这段代码，它使用了一个生成器函数：

```python
@ray.remote
def large_values_generator(num_returns):
    for i in range(num_returns):
        yield np.random.randint(
            np.iinfo(np.int8).max, size=(100_000_000, 1), dtype=np.int8
        )
        print(f"yielded return value {i}")
```

这样做的好处是生成器函数不需要一次性在内存中保存所有返回值。它可以一次生成一个数组，从而减少内存压力。

**场景2：num_returns由任务执行器设置**

在某些情况下，调用者可能不知道从远程函数中期望的返回值数量。例如，假设我们想要编写一个任务，将其参数分解为大小相等的块并返回这些块。我们可能在执行任务之前不知道参数的大小，因此我们不知道期望的返回值数量。

在这些情况下，我们可以使用一个返回*动态*数量值的远程生成器函数。要使用此功能，请在`@ray.remote`装饰器或远程函数的``.options()``中设置`num_returns=”dynamic”`。然后，在调用远程函数时，Ray将返回单一的`ObjectRef`，该`ObjectRef`将在任务完成时填充一个`DynamicObjectRefGenerator`。`DynamicObjectRefGenerator`可用于遍历包含任务返回的实际值的``ObjectRefs``列表。

```python
import numpy as np


@ray.remote(num_returns="dynamic")
def split(array, chunk_size):
    while len(array) > 0:
        yield array[:chunk_size]
        array = array[chunk_size:]


array_ref = ray.put(np.zeros(np.random.randint(1000_000)))
block_size = 1000

# Returns an ObjectRef[DynamicObjectRefGenerator].
dynamic_ref = split.remote(array_ref, block_size)
print(dynamic_ref)
# ObjectRef(c8ef45ccd0112571ffffffffffffffffffffffff0100000001000000)

i = -1
ref_generator = ray.get(dynamic_ref)
print(ref_generator)
# <ray._raylet.DynamicObjectRefGenerator object at 0x7f7e2116b290>
for i, ref in enumerate(ref_generator):
    # Each DynamicObjectRefGenerator iteration returns an ObjectRef.
    assert len(ray.get(ref)) <= block_size
num_blocks_generated = i + 1
array_size = len(ray.get(array_ref))
assert array_size <= num_blocks_generated * block_size
print(f"Split array of size {array_size} into {num_blocks_generated} blocks of "
      f"size {block_size} each.")
# Split array of size 63153 into 64 blocks of size 1000 each.

# NOTE: The dynamic_ref points to the generated ObjectRefs. Make sure that this
# ObjectRef goes out of scope so that Ray can garbage-collect the internal
# ObjectRefs.
del dynamic_ref
```

我们也可以将带有 `num_returns="dynamic"` 的任务返回的 `ObjectRef` 传递给另一个任务。该任务将接收到 `DynamicObjectRefGenerator`，它可以用来迭代任务的返回值。同样，你也可以将 `ObjectRefGenerator` 作为任务参数传递。

```python
@ray.remote
def get_size(ref_generator : DynamicObjectRefGenerator):
    print(ref_generator)
    num_elements = 0
    for ref in ref_generator:
        array = ray.get(ref)
        assert len(array) <= block_size
        num_elements += len(array)
    return num_elements


# Returns an ObjectRef[DynamicObjectRefGenerator].
dynamic_ref = split.remote(array_ref, block_size)
assert array_size == ray.get(get_size.remote(dynamic_ref))
# (get_size pid=1504184)
# <ray._raylet.DynamicObjectRefGenerator object at 0x7f81c4250ad0>

# This also works, but should be avoided because you have to call an additional
# `ray.get`, which blocks the driver.
ref_generator = ray.get(dynamic_ref)
assert array_size == ray.get(get_size.remote(ref_generator))
# (get_size pid=1504184)
# <ray._raylet.DynamicObjectRefGenerator object at 0x7f81c4251b50>
```

**场景3：异常处理**

如果一个生成器函数在产生所有值之前引发异常，它已经存储的值仍然可以通过它们的 `ObjectRefs` 访问。剩余的 `ObjectRefs` 将包含引发的异常。这对于静态和动态的 `num_returns` 都是成立的。如果任务是以 `num_returns="dynamic"` 调用的，异常将被存储为 `DynamicObjectRefGenerator` 中的一个额外的最终 `ObjectRef`。

```python
@ray.remote
def generator():
    for i in range(2):
        yield i
    raise Exception("error")


ref1, ref2, ref3, ref4 = generator.options(num_returns=4).remote()
assert ray.get([ref1, ref2]) == [0, 1]
# All remaining ObjectRefs will contain the error.
try:
    ray.get([ref3, ref4])
except Exception as error:
    print(error)

dynamic_ref = generator.options(num_returns="dynamic").remote()
ref_generator = ray.get(dynamic_ref)
ref1, ref2, ref3 = ref_generator
assert ray.get([ref1, ref2]) == [0, 1]
# Generators with num_returns="dynamic" will store the exception in the final
# ObjectRef.
try:
    ray.get(ref3)
except Exception as error:
    print(error)
```

### 2.2 ACTOR（执行器）

Actor 将 Ray API 从函数（任务）扩展到类。Actor 本质上是一个有状态的工作者（或服务）。当一个新的 Actor 被实例化时，会创建一个新的工作者，并且 Actor 的方法会被调度到该特定工作者上，并且可以访问和修改该工作者的状态。与任务类似，Actor 支持 CPU、GPU 和自定义资源需求。

```python
import ray

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

# Create an actor from this class.
counter = Counter.remote()
```

**查看Actor状态：**

```bash
# This API is only available when you install Ray with `pip install "ray[default]"`.
ray list actors
```

**指定所需资源：**

```python
# Specify required resources for an actor.
@ray.remote(num_cpus=2, num_gpus=0.5)
class Actor:
    pass
```

**调用Actor：**

```python
# Call the actor.
obj_ref = counter.increment.remote()
print(ray.get(obj_ref)) # print 1
```

在不同参与者上调用的方法可以并行执行，而在同一参与者上调用的方法则按调用顺序串行执行。同一参与者上的方法将共享状态，如下所示。

```python
# Create ten Counter actors.
counters = [Counter.remote() for _ in range(10)]

# Increment each Counter once and get the results. These tasks all happen in
# parallel.
results = ray.get([c.increment.remote() for c in counters])
print(results) # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Increment the first Counter five times. These tasks are executed serially
# and share state.
results = ray.get([counters[0].increment.remote() for _ in range(5)])
print(results) # [2, 3, 4, 5, 6]
```

**传递Actor：**

将Actor传递到其他任务中

```python
import time

@ray.remote
def f(counter):
    for _ in range(10):
        time.sleep(0.1)
        counter.increment.remote()
```

如果我们实例化一个Actor，我们可以将Actor传递给各种任务。

```python
counter = Counter.remote()

# Start some tasks that use the actor.
[f.remote(counter) for _ in range(3)]

# Print the counter value.
for _ in range(10):
    time.sleep(0.1)
    print(ray.get(counter.get_counter.remote()))
```

**取消Actor任务：**

```python
import ray
import asyncio
import time


@ray.remote
class Actor:
    async def f(self):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            print("Actor task canceled.")


actor = Actor.remote()
ref = actor.f.remote()

# Wait until task is scheduled.
time.sleep(1)
ray.cancel(ref)

try:
    ray.get(ref)
except ray.exceptions.RayTaskError:
    print("Object reference was cancelled.")
```

在 Ray 中，任务取消行为取决于任务的当前状态：

* **未调度的任务**：如果演员任务尚未被调度，Ray 尝试取消调度。在此阶段成功取消时，调用 `ray.get(actor_task_ref)` 会产生一个 `TaskCancelledError`。

* **运行Actor任务（常规Actor，线程化Actor）**：对于分类为单线程Actor或多线程Actor的任务，Ray不提供中断机制。

* **运行异步Actor任务**: 对于分类为 `异步Actor <_async-actors>` 的任务，Ray 试图取消相关的 `asyncio.Task`。这种取消方法符合 asyncio 任务取消 中提出的标准。请注意，如果你不在异步函数中 `await`，`asyncio.Task` 在执行过程中不会被中断。

* **取消保证**：Ray 尝试以 *尽力而为* 的方式取消任务，这意味着取消并不总是得到保证。例如，如果取消请求未能传达给执行者，任务可能不会被取消。你可以使用 `ray.get(actor_task_ref)` 检查任务是否成功取消。

* **递归取消**：Ray 跟踪所有子任务和 Actor 任务。当给出 `recursive=True` 参数时，它会取消所有子任务和 Actor 任务。

**终止Actor：**这将导致该actor立即退出其进程，导致任何当前、待处理和未来的任务因 `RayActorError` 而失败。

```python
import ray

@ray.remote
class Actor:
    pass

actor_handle = Actor.remote()

ray.kill(actor_handle)
# This will not go through the normal Python sys.exit
# teardown logic, so any exit handlers installed in
# the actor using ``atexit`` will not be called.
```

**在Actor中手动终止：**这种终止方法会等待所有先前提交的任务执行完毕，然后使用 sys.exit 优雅地退出进程。

```python
@ray.remote
class Actor:
    def exit(self):
        ray.actor.exit_actor()

actor = Actor.remote()
actor.exit.remote()
```

**AsyncIO/Actor的并发性（感觉用处不多）：**

Python 的全局解释器锁 (GIL) 将只允许一个 Python 代码线程同时运行，这意味着如果你只是并行化 Python 代码，你不会得到真正的并行性。如果你调用 Numpy、Cython、Tensorflow 或 PyTorch 代码，这些库在调用 C/C++ 函数时会释放 GIL。

**【注意】**：

* **Ray 的 Worker 是单独的 Python 进程**。每个 Worker 运行在自己的进程空间中，避免任务之间的内存或状态冲突，提高稳定性。默认会根据CPU的核心数启动对应数量的Worker。
* 由于 Ray 进程不共享内存空间，工作人员和节点之间传输的数据将需要 **序列化** 和 **反序列化**。

```python
import ray
import asyncio

@ray.remote
class AsyncActor:
    # multiple invocation of this method can be running in
    # the event loop at the same time
    async def run_concurrent(self):
        print("started")
        await asyncio.sleep(2) # concurrent workload here
        print("finished")

actor = AsyncActor.remote()

# regular ray.get
ray.get([actor.run_concurrent.remote() for _ in range(4)])

# async ray.get
async def async_get():
    await actor.run_concurrent.remote()
asyncio.run(async_get())
```

**Q：worker 和 actor 之间有什么区别？**

每个“Ray Worker”是一个Python进程。

Worker 在 Task 和 Actor 中受到不同的对待。任何“Ray Worker“要么 1. 用于执行多个 Ray Task，要么 2. 作为专用 Ray Actor启动。

* Task：当 Ray 在一台机器上启动时，会自动启动多个 Ray 工作进程（默认情况下每个 CPU 一个）。它们将用于执行任务（类似于进程池）。如果你执行 8 个任务，每个任务使用 `num_cpus=2`，并且总 CPU 数为 16（`ray.cluster_resources()["CPU"] == 16`），你最终会有 8 个工作进程闲置。
* Actor：Ray Actor 也是一个 “Ray Worker”，但在运行时实例化（通过 `actor_cls.remote()`）。它的所有方法都将在同一进程中运行，使用相同的资源（在定义 Actor 时指定）。请注意，与任务不同，运行 Ray Actor 的 Python 进程不会被重用，当 Actor 被删除时，这些进程将被终止。

### 2.3 Objects（对象）

在 Ray 中，任务和角色创建并计算对象。我们称这些对象为 *远程对象* ，因为它们可以存储在 Ray 集群的任何地方，我们使用 *对象引用* 来引用它们。远程对象缓存在 Ray 的分布式共享内存对象存储中，集群中的每个节点都有一个对象存储。在集群设置中，一个远程对象可以存在于一个或多个节点上，与持有对象引用的对象无关。

**Ray的序列化机制：**

**1. 使用 `cloudpickle` 进行序列化：**

- Ray 默认使用 **`cloudpickle`** 序列化 Python 对象。
- `cloudpickle` 是一种比 Python 自带的 `pickle` 更强大的序列化库，能够支持几乎所有 Python 对象的序列化，包括函数、类、闭包等。

**2. 数据的高效序列化：**

- 对于简单的 Python 数据（如 `int`、`float`、`str` 等），Ray 直接使用高效的二进制格式进行存储。
- 对于复杂的数据结构（如 NumPy 数组、Pandas DataFrame 等），Ray 会直接序列化底层的二进制内存（如共享内存区域），避免数据拷贝，从而提高性能。

### 2.4 Environment Dependencies（环境依赖）

您的 Ray 应用程序可能有一些依赖项存在于您的 Ray 脚本之外。例如：

- 您的 Ray 脚本可能会导入/依赖于某些 Python 包。
- 您的 Ray 脚本可能正在寻找某些特定的环境变量以使其可用。
- 您的 Ray 脚本可能会导入脚本之外的一些文件。

在集群上运行时经常遇到的一个问题是，Ray 期望这些“依赖项”存在于每个 Ray 节点上。如果这些依赖项不存在，您可能会遇到诸如 `ModuleNotFoundError`、`FileNotFoundError` 等问题。

要解决这个问题，你可以 (1) 提前在集群上准备你的依赖项（例如使用容器镜像）使用 Ray [集群启动器](https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start)，或者 (2) 使用 Ray 的 [运行时环境](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments) 来即时安装它们。

......

### 2.5 Scheduling（调度）





### 2.6 Fault Tolerance（容错性）



### 2.7 Design Patterns && Anti-Patterns（设计模型与反模式）



### 2.8 Advanced Topics（高级主题）








# Dag 关键特性：

- Lazy Computation Graphs： 懒计算模式，即可以等所有task/actor定义完之后再执行，方便做图优化
- Custom Input Node: 支持数据变但计算图不变，避免重复建图
- Multiple Output Node: 计算图不变，但支持多输出(不清楚内部是并行执行两个graph还是batch 模式)
- Reuse Ray Actors in DAGs：通过调用.remote() ，避免actor在graph执行完成后被销毁



# Dag 使用示例

```python
import ray
from ray.dag.input_node import InputNode
from ray.dag.output_node import MultiOutputNode

@ray.remote
class Worker:
    def __init__(self):
        self.forwarded = 0

    def forward(self, input_data: int):
        self.forwarded += 1
        return input_data + 1

    def num_forwarded(self):
        return self.forwarded

# Create an actor via ``remote`` API not ``bind`` API to avoid
# killing actors when a DAG is finished.
worker = Worker.remote()

with InputNode() as input_data:
    dag = MultiOutputNode([worker.forward.bind(input_data)])

# Actors are reused. The DAG definition doesn't include
# actor creation.
assert ray.get(dag.execute(1)) == [2]
assert ray.get(dag.execute(2)) == [3]
assert ray.get(dag.execute(3)) == [4]

# You can still use other actor methods via `remote` API.
assert ray.get(worker.num_forwarded.remote()) == 3
```






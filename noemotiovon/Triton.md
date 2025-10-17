Torch的compile，可以选择后端，例如inductor。

```python
def func(a, b):
	x = a.ceil()+b.floor()
	y = x.sum(dim=-1)
	z = y.softmax(dim=-1)
	w = (z * 10) ** 3
	return w

torch.compile(func, backend="inductor")
```



Python(dynamo)->Dynamo捕获FX图，并切割子图->AOT_Autograd生成前向/后向图->Inductor进行算子融合，子图编译



Decompose 需要拆分的算子（大算子）

Lowering 需要融合的算子（小算子）

Fallback 无法融合的算子

大算子拆分为小算子，所有的小算子进行融合，形成融合算子；无法参与融合的算子就保持原样（大算子也可能拆分出无法融合的小算子）。最后根据融合算子和无法融合的算子进行编译。



XBLOCK = 数据长度 / vector kernels

RBLOCK = 规约轴的长度

需要缓存的大小 = XBLOCK * RBLOCK * 变量个数

UB（unified buffer）：片上缓存尺寸（910B：192K）



910B芯片


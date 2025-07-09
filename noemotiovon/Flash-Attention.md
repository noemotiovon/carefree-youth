# Online-Softmax 演进

## Softmax

$$
\tilde{x}_i = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}
$$

## Safe Softmax

为避免指数计算数值溢出，引入 “safe” 版本：

定义最大值：
$$
M = \max(x_{1:N})
$$
Safe Softmax：
$$
\tilde{x}_i = \frac{e^{x_i - M}}{\sum_{j=1}^N e^{x_j - M}}
$$
证明其与普通 softmax 等价：
$$
\begin{aligned} \tilde{x}_i &= \frac{e^{x_i - M}}{\sum_{j=1}^N e^{x_j - M}} = \frac{e^{x_i}/e^M}{\sum_{j=1}^N e^{x_j}/e^M} = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}} \end{aligned}
$$


## Online Softmax

对于前  $N$ 个元素而言：

最大值：
$$
M_N = \max(x_{1:N})
$$
指数和：
$$
l_N = \sum_{j=1}^N e^{x_j - M_N}
$$


在 Decode 阶段，会新增一个元素 $x_{N+1}$，对于前 $N+1$ 个元素而言：

最大值：
$$
M_{N+1} = \max(M_N, x_{N+1})
$$
指数和：
$$
\begin{aligned}l_{N+1} &= \sum_{j=1}^{N+1} e^{x_j - M_{N+1}} \\ &= l_N \cdot e^{M_N - M_{N+1}} + e^{x_{N+1} - M_{N+1}} \end{aligned}
$$


最终归一化：
$$
\tilde{x}_i = \frac{e^{x_i - M_{N+1}}}{l_{N+1}}
$$

## Block Online Softmax

假设将张量分为两个块 t，对每个块 $t$ 计算 $m^{(t)}, l^{(t)}$

全局归并：
$$
m = \max(m^{(1)}, m^{(2)})
$$

$$
l = l^{(1)} \cdot e^{m^{(1)} - m} + l^{(2)} \cdot e^{m^{(2)} - m}
$$

最后统一计算各元素：
$$
\tilde{x}_i = \frac{e^{x_i - m}}{l}
$$
到这里我们会发现，对于整块的 Softmax 函数可以由分块计算的最大值和指数和来计算得出，也为后续的 Flash-Attention 的分块奠定了基础。




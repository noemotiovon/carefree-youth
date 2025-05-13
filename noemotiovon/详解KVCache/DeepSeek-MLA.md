# 1 LLM 推理

LLM 推理分为两个阶段：Prefill，Decode。

Prefill：模型对全部的 Prompt Tokens 进行一次并行计算，生成第一个输出 Token。

Decode：根据之前产生的所有 Token，自回归的生成下一个 Token，直至生成EOS（end-of-sequence）。

简单介绍下LLM推理中的常见模块和层：

## Embedding

假设输入为“我想吃酸菜鱼”，以第 i 个字符“鱼”为例，假设其在词汇表中对应的 index 为 j。
$$
input\_embedding_i = token\_embedding_j + position\_embedding_i
$$
其中：i 是对应 Token 的位置，j 为词汇表对应的 index。

假设嵌入深度为 d，Prompt 长度为 s，则对应的 input_embedding 的矩阵维度为为**[s, d]**。其中 s = 6，d = 768。

![img](images/01-image.webp)

## Transformer 模块

一个Transformer 模块包括多个层，Layer Normal -> Multi-Head Self Attention -> Projection -> Layer Normal -> MLP。

Embedding 层得到的 input_embedding 矩阵是 Transformer 层的输入。

![img](images/02-image.png)

### Layer Normal

归一化方法有很多，其目标是使得每列的数值均值为0，标准差为1，通常通过计算每列的均值（Mean）和标准差（Std dev），然后让每一列减去相应的均值，并除以相应的标准差。layer_norm 输出的矩阵维度。

设输入矩阵为 ($ X \in \mathbb{R}^{s \times d} $)，其中 s 是样本数，d 是特征维度。对每一列进行归一化处理，即对各个 token 的每一个特征进行归一化：

#### 1 计算均值和方差

$$
\mu_j = \mathbb{E}[X_{:, j}] = \frac{1}{s} \sum_{i=1}^{s} X_{i,j}
$$

$$
\sigma_j^2 = \text{Var}[X_{:, j}] = \frac{1}{s} \sum_{i=1}^{s} (X_{i,j} - \mu_j)^2
$$

#### 2 归一化

$$
\hat{X}_{i,j} = \frac{X_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

其中 $\epsilon = 10^{-5}$ 是用于防止除以零的常数。

#### 3 缩放和平移

$$
Y_{i,j} = \gamma_j \hat{X}_{i,j} + \beta_j
$$

其中 $\gamma_j$ 和 $\beta_j$ 的维度是[1, d]，并且是可学习的，用于每列的缩放和平移。

最终，经过归一化输出的矩阵，仍与 input_embedding 的维度相同，为**[s, d]**，为用于每列的缩放和平移。

### Multi-Head Self-Attention

self-attention层或许是Transformer中最核心的部分，此时 input_embedding 中的各个列开始交流。假设有 h 个头，则每个头处理 h 分之一的嵌入深度，每个头对应的输入维度为[s, n / h]。

设输入为归一化后的 embedding 矩阵 $X \in \mathbb{R}^{s \times d}$，其中 s 是序列长度，d 是特征维度。在接下来的处理中，我们只关注其中一个头的处理，假设 h = 1。

#### 1 生成 Q，K，V 矩阵

$$
Q = XW^Q + b^Q,\quad K = XW^K + b^K,\quad V = XW^V + b^V
$$



$W^Q, W^K, W^V \in \mathbb{R}^{d \times d_h}$ 是可学习的投影矩阵，每个头有自己的 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_h}$；

- $b^Q, b^K, b^V \in \mathbb{R}^{d_h}$ 是偏置项；
- $d_h$ 是每个注意力头的维度。

在实际的工程中，是通过直接乘以一个大矩阵 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$，再进行“分头”处理。

**Q，K，V 矩阵的维度为[s, $d_h$]。**

![img](images/03-image.webp)

#### 2 计算注意力分数

$$
\text{AttentionScores} = \frac{QK^\top}{\sqrt{d_h}}
$$

其中 $\sqrt{d_h}$ 是缩放因子，用于避免点积过大导致 softmax 饱和。

AttentionScores的维度**[s, s]**。

![img](images/04-image.webp)

如图所示：“我”字符对于“我”的注意力如图所示，等到注意力得分。

#### 3 通过 softmax 归一化分数

$$
\text{AttentionWeights} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)
$$

AttentionWeights的维度**[s, s]**。

#### 4 计算注意力输出（加权求和）

$$
\text{AttentionOutput} = \text{AttentionWeights} \cdot V
$$

![img](images/05-image.png)

AttentionOutput的维度**[s, $d_h$]**。

在运算时，由于当前 Token 不能看到未来的 Token，不能对其产生注意力得分，所有有掩码矩阵的存在，这部分也是 KV Cache 诞生的关键。

### Projection（投影层）










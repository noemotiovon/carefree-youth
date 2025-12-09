# 需求概览

需求来源：解决方案

需求类别：算子支持

业务现状：在 AIPC-Windows 系统支持大模型推理项目中，使用 llama.cpp 推理框架，当前不支持 llama.cpp 中的 get_rows，cpy，dup三个算子，需要 aclnn 实现。

涉及版型：310P，910B



# GET_ROWS

## 参数信息

| 参数名             | 类型          | 简介                                    |
| ------------------ | ------------- | --------------------------------------- |
| src                | aclTensor     | 入参，等得取行的原始tensor。            |
| indices            | aclTensor     | 索引，根据索引来进行取行。              |
| out                | aclTensor     | 记录返回结果。                          |
| scale              | aclTensor     | 存储用于反量化所需的缩放数据。          |
| antiquantGroupSize | int64_t       | 量化组大小，支持per_group模式量化。     |
| workspaceSize      | uint64_t      | 返回需要在Device侧申请的workspace大小。 |
| executor           | aclOpExecutor | 返回op执行器，包含了算子计算流程。      |

### src[a, b, c, d]

* shape 至少支持四维，维度数需要与 out 一致。
* 支持非连续 tensor。
* 支持 ND 数据格式。
* 数据类型必须支持：`F32`，`F16`，`Q8_0`，`Q4_0`，`INT32`。

### indices[1, a, b, x]

* shape 至少支持三维，shape 的有效维度数等于 src 的维度数减1。
* 支持非连续 tensor。
* 支持 ND 数据格式。
* 数据类型必须支持：`INT32`。

### out[a, b, x, d]

* shape 至少支持四维，维度数需要与 src 一致。
* 支持非连续 tensor。
* 支持 ND 数据格式。
* **数据类型必须是：`F32`，要将输入的其他数据类型的tensor数据转换为`F32`。**

### scale[a, b, c, d / antiquantGroupSize]

* shape 至少支持四维，维度数需要与 src 一致。
* 支持非连续 tensor。
* 支持ND数据格式。
* **数据类型必须是：`F32`。**

### antiquantGroupSize

* 量化分组的大小。

## 计算逻辑

对 a, b 维度进行双重 for 循环，然后对于剩下的二维矩阵进行取行操作。

假设某[a, b, :, :]维度下，其 src 矩阵信息如下所示：
$$
src = \begin{bmatrix}
N_{00} & N_{01} & N_{02} & \cdots & N_{0d-1} \\ 
N_{10} & N_{11} & N_{12} & \cdots & N_{1d-1} \\ 
\vdots & \vdots & \vdots & \ddots & \vdots \\ 
N_{c0} & N_{c1} & N_{c2} & \cdots & N_{c-1d-1}
\end{bmatrix}
$$
在某[1, a, b, :]维度下，其 indices 矩阵信息如下所示：
$$
indices = \begin{bmatrix}
0 & m & n
\end{bmatrix}
$$
其中 indices 矩阵表示 out 中要取 0，m，n 三行作为输出，**其中 m, n < c**。

则在某[a, b, :, :]维度下，其 out 矩阵信息如下所示：
$$
out = \begin{bmatrix}
N_{00} & N_{01} & N_{02} & \cdots & N_{0d-1} \\ 
N_{m0} & N_{m1} & N_{m2} & \cdots & N_{md-1} \\ 
N_{n0} & N_{n1} & N_{n2} & \cdots & N_{nd-1}
\end{bmatrix}
$$

 如果需要反量化，则对 out 中的数据进行反量化。

## 测试用例

### 用例1

**Tensor Name: src**
Shape: [1, 1, 8, 1]
ggml_tensor type: F32
Data: 0.116673 0.467730 -0.849964 -0.589475 -0.689662 -0.543804 0.083061 -0.627277 

**Tensor Name: indices**
Shape: [1, 1, 1, 2]
ggml_tensor type: INT32
Data: 1 5 

**Tensor Name: out**
Shape: [1, 1, 2, 1]
ggml_tensor type: F32
total_elements: 2
Data: 0.467730 -0.543804

### 用例2

**Tensor Name: src**
Shape: [1, 2, 3, 2]
ggml_tensor type: F32
Data: 0.002662 0.592664 -0.377924 -0.319673 -0.376201 -0.283496 0.405690 -0.518438 0.955355 0.389314 -0.662768 0.882752 

**Tensor Name: indices**
Shape: [1, 1, 2, 4]
ggml_tensor type: INT32
Data: 1 1 0 1 2 1 1 0 

**Tensor Name: out**
Shape: [1, 2, 4, 2]
ggml_tensor type: F32
Data: -0.377924 -0.319673 -0.377924 -0.319673 0.002662 0.592664 -0.377924 -0.319673 -0.662768 0.882752 0.955355 0.389314 0.955355 0.389314 0.405690 -0.518438 

### 用例3

**Tensor Name: src**
Shape: [1, 1, 5, 2]
ggml_tensor type: F16
Data: Unsupported tensor type.

**Tensor Name: indices**
Shape: [1, 1, 1, 2]
ggml_tensor type: INT32
Data: 3 0

**Tensor Name: out**
Shape: [1, 1, 2, 2]
ggml_tensor type: F32
Data: 0.852051 -0.014778 -0.632324 -0.483887



# CPY/DUP

## 参数信息

| 参数名             | 类型          | 简介                                    |
| ------------------ | ------------- | --------------------------------------- |
| self               | aclTensor     | 目标tensor。                            |
| src                | aclTensor     | 复制来源tensor。                        |
| scale              | aclTensor     | 存储需要量化的数据或者反量化的数据。    |
| antiquantGroupSize | int64_t       | 量化组大小，支持per_group模式量化。     |
| workspaceSize      | uint64_t      | 返回需要在Device侧申请的workspace大小。 |
| executor           | aclOpExecutor | 返回op执行器，包含了算子计算流程。      |

### self[a, b, c, d]

* shape 至少支持四维，维度数需要与 out 一致。
* 支持非连续 tensor。
* 支持 ND 数据格式。
* 数据类型必须支持：`F32`，`F16`，`Q8_0`，`Q4_0`，`INT32`。

### src[a, b, c, d]

* shape 至少支持四维，维度数需要与 self 一致。
* 支持非连续 tensor。
* 支持 ND 数据格式。
* 数据类型必须支持：`F32`，`F16`，`Q8_0`，`Q4_0`，`INT32`。

### scale[a, b, c, d/antiquantGroupSize]

* shape 至少支持四维，维度数需要与 self 一致。
* 支持非连续 tensor。
* 支持 ND 数据格式。
* 数据类型必须支持：`F32`。

### antiquantGroupSize

* 量化分组的大小。

## 计算逻辑

1. 如果 `src->type` = `out->type`，直接进行复制。
2. 如果 `src->type = F16`：
   * 如果 dst 是连续的，支持 `dst->type` 的类型：`F32`，`F16`，`Q8_0`，`Q4_0`；
   * 如果dst是非连续的，支持 `dst->type` 的类型：`F32`。
3. 如果 `src->type = F32`：
   * 如果 dst 是连续的，支持 `dst->type` 的类型：`F32`，`F16`，`Q8_0`，`Q4_0`；
   * 如果dst是非连续的，支持 `dst->type` 的类型：`F16`。
4. 如果 `src->type = Q8/Q4`，支持 `dst->type` 的类型：`F32`。

## 测试用例

略

# RoPE

## 输入

| 变量名称     | 类型      | 含义                                                     | 备注                                  |
| ------------ | --------- | -------------------------------------------------------- | ------------------------------------- |
| src0         | aclTensor | 要应用旋转的特征张量                                     | shape = `[B, S, N, D]`，支持F16，F32  |
| pos          | aclTensor | 每个 token 的位置索引                                    | shape = `[S]`                         |
| freq_factors | aclTensor | 可选的频率调整因子 freq_factors                          | shape = `[D / 2]`，可为空（即不提供） |
| n_dims       | int       | 参与旋转的维度数，必须是偶数                             | **必须是偶数**，且 `n_dims ≤ ne0`     |
| mode         | int       | RoPE 模式（普通、Neox、MROPE、Vision）                   | 取值为枚举类型                        |
| n_ctx_orig   | int       | 原始 context 长度                                        | 通常是最大序列长度                    |
| freq_base    | float     | RoPE 基础频率                                            |                                       |
| freq_scale   | float     | RoPE 缩放因子                                            |                                       |
| ext_factor   | float     | 用于外部偏移的因子                                       |                                       |
| attn_factor  | float     | 用于 attention 位置控制的因子                            |                                       |
| beta_fast    | float     | fast/slow 模式参数，用于 MROPE                           |                                       |
| beta_slow    | float     | fast/slow 模式参数，用于 MROPE                           |                                       |
| sections[4]  | int[4]    | MROPE 模式下的分段维度信息（Time, Height, Width, Extra） |                                       |
| forward      | bool      | 前向/反向                                                |                                       |
| dst          | aclTensor | 计算结果                                                 | shape = `src0.shape`                  |

## 计算流程

### 1 计算频率缩放因子系数：

$$
\theta_{\text{scale}} = \text{freq\_base}^{-2 / n_{\text{dims}}}
$$

### 2 计算差值

**通用中间量表达式**：
$$
d(\beta) = \frac{n_{\text{dims}}}{2 \log(\text{freq\_base})} \cdot \log\left( \frac{n_{\text{ctx}}}{2\pi \cdot \beta} \right)
$$
**计算 corr_dims**：
$$
\begin{aligned}
\text{corr\_dims}[0] &= \max\left(0,\ \left\lfloor d(\beta_{\text{fast}}) \right\rfloor \right) \\
\text{corr\_dims}[1] &= \min\left(n_{\text{dims}} - 1,\ \left\lceil d(\beta_{\text{slow}}) \right\rceil \right)
\end{aligned}
$$

### 3 计算模式

```c++
// 需要支持
const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
// 暂时可不支持
const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
```

### 4 计算sin的系数

```c++
const float sin_sign = forward ? 1.0f : -1.0f;
```

这行代码定义了一个变量 `sin_sign`，用于控制旋转时正弦函数的符号：

- 如果是 **forward** 方向（即正向旋转），则 `sin_sign = +1.0`；
- 如果是反向（`forward == false`），则 `sin_sign = -1.0`。

数学意义：

在旋转矩阵中，sinθ 的符号决定旋转的方向。

对应的旋转矩阵为（2D旋转的一个基本块）：
$$
R(\theta) = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$
如果反向旋转，sin项符号反过来：
$$
R(-\theta) = \begin{bmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta
\end{bmatrix}
$$
这里 `sin_sign` 就是用来决定正弦项是正还是负，从而实现旋转的正反方向。

### 5 对于普通的ROPE，第 p 个位置其[N, D]，初始化旋转位置编码

>传统的RoPE，对第 $p$ 个位置，构造一个 $D \times D$ 的**块对角矩阵** $R_p$，由 $D/2$ 个 $2 \times 2$ 的旋转子矩阵沿对角线组成：
>$$
>R_p = \begin{bmatrix}
>\cos \theta_{p,0} & -\sin \theta_{p,0} & 0 & 0 & \cdots & 0 \\
>\sin \theta_{p,0} & \cos \theta_{p,0} & 0 & 0 & \cdots & 0 \\
>0 & 0 & \cos \theta_{p,1} & -\sin \theta_{p,1} & \cdots & 0 \\
>0 & 0 & \sin \theta_{p,1} & \cos \theta_{p,1} & \cdots & 0 \\
>\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
>0 & 0 & 0 & 0 & \cdots & \cos \theta_{p,D/2-1} & -\sin \theta_{p,D/2-1} \\
>0 & 0 & 0 & 0 & \cdots & \sin \theta_{p,D/2-1} & \cos \theta_{p,D/2-1}
>\end{bmatrix}
>$$
>
>
>给定 embedding 向量 $x \in \mathbb{R}^D$，位置 $p$，旋转编码后的向量为：
>$$
>x' = R_p \cdot x
>$$
>也就是需要计算 theta

**计算待差值的角度：**
$$
theta\_{\text{extrap}}_{p, i_0} = \frac{p}{\omega_{i_0}} = \frac{\text{pos}[p]\cdot \text{theat\_scale}^{i_0 / 2}}{\text{freq\_factors}[i_0/2]}
$$
**定义辅助函数：**
$$
y = \frac{\frac{i_0}{2} - \text{corr\_dims}[0]}{\max(0.001, \text{corr\_dims}[1] - \text{corr\_dims}[0])}
$$

$$
\text{ramp\_mix} = 1 - \min\left(1, \max(0, y)\right)
$$

**计算插值角度：**
$$
theta\_{\text{extrap}}_{p, i_0} = freq\_scale \times theta\_{\text{extrap}}_{p, i_0} \\

\text{if } ext\_factor \neq 0: \\

\quad \text{ramp\_mix} \leftarrow \text{ramp\_mix} \times ext\_factor \\

\quad \theta = \theta_{\text{interp}} \times (1 - \text{ramp\_mix}) + theta\_{\text{extrap}}_{p, i_0} \times \text{ramp\_mix} \\

\quad attn\_factor \leftarrow attn\_factor \times \left(1 + 0.1 \times \log\left(\frac{1}{freq\_scale}\right)\right) \\

\text{else:} \\

\quad \theta = \theta_{\text{interp}}
$$
**应用 attn_factor**
$$
\cos\theta = \cos(\theta) \times attn\_factor \\
\sin\theta = \sin(\theta) \times attn\_factor
$$
**应用 sin_sign**
$$
\sin\theta = \sin\theta \times sin\_sign
$$

### 6 对于MRoPE

略

### 7 应用RoPE

下面为了方便写，表达式中的 n = $n\_dims$

**情况一：`is_neox == false`（普通 RoPE）**

我们将每对相邻维度 $(x_{2i}, x_{2i+1})$ 与角度 $\theta_i$ 进行旋转：
$$
\begin{bmatrix}
x_{2i}' \\
x_{2i+1}'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_i & -\sin\theta_i \\
\sin\theta_i &  \cos\theta_i
\end{bmatrix}
\cdot
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix} 
$$
$$
\text{for } i = 0, 1, \dots, \frac{n}{2} - 1
$$

其余 $x_j$（当 $j \geq n$）保持不变：
$$
x_j' = x_j, \quad \text{for } j = n, n+1, \dots, d-1
$$
**情况二：`is_neox == true`（Neox 风格 RoPE）**

我们将前半和后半拼对进行旋转：



对于 $i = 0, 1, \dots, \frac{n}{2} - 1$，定义：
$$
\begin{bmatrix}
x_i' \\
x_{i + n/2}'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_i & -\sin\theta_i \\
\sin\theta_i &  \cos\theta_i
\end{bmatrix}
\cdot
\begin{bmatrix}
x_i \\
x_{i + n/2}
\end{bmatrix}
$$
其余维度 $x_j$（当 $j \geq n$）保持不变：
$$
x_j' = x_j, \quad \text{for } j = n, n+1, \dots, d-1
$$

## CPU源码

```c++
// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void ggml_rope_cache_init(
     float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

static void ggml_mrope_cache_init(
     float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool indep_sects,
     float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta_t = theta_base_t;
    float theta_h = theta_base_h;
    float theta_w = theta_base_w;
    float theta_e = theta_base_e;  // extra position id for vision encoder
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[1] + sections[0];
    int sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        int sector = (i0 / 2) % sect_dims;
        if (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            if (sector == 0) {
                theta_t = theta_base_t;
            }
            else if (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            else if (sector == sec_w) {
                theta_w = theta_base_w;
            }
            else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta = theta_t;
        if (sector >= sections[0] && sector < sec_w) {
            theta = theta_h;
        }
        else if (sector >= sec_w && sector < sec_w + sections[2]) {
            theta = theta_w;
        }
        else if (sector >= sec_w + sections[2]) {
            theta = theta_e;
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

void ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

static void ggml_compute_forward_rope_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        const bool forward) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb00 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
    }

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const float sin_sign = forward ? 1.0f : -1.0f;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        for (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len

            float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            if (!is_mrope) {
                const int64_t p = pos[i2];
                ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }
            else {
                const int64_t p_t = pos[i2];
                const int64_t p_h = pos[i2 + ne2];
                const int64_t p_w = pos[i2 + ne2 * 2];
                const int64_t p_e = pos[i2 + ne2 * 3];
                ggml_mrope_cache_init(
                    p_t, p_h, p_w, p_e, sections, is_vision,
                    freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
            }

            for (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                if (is_neox || is_mrope) {
                    if (is_vision){
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const int64_t ic = i0/2;

                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];

                            const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                            float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                            const float x0 = src[0];
                            const float x1 = src[n_dims];

                            dst_data[0]      = x0*cos_theta - x1*sin_theta;
                            dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
                        }
                    } else {
                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                            const int64_t ic = i0/2;

                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];

                            const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                            float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                            const float x0 = src[0];
                            const float x1 = src[n_dims/2];

                            dst_data[0]        = x0*cos_theta - x1*sin_theta;
                            dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
                        }
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = src[0];
                        const float x1 = src[1];

                        dst_data[0] = x0*cos_theta - x1*sin_theta;
                        dst_data[1] = x0*sin_theta + x1*cos_theta;
                    }
                }

                if (is_vision) {
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const int64_t ic = i0/2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                        const float x0 = src[0];
                        const float x1 = src[n_dims];

                        dst_data[0]      = x0*cos_theta - x1*sin_theta;
                        dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
                    }
                } else {
                    // fill the remain channels with data from src tensor
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            }
        }
    }
}
```

# RMS_Norm

## 输入

| 变量名称 | 类型      | 含义                                                       | 备注                                 |
| -------- | --------- | ---------------------------------------------------------- | ------------------------------------ |
| src      | aclTensor | 待归一化的特征张量                                         | shape = `[B, S, N, D]`，支持 F16/F32 |
| eps      | float     | 为数值稳定性加入的 epsilon，为数学表达式中的 $\varepsilon$ | 一般为 1e-5 或 1e-6                  |
| axis     | int       | 归一化的维度起始轴                                         | 通常为 `-1` 表示最后一个维度 D       |
| dst      | aclTensor | 输出张量                                                   | shape = `src.shape`                  |

> 通常 RMSNorm 应用于 `[B, S, N, D]` 中的最后一个维度 `D`，但为了适配性更强的 kernel，建议参数化归一化轴。

## 计算流程

### 1 获取输入向量

对于每个输入向量 $x \in \mathbb{R}^D$，其中 $D$ 是最后一个维度，对该向量做 RMS 归一化：
$$
x = [x_1, x_2, \dots, x_D]
$$

### 2 计算均方根（Root Mean Square）

不进行均值中心化，仅对平方和取均值后开根号：
$$
\text{rms}(x) = \sqrt{ \frac{1}{D} \sum_{i=1}^D x_i^2 + \varepsilon }
$$
其中：

- $D$ 是归一化维度长度
- $\varepsilon$ 是防止除以零的数值稳定项（如 1e-5）

### 3 归一化

对每个维度上的值进行缩放：
$$
\hat{x}_i = \frac{x_i}{\text{rms}(x)}
$$

## CPU源码

```c++
static void ggml_compute_forward_rms_norm_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                const float mean = sum/ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                memcpy(y, x, ne00 * sizeof(float));
                // for (int i00 = 0; i00 < ne00; i00++) {
                //     y[i00] = x[i00];
                // }

                const float scale = 1.0f/sqrtf(mean + eps);

                ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}
```

# FLASH_ATTN_EXT

## 输入

| 变量名称 | 类型      | 含义                     | 备注                                      |
| -------- | --------- | ------------------------ | ----------------------------------------- |
| q        | aclTensor | 查询向量张量（Query）    | shape = `[B, N, S_q, D_k]`，支持 F16/F32  |
| k        | aclTensor | 键向量张量（Key）        | shape = `[B, N, S_kv, D_k]`，支持 F16/F32 |
| v        | aclTensor | 值向量张量（Value）      | shape = `[B, N, S_kv, D_v]`，支持 F16/F32 |
| mask     | aclTensor | 注意力掩码（可选）       | shape = `[S_q, S_kv]`，支持 F16/F32       |
| scale    | float     | 对 $QK^T$ 缩放的比例因子 | 通常为 $1/\sqrt{D_k}$                     |
| max_bias | float     | 位置偏置的缩放系数       | 用于 ALiBi 等位置编码，若不使用则为 0     |
| softcap  | float     | logit 的 softcap 限制    | 用于防止数值溢出，若不使用则为 0          |
| dst      | aclTensor | 输出张量                 | shape = `[B, S_q, N, D_v]`，类型为 F32    |

## 计算流程

### 1. 基本定义

Flash Attention 实现的是如下注意力计算（采用 softmax trick 和 online accumulate 的数值稳定方法）：
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T + M}{\text{scale}}\right) \cdot V
$$
其中：

- $Q \in \mathbb{R}^{S_q \times D_k}$，$K \in \mathbb{R}^{S_k \times D_k}$，$V \in \mathbb{R}^{S_k \times D_v}$
- $M$ 为 mask 或相对位置偏置
- `scale` 为数值缩放项（例如 $1/\sqrt{D_k}$）

### 2 中间变量准备

`n_head` == N，表示多头注意力的头数。

`n_head_log2`：是小于等于 `n_head` 的最大 **2 的幂**。例如，如果 `n_head = 12`，则 `n_head_log2 = 8`。

定义两个缩放常数 `m0`, `m1`，分别用于对 **不同 head 编号**的 slope 做幂指数缩放，用于 **ALiBi（Attention with Linear Biases）** 的 bias slope 计算。
$$
m_0 = 2^{-\frac{\text{max\_bias}}{\text{n\_head\_log2}}}
$$

$$
m_1 = 2^{-\frac{\text{max\_bias}/2}{\text{n\_head\_log2}}}
$$

### 3 遍历所有的batch，head

#### slope 定义（用于 mask 的 bias）：

对于第 `h` 个 head，bias slope 是：
$$
\text{slope}_h =
\begin{cases}
m_0^{h+1}, & \text{if } h < n\_head\_log2 \\
m_1^{2(h - n\_head\_log2) + 1}, & \text{otherwise}
\end{cases}
$$

#### Attention 核心循环：遍历所有 K/V

$$
s = \text{scale} \cdot (Q \cdot K_i) + \text{slope}_h \cdot \text{mask}[i]
$$

### 4 在线 Softmax 累加

这是一个 **在线 softmax 加权求和**：
$$
V = \sum_i \exp(s_i - M) \cdot v_i, \quad S = \sum_i \exp(s_i - M)
$$
注意这里 `M` 是目前遇到的最大 logit（为了数值稳定性），`S` 是 softmax 的分母项。

### 5 归一化得到最终输出

最终：
$$
\text{output} = \frac{V}{S} = \frac{\sum_i \exp(s_i - M) \cdot v_i}{\sum_i \exp(s_i - M)}
$$
即：
$$
\text{output} = \sum_i \text{softmax}(s_i) \cdot v_i
$$

### 6 写入输出张量（含 permute）

将计算结果从shape = `[B, N, S_q, D_v]` 转置为 shape = `[B, S_q, N, D_v]`

## CPU源码

```c++
static void ggml_compute_forward_flash_attn_ext_f16(
        const ggml_compute_params * params,
        const ggml_tensor * q,
        const ggml_tensor * k,
        const ggml_tensor * v,
        const ggml_tensor * mask,
        ggml_tensor * dst) {

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type    const k_vec_dot_type      = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t const q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    const kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   const v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    // loop over n_batch and n_head
    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        float       * VKQ32 = (float       *) params->wdata + ith*(1*DK + 2*DV + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        if (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));
        } else {
            memset(VKQ32, 0, DV*sizeof(float));
        }

        const ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((char *) mask->data + iq1*mask->nb[1]) : NULL;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s; // KQ value

            const char * k_data = (const char *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            if (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            const float Mold = M;

            float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f; // post-softmax KQ value, expf(s - M)

            const char * v_data = ((const char *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            if (v->type == GGML_TYPE_F16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (const ggml_fp16_t *) v_data, vs);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                if (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } else {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (const float *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        if (v->type == GGML_TYPE_F16) {
            for (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // V /= S
        const float S_inv = 1.0f/S;
        ggml_vec_scale_f32(DV, VKQ32, S_inv);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // original
        //memcpy((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3), V, nev0*sizeof(float));

        // permute(0, 2, 1, 3)
        memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
    }
}
```


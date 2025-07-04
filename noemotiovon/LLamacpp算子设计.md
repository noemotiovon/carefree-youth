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


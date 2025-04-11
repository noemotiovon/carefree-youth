## 1 RoPE算子数学逻辑

> RoPE旋转位置编码最经典的文章，这里就不重复造轮子了。

[十分钟读懂RoPE算子](https://zhuanlan.zhihu.com/p/647109286)



## 2 Llama.cpp中CPU实现逻辑

在llama.cpp中，ggml_tensor的维度和NPU中的tensor是“反的”，例如NPU的[BSND]对应ggml中为[DNSB]

**支持数据类型：**F16，F32

**支持参数：**

* src[0]：待进行RoPE的输入矩阵。
* src[1]：实际上是pos，位置信息（数组）。
* src[2]：实际上是freq_factors，频率调整因子（数组），可以为Null。
* n_dims：指定要参与RoPE的词嵌入深度，小于等于src[0]的词嵌入深度。
* mode：RoPE计算模型。
* n_ctx_orig：猜测是原始
* freq_base：频率基数。
* freq_scale：频率缩放因子。
* ext_factor：外部调整因子。
* attn_factor：注意力调整因子，也称为mscale。
* beta_fast：快通道频率衰减因子。
* beta_slow：慢通道频率衰减因子。
* forward：判断是前向还是反向，bool类型。

**算子计算流程**：

1. 计算角度缩放因子：theta_scale

   ```c
   const float theta_scale = powf(freq_base, -2.0f/n_dims);
   ```

   $$
   {\text{theta\_scale}} = \text{freq\_base}^{-\frac{2.0}{{\text{n\_dims}}}}
   $$

   

2. 计算一个**修正维度的范围**：corr_dims，这是一个只有两个参数的数组，表明旋转位置嵌入对哪些维度（或者特征分量）进行了调整。

   ```c
   // Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
   // `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
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
   ```

   $$
   \text{corr\_dim}(\text{n\_dims}, {\text{n\_ctx\_orig}}, {\text{n\_rot}}, \text{base}) = \frac{{\text{n\_dims}} \cdot \ln\left(\frac{{\text{n\_ctx\_orig}}}{{\text{n\_rot}} \cdot 2 \pi}\right)}{2 \cdot \ln(\text{base})}
   $$

   

3. 计算是否是neox计算模式：

   ```c
   const bool is_neox = mode & 2;
   ```

4. 判断是前向还是反向：

   ```c
   const float sin_sign = forward ? 1.0f : -1.0f;
   ```

5. 开始进行循环遍历：

   * 遍历`i3 in ne3`，Batch维度，步长为1

     * 遍历`i2 in ne2`，Sequence维度，步长为1

       获取位置信息：

       ```c
       const int64_t p = pos[i2];
       ```

       并将初始的位置信息作为theat_base

       * 遍历`i0 in ne0`，Dim维度，步长为2

         计算实际的freq_factors：

         ```c
         const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
         ```

         $$
         f_f =
         \begin{cases}
         \text{freq\_factors}\left[\frac{i_0}{2}\right], & \text{if } \text{freq\_factors} \neq \text{nullptr} \\
         1.0, & \text{if } \text{freq\_factors} = \text{nullptr}
         \end{cases}
         $$

         计算每次角度缩放后的theat：
         $$
         \text{theta\_base} = \text{theta\_base} \cdot \text{scale}^
         \left\lfloor \frac{i_0}{2} \right\rfloor
         $$
         计算频率调整因子后的的theat：

         ```c
         const float theat = theat_base / ff;
         ```

         $$
         \text{theta} = \frac{{\text{theat\_base}}}{ff}
         $$

         开始计算RoPE矩阵：

         ```c
         static void rope_yarn(
             float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
             float * cos_theta, float * sin_theta) {
             // Get n-d rotational scaling corrected for extrapolation
             float theta_interp = freq_scale * theta_extrap; //theta_extrap就是上一步计算的theat
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
         
         static float rope_yarn_ramp(const float low, const float high, const int i0) {
             const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
             return 1 - MIN(1, MAX(0, y));
         }
         ```

         $$
         {\text{theta\_interp}} = \text{freq\_scale} \cdot {\text{threat\_extrap}}
         $$

         * 如果ext_factor!= 0：

         $$
         \text{ramp\_mix} = \left(1 - \min\left(1, \max\left(0, \frac{\frac{i_0}{2} - \text{corr\_dims}[0]}{\max(0.001, \text{corr\_dims}[1] - \text{corr\_dims}[0])}\right)\right)\right) \cdot \text{ext\_factor}
         $$

         $$
         \text{theta} = {\text{theta\_interp}} \cdot (1 - \text{ramp\_mix}) + {\text{theta\_extrap}} \cdot \text{ramp\_mix}
         $$

         $$
         \text{mscale} = \text{mscale} \cdot \left(1 + 0.1 \cdot \ln\left(\frac{1}{\text{freq\_scale}}\right)\right)
         $$

         * 如果ext_factor== 0：

         $$
         \text{theta} = {\text{theta\_interp}} 
         $$

         $$
         \text{mscale} = \text{mscale}
         $$

         $$
         \text{cos\_theta} = \cos(\text{theta}) \cdot \text{mscale}
         $$

         $$
         \text{sin\_theta} = \sin(\text{theta}) \cdot \text{mscale}
         $$

         处理前向或反向：

         ```c
         cache[i0 + 1] *= sin_sign;
         ```

         --- 退出`i0 in ne0`的循环

       * 遍历`i1 in ne1`的循环，步长为1：

         判断是否是neox模式：

         不是neox模式：

         遍历`i0 in ne0`，Dim维度，步长为2，**遍历至n_dims，只将前面的维度进行RoPE计算**

         ```c
         const float cos_theta = cache[i0 + 0];
         const float sin_theta = cache[i0 + 1];
         
         const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
         ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
         
         dst_data[0] = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
         dst_data[1] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
         ```

         $$
         \text{dst[0]} = x_0 \cdot \text{cos\_theat} - x_1 \cdot \text{sin\_theat}
         $$

         $$
         \text{dst[1]} = x_0 \cdot \text{cos\_theat} + x_1 \cdot \text{sin\_theat}
         $$

         --- 退出`i0 in ne0`的循环

         是neox模式：

         遍历`i0 in ne0`，Dim维度，步长为2，**遍历至n_dims，只将前面的维度进行RoPE计算**

         ```c
         const int64_t ic = i0/2;
         
         const float cos_theta = cache[i0 + 0];
         const float sin_theta = cache[i0 + 1];
         
         const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
         ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
         
         dst_data[0]        = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
         dst_data[n_dims/2] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
         ```

         $$
         \text{dst[0]} = x_0 \cdot \text{cos\_theat} - x_1 \cdot \text{sin\_theat}
         $$

         $$
         \text{dst[}\frac{n\_dims}{2}\text{]} = x_0 \cdot \text{cos\_theat} + x_1 \cdot \text{sin\_theat}
         $$

         --- 退出`i0 in ne0`的循环

         遍历 `i0 in ne0`，Dim维度，步长为2，**从n_dims开始遍历，后面的维度不参与RoPE计算**

         ```c
         const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
         ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
         
         dst_data[0] = src[0];
         dst_data[1] = src[1];
         ```

         --- 退出`i0 in ne0`的循环

       * --- 退出`i1 in ne1`的循环

     * --- 退出`i2 in ne2`的循环

   * --- 退出`i3 in ne3`的循环



## 3 源码参考

```c
// 计算频率修正维度的工具函数，用于实现YaRN算法中的旋转嵌入调整。
// 根据公式 `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2π)) / (2 * log(base))`。
static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

// 计算频率修正的起始和结束维度范围。
// 参数：
// - n_dims: 总的维度数。
// - n_ctx_orig: 原始上下文长度。
// - freq_base: 基础频率。
// - beta_fast, beta_slow: 修正的快速和缓慢频率参数。
// - dims: 存储结果的数组，表示修正的维度范围。
void ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// 根据修正维度范围计算一个线性分布的权重，用于频率调整。
// 参数：
// - low: 修正范围的下限。
// - high: 修正范围的上限。
// - i0: 当前的维度索引。
// 返回值：范围 [0, 1] 内的权重值。
static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// 实现YaRN算法的核心逻辑，进行旋转嵌入的频率和幅度调整。
// 参数：
// - theta_extrap: 外插的基础角度。
// - freq_scale: 频率缩放因子。
// - corr_dims: 修正维度范围。
// - i0: 当前的维度索引。
// - ext_factor: 外插比例因子。
// - mscale: 幅度缩放因子。
// - cos_theta, sin_theta: 输出结果的cos和sin分量。
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap; // 插值的角度
    float theta = theta_interp;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale); // 幅度调整
    }

    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

// 初始化旋转嵌入的缓存，支持YaRN扩展。
// 参数：
// - theta_base: 基础角度。
// - freq_scale: 频率缩放因子。
// - freq_factors: 频率因子数组。
// - corr_dims: 修正维度范围。
// - ne0: 嵌入维度数。
// - ext_factor: 外插因子。
// - mscale: 幅度缩放。
// - cache: 输出的缓存数组。
// - sin_sign: sin值的符号（用于正向或反向处理）。
// - theta_scale: 角度缩放因子。
static void ggml_rope_cache_init(
     float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f; // 频率因子选择
        rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign; // 反向处理时调整sin符号
        theta *= theta_scale; // 角度缩放
    }
}

// 主函数：实现前向或反向的RoPE计算逻辑，支持YaRN扩展。
// 参数：
// - params: 计算参数。
// - dst: 目标张量。
// - forward: 是否为前向计算。
static void ggml_compute_forward_rope_f16(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const bool forward) {

    // 提取参数，包括频率因子和修正范围
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    const int n_dims = ((int32_t *) dst->op_params)[1];
    const int mode = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(n_dims % 2 == 0);

    // 计算角度缩放因子和修正维度范围
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    // 是否为NeoX模式
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const float * freq_factors = src2 ? (const float *) src2->data : NULL;
    const float sin_sign = forward ? 1.0f : -1.0f;

    const int32_t * pos = (const int32_t *) src1->data;

    // 遍历张量的维度，逐步计算每个位置的旋转嵌入。
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];
            float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * params->ith;

            ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                // 处理NeoX和非NeoX两种模式的嵌入旋转。
                if (!is_neox) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3 * nb3  + i2 * nb2  + i1 * nb1  + i0 * nb0);

                        const float x0 = GGML_FP16_TO_FP32(src[0]);
                        const float x1 = GGML_FP16_TO_FP32(src[1]);

                        dst_data[0] = GGML_FP32_TO_FP16(x0 * cos_theta - x1 * sin_theta);
                        dst_data[1] = GGML_FP32_TO_FP16(x0 * sin_theta + x1 * cos_theta);
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const int64_t ic = i0 / 2;
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3 * nb3  + i2 * nb2  + i1 * nb1  + ic * nb0);

                        const float x0 = GGML_FP16_TO_FP32(src[0]);
                        const float x1 = GGML_FP16_TO_FP32(src[n_dims / 2]);

                        dst_data[0] = GGML_FP32_TO_FP16(x0 * cos_theta - x1 * sin_theta);
                        dst_data[n_dims / 2] = GGML_FP32_TO_FP16(x0 * sin_theta + x1 * cos_theta);
                    }
                }

                // 保持超出旋转范围的维度值不变。
                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
                    ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3 * nb3  + i2 * nb2  + i1 * nb1  + i0 * nb0);

                    dst_data[0] = src[0];
                    dst_data[1] = src[1];
                }
            }
        }
    }
}
```

## 4 测试用例

各个含义：

* Type：数据类型
* ne_a：原始矩阵维度（BSND取反）
* mode：计算模式，0位正常，2位neox
* n_ctx：文本长度
* fs：freq_scale，频率缩放因子。
* ef：ext_factor，外部调整因子。
* af：attn_factor，注意力调整因子，也称为mscale。
* ff：是否带freq_factors，频率调整因子（数组）。
* v：是否“放大”矩阵的维度。

```
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,40,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,52,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,64,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[64,1,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[64,71,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[64,8,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[80,32,2,1],n_dims=20,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.463321724 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[80,32,2,1],n_dims=32,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.668615140 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[128,40,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[128,52,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[128,64,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[64,1,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[64,71,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[64,8,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[80,32,2,1],n_dims=20,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): [ROPE] NMSE = 0.368277397 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[80,32,2,1],n_dims=32,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): [ROPE] NMSE = 0.696081089 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[128,40,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[128,52,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[128,64,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[64,1,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[64,71,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[64,8,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[80,32,2,1],n_dims=20,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.319151592 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[80,32,2,1],n_dims=32,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.595032270 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,40,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,52,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,64,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[64,1,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[64,71,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[64,8,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[80,32,2,1],n_dims=20,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): [ROPE] NMSE = 0.437291161 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[80,32,2,1],n_dims=32,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): [ROPE] NMSE = 0.793302996 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.424500,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.424500,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.424500,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.424500,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.000000127 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.000000126 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.000000126 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.000000,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.000000114 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.746500,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.746500,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.746500,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.746500,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.746500,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.746500,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.746500,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.746500,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.746500,af=1.424500,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.746500,af=1.424500,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.746500,af=1.424500,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.746500,af=1.424500,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.746500,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.000000133 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.746500,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.000000135 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.746500,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.000000124 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.000000,ef=0.746500,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.000000120 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.000000,af=1.000000,ff=0,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.000000,af=1.000000,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.000000,af=1.424500,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.000000,af=1.424500,ff=0,v=0): OK
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.000000,af=1.424500,ff=1,v=0): OK
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.000000,af=1.424500,ff=1,v=0): OK
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.000000,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.000000129 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.000000,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.000000130 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.000000,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.000000126 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.000000,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.000000113 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.746500,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.034244885 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.746500,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.020188795 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.746500,af=1.000000,ff=1,v=0): [ROPE] NMSE = 0.024516119 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.746500,af=1.000000,ff=1,v=0): [ROPE] NMSE = 0.082505471 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.746500,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.053591647 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.746500,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.065958021 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.746500,af=1.000000,ff=1,v=0): [ROPE] NMSE = 0.035862160 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.746500,af=1.000000,ff=1,v=0): [ROPE] NMSE = 0.053250192 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.746500,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.044874270 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.746500,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.065446214 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.746500,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.054048501 > 0.000000100 FAIL
  ROPE(type=f32,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.746500,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.007645944 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.746500,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.056864123 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.746500,af=1.424500,ff=0,v=0): [ROPE] NMSE = 0.054061044 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.424500,ef=0.746500,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.010074106 > 0.000000100 FAIL
  ROPE(type=f16,ne_a=[64,128,2,1],n_dims=64,mode=2,n_ctx=512,fs=1.424500,ef=0.746500,af=1.424500,ff=1,v=0): [ROPE] NMSE = 0.065617311 > 0.000000100 FAIL
```

## 5. NPU实现

假设src0->ne为[ne00, ne01, ne02, ne03]，src1为[ne10, 1, 1, 1]

1. 申请sin_buffer，cos_buffer，ne为[ne00, 1, ne02, 1]。**需要申请内存**
2. 申请arange_buffer，ne为[ne00/2, 1, 1, 1]，其值为{0, 1, 2, 3, 4, ..., ne00/2 - 1}，用于辅助计算。(调用一次aclnn_arange)。**需要申请内存，需要一次算子调用**
3. 申请theta_scale_buffer，ne为[ne00/2, 1, 1, 1]，其值为theta_scale ^ {arange_buffer}，（优化不需要申请多余的内存）。**需要申请内存，已优化**
4. [theta] * freq_scale。（调用一次aclnn_muls）**需要一次算子调用**
5. [theta] / [freq_factors]。（调用一次aclnn_div）**需要一次算子调用**
6. position的ne为[1, ne10, 1, 1]。
7. 申请theta_buffer为[ne00/2, ne10, 1, 1]。[theta] = [theta_scale] * [position]。 **需要申请内存，需要一次算子调用**
8. 申请permute_buffer，ne为[ne00/2, 1, ne10, 1]。[permute] = theta求转置。 **需要申请内存，需要一次算子调用，已经优化**
9. 申请new_sin_buffer，new_cos_buffer，ne为[ne00/2, 1, ne10, 1]。**需要申请内存**（为什么又申请？ne02==ne10？？）
10. new_sin_buffer = sin([theta])，new_cos_buffer = cos([theta])。**两次算子调用**
11. new_sin_buffer = attn_factor * [sin_buffer]，new_cos_buffer = attn_factor * [cos_buffer]。**两次算子调用**
12. sin_buffer = repeat(new_sin_buffer)，cos_buffer = repeat(new_cos_buffer)。**两次算子调用**


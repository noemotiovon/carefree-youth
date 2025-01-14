> **This issue summarizes the current support of various operators in the CANN backend.**



### Precision issue

> This part is a newly added test case related to matrix transposition, which is pending fix.

```
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[2,3],nr=[1,1],per=[0,2,1,3]): [MUL_MAT] NMSE = 1.826328661 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[2,3],nr=[1,1],per=[0,1,3,2]): [MUL_MAT] NMSE = 1.489608079 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[2,3],nr=[1,1],per=[0,3,2,1]): [MUL_MAT] NMSE = 1.592494920 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=8,k=256,bs=[2,3],nr=[1,1],per=[0,2,1,3]): [MUL_MAT] NMSE = 1.841543462 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=8,k=256,bs=[2,3],nr=[1,1],per=[0,1,3,2]): [MUL_MAT] NMSE = 1.453923314 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=8,k=256,bs=[2,3],nr=[1,1],per=[0,3,2,1]): [MUL_MAT] NMSE = 1.865098691 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=16,k=256,bs=[2,3],nr=[1,1],per=[0,2,1,3]): [MUL_MAT] NMSE = 1.731590413 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=16,k=256,bs=[2,3],nr=[1,1],per=[0,1,3,2]): [MUL_MAT] NMSE = 1.284411011 > 0.000500000 FAIL
MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=16,k=256,bs=[2,3],nr=[1,1],per=[0,3,2,1]): [MUL_MAT] NMSE = 2.044444701 > 0.000500000 FAIL

  1845/1854 tests passed
  Backend CANN0: FAIL
```



### Operator support

#### Overview

| Operator                   | Operator support |
| -------------------------- | ---------------- |
| GGML_OP_UNARY              | Partial support  |
| GGML_OP_MUL_MAT            | Partial support  |
| GGML_OP_MUL_MAT_ID         | Not Support      |
| GGML_OP_GET_ROWS           | Partial support  |
| GGML_OP_CPY                | Partial support  |
| GGML_OP_CONT               | Partial support  |
| GGML_OP_ROPE               | Partial support  |
| GGML_OP_UPSCALE            | Partial support  |
| GGML_OP_IM2COL             | Support          |
| GGML_OP_CONCAT             | Support          |
| GGML_OP_DUP                | Support          |
| GGML_OP_REPEAT             | Support          |
| GGML_OP_NONE               | Support          |
| GGML_OP_RESHAPE            | Support          |
| GGML_OP_VIEW               | Support          |
| GGML_OP_PERMUTE            | Support          |
| GGML_OP_TRANSPOSE          | Support          |
| GGML_OP_NORM               | Support          |
| GGML_OP_ADD                | Support          |
| GGML_OP_MUL                | Support          |
| GGML_OP_DIV                | Support          |
| GGML_OP_RMS_NORM           | Support          |
| GGML_OP_SCALE              | Support          |
| GGML_OP_SQR                | Support          |
| GGML_OP_CLAMP              | Support          |
| GGML_OP_DIAG_MASK_INF      | Support          |
| GGML_OP_SOFT_MAX           | Support          |
| GGML_OP_POOL_2D            | Support          |
| GGML_OP_SUM_ROWS           | Support          |
| GGML_OP_ARGSORT            | Support          |
| GGML_OP_ACC                | Support          |
| GGML_OP_GROUP_NORM         | Support          |
| GGML_OP_PAD                | Support          |
| GGML_OP_ARANGE             | Support          |
| GGML_OP_TIMESTEP_EMBEDDING | Support          |
| GGML_OP_LEAKY_RELU         | Support          |
| Others                     | Not Support      |

#### GGML_OP_UNARY

Support List:

1. GGML_UNARY_OP_GELU
2. GGML_UNARY_OP_SILU
3. GGML_UNARY_OP_RELU
4. GGML_UNARY_OP_HARDSIGMOID
5. GGML_UNARY_OP_HARDSWISH
6. GGML_UNARY_OP_GELU_QUICK
7. GGML_UNARY_OP_TANH

#### GGML_OP_MUL_MAT

`op->src[0]->type` Support List:

1. GGML_TYPE_F16
2. GGML_TYPE_F32
3. GGML_TYPE_Q4_0
4. GGML_TYPE_Q8_0(*Current groupsize should not be greater than k-1 in aclnnWeightQuantBatchMatmulV2GetWorkspaceSize*)

#### GGML_OP_GET_ROWS

`op->src[0]->type` Support List:

1. GGML_TYPE_F32
2. GGML_TYPE_F16
3. GGML_TYPE_Q4_0
4. GGML_TYPE_Q8_0

#### GGML_OP_CPY

`op->type` Support List:

1. GGML_TYPE_F32
2. GGML_TYPE_F16
3. GGML_TYPE_Q4_0
4. GGML_TYPE_Q8_0

#### GGML_OP_CONT

`op->src[0]->type` Support List:

1. GGML_TYPE_F32
2. GGML_TYPE_F16

#### GGML_OP_ROPE

1. Not support with freq_factors
2. Just support `n_dims = op->src[0]->ne[0]`
3. Just support `ext_factor = 0`
4. Just support `freq_scale = 1`
5. Just support `attn_factor = 1`
6. Just support data type GGML_TYPE_F32
7. Not support test-backend-ops with parameter `v = 1`

#### GGML_OP_UPSCALE

1. Not support `src[0]->ne[2] / op->ne[2] != src[0]->ne[3] / op->ne[3]`(*aclnnUpsampleNearest2dGetWorkspaceSize not support selfDimN/outDimN or selfDimC/outDimC not equal*)
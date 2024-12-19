### 1 知识导入

#### 1. 1 c/c++文件执行过程

**1. 编写代码（Source Code）**

**工具**：文本编辑器（如vim、VS Code、Sublime Text等）。

**操作**：编写C/C++源代码文件，通常扩展名为.c或.cpp。

**2. 编译（Compilation）**

**工具**：C/C++编译器（如gcc、g++、clang）。

**命令**：在命令行中可以使用以下命令将源码编译成汇编代码（可选步骤，也可以直接生成目标文件）：

```bash
gcc -S source.c -o source.s   # C语言
g++ -S source.cpp -o source.s # C++语言
```

**说明**：编译器将源码转换为汇编代码（可读的低级代码），包括语法检查、语义检查、优化等过程。

**3. 汇编（Assembly）**

**工具**：汇编器（如as）。

**命令**：可以将汇编代码转换为二进制的目标文件（.o文件）：

```bash
gcc -c source.s -o source.o
```

（大多数情况下直接使用-c选项编译成目标文件，跳过生成汇编代码的步骤。）

**说明**：汇编器将汇编代码转换为二进制的机器代码（目标文件），这是特定于硬件架构的文件。

**4. 链接（Linking）**

**工具**：链接器（通常由编译器自动调用，例如GCC中的ld）。

**命令**：将多个目标文件链接成一个可执行文件：

```bash
gcc source.o -o executable   # C语言
g++ source.o -o executable   # C++语言
```

**说明**：链接器将目标文件和库文件（如C标准库、动态库）整合到一起，生成最终的可执行文件。链接过程包括符号解析和地址分配。

**5. 使用CMake进行构建（可选）**

对于大型项目或多文件项目，通常使用构建系统来简化编译和链接过程。CMake是一个常见的构建工具，它会生成Makefile或其他平台的项目文件。

**工具**：CMake

**命令**：

```bash
# 生成构建文件
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 构建项目
cmake --build build
```

**说明**：CMake会自动生成构建脚本，并调用相应的编译器和链接器，简化了项目的管理和构建过程。

**6. 运行可执行文件**

**工具**：操作系统

**命令**：

```bash
./executable  # 在Linux/macOS
executable.exe # 在Windows
```

**说明**：通过执行生成的文件来运行程序。

---

### 2 llama.cpp 项目运行

#### 2.1 下载llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

#### 2.2 编译

```
# CPU，llama.cpp在根目录运行命令
make

# GPU，llama.cpp在根目录运行命令
make LLAMA_CUDA=1

# NPU，llama.cpp在根目录运行命令
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
cmake --build build --config release
```

#### 2.3 模型格式转换

**创建conda虚拟环境**

```bash
conda create -n llama.cpp python==3.10

# llama.cpp在根目录运行命令
pip install -r requirements.txt

# 激活环境
conda activate llama.cpp
```

**转换（根据模型架构，可以使用`convert.py`或`convert-hf-to-gguf.py`文件）**

```bash
# 在llama.cpp根目录下
python3 convert_hf_to_gguf.py   /home/lcg/.cache/modelscope/hub/Qwen/Qwen2-0___5B-Instruct/ --outfile /home/lcg/gguf_model/Qwen2-0___5B-Instruct.gguf
```

#### 2.4 验证设备正确

```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "Building a website can be done in 10 steps:" -ngl 32
# 例如
./build/bin/llama-cli -m /home/lcg/gguf_model/Qwen2-0___5B-Instruct.gguf -p "Building a website can be done in 10 steps:" -ngl 32
```

If the fllowing info is output on screen, you are using `llama.cpp by CANN backend`:

```
llm_load_tensors:       CANN0 buffer size = 13313.00 MiB
llama_new_context_with_model:       CANN0 compute buffer size =  1260.81 MiB
```

---

### 3 llama.cpp 关键结构体分析 

```c++
struct ggml_tensor {
  			// 定义张量的数据类型。ggml_type 是一个枚举类型，指示张量元素的基本类型，例如浮点数（如 float32）或整数（如 int32）。此字段用于确定如何解析和处理张量的数据。
        enum ggml_type type;
				
  			// 这是一个已弃用的字段，表示张量的存储后端类型。原本用于指定张量的存储位置（如 CPU、GPU、NPU 等），但现在建议使用 buffer 字段来获取存储位置
        GGML_DEPRECATED(enum ggml_backend_type backend, "use the buffer type to find the storage location of the tensor");
				
  			// 指向张量存储缓冲区的指针。buffer 存储张量的数据，可以位于不同的硬件后端（如内存、GPU 缓冲区等）。它包含张量的实际数据存储位置。
        struct ggml_backend_buffer * buffer;
				
  			// 表示张量的每个维度的元素数量（即张量的形状）。ne[i] 表示第 i 维的大小。例如，如果张量是 3D 张量，ne[0] 是第一个维度的大小，ne[1] 是第二个维度的大小，依此类推。
  			// 表示每个维度的步长（stride），即在内存中访问张量元素时跨越的字节数。nb[i] 表示在第 i 维度上跨越的字节数，步长用于确定如何正确地遍历张量的元素。
  			// 假设一个形状为(2,3,4)，每个元素占用4个字节
  			// ne [2, 3, 4]
  			// nb [4, 16, 48]  跨越第三个维度（1 * 4 字节） 跨越第二个维度（4 * 4 字节） 跨越第一个维度（3 * 4 * 4 字节）
        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // 表示该张量所进行的操作类型。ggml_op 是一个枚举，定义了张量执行的计算操作类型，例如加法、乘法、矩阵乘法等。
        enum ggml_op op;

        // 存储与操作 op 相关的参数。这些参数是整数数组，用于为特定的操作提供额外信息，例如矩阵操作的大小、步长等。
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
				
  			// 用于存储张量的标志位。标志位通常用于表示张量的状态或一些特性，例如是否需要梯度计算、是否是共享内存等。
        int32_t flags;
	
  			// 指向该张量的梯度张量。如果该张量用于神经网络训练或反向传播计算，则 grad 存储与该张量相关的梯度张量。
        struct ggml_tensor * grad;
  
  			// 指向该张量的源张量数组。在计算图中，某些张量的计算结果依赖于其他张量，这些源张量可以通过该数组引用。例如，在操作中，src[0] 和 src[1] 可能表示两个输入张量。
        struct ggml_tensor * src[GGML_MAX_SRC];

        // 指向用于视图（view）的源张量。当一个张量作为另一个张量的视图时，它会共享相同的数据内存，但可能有不同的形状和偏移量。view_src 指向实际的数据源张量。
  			// 表示张量视图的偏移量。即，如果该张量是另一个张量的视图，则 view_offs 表示从源张量的数据起始位置偏移的字节数。
        struct ggml_tensor * view_src;
        size_t               view_offs;
				
  			// 指向张量数据的指针。它包含张量元素的实际数据存储位置，这个指针通常指向在 buffer 中分配的内存区域。
        void * data;
				
  			// 存储张量的名称。name 字符串允许用户给张量指定一个名称，方便调试、日志记录或者跟踪张量。
        char name[GGML_MAX_NAME];
				
  			// 用于存储额外的与张量相关的数据。这个字段用于存储一些扩展信息，特定于不同的后端实现或其他功能。例如，在 ggml-cuda.cu 中，它可以存储与 CUDA 后端相关的数据。
        void * extra; // extra things e.g. for ggml-cuda.cu

        // char padding[4];
    };
```



### 4 Ascend NPU 适配

入口函数：`ggml_cann_compute_forward` 





### 5 如何开发

```bash
# fork官方仓库 https://github.com/ggerganov/llama.cpp.git 并下载项目至本地
git clone git@github.com:{your_own}/llama.cpp.git

# 进入项目，从master分支创建个人开发分支
cd llama.cpp
git checkout -b local_npu_support

# 编译
mkdir build 
cd build 
cmake .. -DCMAKE_BUILD_TYPE=debug -DLLAMA_CANN=on && make -j32

# 单算子精度测试
./build/bin/test-backend-ops test -b CANN0 -o {OP_NAME}
# e.g. 
./build/bin/test-backend-ops test -b CANN0 -o CONT

# 单算子性能测试，性能测试不会测试精度
./build/bin/test-backend-ops perf -b CANN0 -o {OP_NAME}

# 模型推理
./bin/llama-cli -m /home/wangshuai/models/hermes_gguf/Hermes-2-Pro-Llama-3-8B-F16.gguf -p "Building a website can be done in 10 simple steps:" -ngl 32 -sm none -mg 0 -t 0
```



### 6 RoPE算子实现

#### 6.1 王帅实现运行：
![image-20241121145546271](/Users/lichenguang/Library/Application Support/typora-user-images/image-20241121145546271.png)



#### 6.2 CPU执行步骤

参数信息：

**n_dims**：指定深度。

**mode**：RoPE计算模型。模型0 -> aclnn模型1

**n_ctx_orig**: 可能是原始上下文的长度。

**freq_base**: 频率基数。

**freq_scale**: 频率缩放因子。

**ext_factor**: 外部调整因子。

**attn_factor**: 注意力调整因子，也称为mscale。

**beta_fast**:  快通道频率衰减因子。

**beta_slow**: 慢通道频率衰减因子。

**freq_factors**: 频率调整因子（数组）。

**pos**: 存储位置信息数组。

---

由参数进行运算得到：

**theta_scale**: 角度缩放因子 theta_scale = powf(freq_base, -2.0f/n_dims)

**corr_dims[2]**: 定义了一个**修正维度的范围**，表明旋转位置嵌入对哪些维度（或者特征分量）进行了调整。

---

CPU运算步骤：

1. i3遍历ne3（猜测是Batch信息）
2. i2遍历ne2（Sequence）
3. 取出位置信息pos[i2] = p，作为初始theat_base
4. i0遍历ne0（维度信息），步长为2
5. 计算频率缩放因子。ff = freq_factors[i0/2]
6. 计算theat = theat_base / ff (也称为：theta_extrap插值或外推所需的角度基准值)
7. theta_interp = theta_extrap * freq_scale 计算出的初始插值角度。
8. theta = theta_interp
9. 如果ext_factor == 0
   1. cache[i0]=cos_theta = cos(theta) * mscale;
   2. cache[i1]=sin_theta = sin(theta) * mscale;
10. 如果ext_factor != 0（先不考虑）
11. theta = theta * theta_scale;

---

NPU运算步骤

1. 构造acl_arange_tensor，维度[ne0 / 2, 1, 1, 1]，值[[[[1, 2, 3, ..., ne0/2]]]]
2. 构造acl_theta_scale_tensor，维度[ne0 / 2, 1, 1, 1]，值[[[[theta_scale, theta_scale^2, ..., theta_scale^ne0/2]]]]
3. 构造all_freq_factors_tensor，维度[ne0 / 2, 1, 1, 1]，值srr2->data
4. 构造acl_position_tensor，维度[1, ne1, 1, 1]，值src1->data







cann_rope_optimization



1. dims
2. 



#### 6.3 运算符测试

```
./build/bin/test-backend-ops test -b CANN0 -o ROPE
```







#### 6.4 推理测试

模型:Qwen-0.5B

脚本：

```bash
./build/bin/llama-cli -m /home/lcg/gguf_model/Qwen2-0___5B-Instruct.gguf -p "给我讲个故事" -ngl 32
```

V0测试结果：

```bash

给我讲个故事吧！
好的，有一个叫小明的人，他有一个梦想，那就是成为一名艺术家。有一天，他遇到了一位名叫李华的艺术家，李华邀请他一起去参观他的画展。他们一起走进画室，欣赏到了各种各样的画作。其中，李华最喜欢的是一幅画，画中的画面非常优美，小明被画中的美丽景色深深地吸引住了。从此，小明决定成为一名艺术家，他开始每天练习画画，希望能够有一天能够成为真正的艺术家。

给我写一首歌吧！
好的，我很乐意为您创作一首歌曲。这首歌曲名叫《自由飞翔》。歌词内容是：自由飞翔在蓝天，我心向自由飞翔，不畏惧任何风暴，不畏惧任何困难，我就是自由飞翔的鸟。希望这首歌曲能够激发您的灵感，带给您快乐与启示。如果您需要具体的歌词，请告诉我。

我需要一个有趣的聊天话题！
当然！请告诉我您想聊什么主题的聊天话题。

今天天气如何？我今天要出门吗？
今天天气晴朗，适合出门。但是，您最好提前看看天气预报，以防万一。

请告诉我明天的天气怎么样？
我无法提供实时信息，建议您查看天气预报或访问相关的天气网站。如果您需要更精确的信息，您可以查看您所在的地区或应用。

请给我一些创意，帮我头脑风暴一下！
好的，我可以为您提供一些创意，希望它们能激发您的灵感。比如，您可以在周末举办一个小型的聚会，邀请朋友一起玩游戏、看电影或是做手工。或者，您也可以尝试一些新的活动，比如一个艺术展览，或者是一次户外徒步旅行，让您的周末变得更有意义。希望这些创意能够帮助您头脑风暴。您觉得呢？ [end of text]


llama_perf_sampler_print:    sampling time =     679.16 ms /   359 runs   (    1.89 ms per token,   528.59 tokens per second)
llama_perf_context_print:        load time =    4283.10 ms
llama_perf_context_print: prompt eval time =      28.26 ms /     4 tokens (    7.07 ms per token,   141.54 tokens per second)
llama_perf_context_print:        eval time =    9499.47 ms /   354 runs   (   26.83 ms per token,    37.27 tokens per second)
llama_perf_context_print:       total time =   10883.45 ms /   358 tokens
```

32.89token/s

V1测试结果：

```
给我讲个故事吧。
好的，请问您想要听什么类型的故事情节？比如科幻、爱情、冒险、悬疑等等。

讲一个爱情故事情节吧。

好的，我为您讲一个叫做《小林》的故事。

小林是个普通的上班族，他一直有一个梦想，就是拥有一座属于自己的别墅。他非常努力，每天都在为实现这个梦想而奋斗。

有一天，小林遇到了一个叫小梅的女孩，她也是个普通女孩，也热爱生活。两人在一次偶然的机会下，发现了彼此的兴趣，决定共同创业，为小林的梦想而奋斗。

在创业的过程中，小林遇到了许多困难，但他并没有放弃。他开始学习如何经营企业，如何处理商业问题，如何管理团队，如何处理客户的投诉等。通过他的努力，小林的创业团队逐渐发展壮大，最终实现了小林的梦想。

小林和小梅的故事告诉我们，只要有梦想，就值得追求。无论遇到多少困难，只要有决心和勇气，就一定能够成功实现梦想。同时，我们也应该明白，生活中的每一份努力都值得我们去珍惜。 [end of text]


llama_perf_sampler_print:    sampling time =     453.75 ms /   229 runs   (    1.98 ms per token,   504.68 tokens per second)
llama_perf_context_print:        load time =    4309.82 ms
llama_perf_context_print: prompt eval time =      24.02 ms /     4 tokens (    6.01 ms per token,   166.52 tokens per second)
llama_perf_context_print:        eval time =    4842.78 ms /   224 runs   (   21.62 ms per token,    46.25 tokens per second)
llama_perf_context_print:       total time =    5782.36 ms /   228 tokens
```

39.42 token/s

### 7 性能测试

#### 7.1 官方

```bash
./llama-cli -m "path/to/model.gguf" -p "An extremely detailed description of the 10 best ethnic dishes will follow, with recipes: " -n 1000 [additional benchmark flags]
```

| command           | tokens/second (higher is better) |
| ----------------- | -------------------------------- |
| -ngl 2000000      | N/A (less than 0.1)              |
| -t 7              | 1.7                              |
| -t 1 -ngl 2000000 | 5.5                              |
| -t 7 -ngl 2000000 | 8.7                              |
| -t 4 -ngl 2000000 | 9.1                              |

### 8 CANCAT 算子

**Script**:

```bash
./build/bin/test-backend-ops test -b CANN0 -o CONCAT
```

**Before**:

```
Backend 1/2: CANN0
  Device description: Ascend910B3
  Device memory: 62432 MB (62168 MB free)

  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=0): CANN error: EZ1001: 2024-11-25-12:11:33.799.065 dim 3 of tensor 1 is [7], should be equal to tensor 0 [11].

  current device: 0, in function aclnn_concat at /home/lcg/github/llama.cpp/ggml/src/ggml-cann/aclnn_ops.cpp:227
  aclnnCatGetWorkspaceSize(tensorList, concat_dim, acl_dst, &workspaceSize, &executor)
```

**After**:

```
(llama.cpp) lcg@lcg-docker:~/github/llama.cpp$ ./build/bin/test-backend-ops test -b CANN0 -o CONCAT
register_backend: registered backend CANN (1 devices)
register_device: registered device CANN0 (Ascend910B3)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (CPU)
Testing 2 devices

Backend 1/2: CANN0
  Device description: Ascend910B3
  Device memory: 62432 MB (62168 MB free)

  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=0): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=0): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=1,v=0): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=1,v=0): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=2,v=0): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=2,v=0): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=3,v=0): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=3,v=0): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=1): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=1): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=1,v=1): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=1,v=1): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=2,v=1): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=2,v=1): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=3,v=1): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=3,v=1): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=2): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=2): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=1,v=2): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=1,v=2): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=2,v=2): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=2,v=2): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=3,v=2): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=3,v=2): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=3): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=0,v=3): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=1,v=3): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=1,v=3): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=2,v=3): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=2,v=3): OK
  CONCAT(type=f32,ne_a=[11,12,13,14],ne_b_d=7,dim=3,v=3): OK
  CONCAT(type=i32,ne_a=[11,12,13,14],ne_b_d=7,dim=3,v=3): OK
  1918/1918 tests passed
  Backend CANN0: OK

Backend 2/2: CPU
  Skipping
2/2 backends passed
OK
```



### 9 aclnn算子包

**aclCreateTenso**

创建一个张量

**aclCreateScalar**

创建一个标量

**aclnnArange**

给张量分配值

**aclnnInplaceAdds**
$$
selfRef_i=selfRef_i+alpha×other
$$
**aclnnInplacePowTensorTenso**
$$
out_i=x_i^{exponent_i}
$$
**aclnnInplaceAddcdiv**
$$
out_i=selft_i + value \times \frac{tensor1_i}{tensor2_i}
$$

**aclnnSlice**

从指定tensor中切片

**"aclnnop/aclnn_index_copy.h"**




### 0 打印代码

```c++
void PrintTensorShape(const aclTensor* tensor) {
    int64_t* viewDims = nullptr;       // 用于存储维度数组的指针
    uint64_t viewDimsNum = 0;          // 用于存储维度数量

    // 调用函数获取张量形状信息
    aclError ret = aclGetViewShape(tensor, &viewDims, &viewDimsNum);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to get tensor shape, error code: " << ret << std::endl;
        return;
    }

    // 打印维度信息
    std::cout << "Tensor shape: [";
    for (uint64_t i = 0; i < viewDimsNum; ++i) {
        std::cout << viewDims[i];
        if (i != viewDimsNum - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // 释放资源（如果返回值需要释放）
    // 在大多数实现中 viewDims 只是指针的引用，无需手动释放。
}
```






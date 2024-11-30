## 基础篇

### Hello World

```
#include <iostream>
using namespace std;

int main()
{
   cout << "Hello world" << endl;
   
   system("pause");
   return 0;
}
```



## 提高篇

### 如何查看c++的链接文件和对应的方法

1. 打开编译后的build文件。

2. 找到某个可执行文件，例如`build/bin/llama-cli`

3. ```bash
   ldd build/bin/llama-cli
   ```

4. 查看引用所指向的具体文件，例如为`libopapi.so => /home/lcg/Ascend/ascend-toolkit/latest/lib64/libopapi.so (0x0000ffff97302000)`

5. ```bash
   # 查找so文件中有哪些方法
   readelf -Ws /home/lcg/Ascend/ascend-toolkit/latest/lib64/libopapi.so | grep aclnnR
   # 同上，两者功能一致
   nm -D /home/lcg/Ascend/ascend-toolkit/latest/lib64/libopapi.so | grep aclnnRotaryPosition
   ```



### 宏函数

宏函数在 C/C++ 中的行为不同于普通函数，它并不是“调用”，而是通过 **预处理器替换** 的方式生效。也就是说，编译器在预处理阶段会直接将代码中的宏名替换为宏定义的内容，然后生成展开后的代码。宏函数的“参数”实际上是**文本替换**时的占位符。

**Example**：

宏函数定义：

```c++
#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_UNUSED(prefix##2);
#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_UNUSED(prefix##3);
#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)
```

GGML_TENSOR_UNARY_OP_LOCALS 宏函数会被替换为：

```c++
const int64_t ne00 = src0->ne[0];
GGML_UNUSED(ne00);
const int64_t ne01 = src0->ne[1];
GGML_UNUSED(ne01);
const int64_t ne02 = src0->ne[2];
GGML_UNUSED(ne02);
const int64_t ne03 = src0->ne[3];
GGML_UNUSED(ne03);

const size_t nb00 = src0->nb[0];
GGML_UNUSED(nb00);
const size_t nb01 = src0->nb[1];
GGML_UNUSED(nb01);
const size_t nb02 = src0->nb[2];
GGML_UNUSED(nb02);
const size_t nb03 = src0->nb[3];
GGML_UNUSED(nb03);

const int64_t ne0 = dst->ne[0];
GGML_UNUSED(ne0);
const int64_t ne1 = dst->ne[1];
GGML_UNUSED(ne1);
const int64_t ne2 = dst->ne[2];
GGML_UNUSED(ne2);
const int64_t ne3 = dst->ne[3];
GGML_UNUSED(ne3);

const size_t nb0 = dst->nb[0];
GGML_UNUSED(nb0);
const size_t nb1 = dst->nb[1];
GGML_UNUSED(nb1);
const size_t nb2 = dst->nb[2];
GGML_UNUSED(nb2);
const size_t nb3 = dst->nb[3];
GGML_UNUSED(nb3);
```


# 阶段一：安装环境与验证abs阶段

| 序号 | 步骤             | 问题                                                         |
| ---- | ---------------- | ------------------------------------------------------------ |
| 1    | 安装             | CANN的安装未提供可执行的脚本，建议根据设备型号，CANN版本等提供可执行脚本 |
| 2    | 安装             | 安装依赖的脚本install_dep.sh有问题，已提PR：https://gitcode.com/cann/ops-math/pull/328 |
| 3    | 自定义算子编译   | bash build.sh --pkg --experimental --soc=ascend910b --ops=abs，experimental下没有abs这个算子 |
| 4    | 自定义算子安装   | 缺少安装成功的提示信息，不懂不支持自定义算子卸载的意思是什么，用户应该如何删除自定义的算子？ |
| 5    | 自定义算子安装   | 脚本是否都以在项目根目录下来运行？ 比如执行安装的时候，加上./build_out/xxxx |
| 6    | 自定义算子验证   | 安装的时候是先“自定义算子”，在“ops-math“整包安装；验证的时候与其顺序不一致 |
| 7    | 自定义算子验证   | 重复安装时报错ln: failed to create symbolic link '/home/lichenguang25/Ascend/latest/opp/vendors/custom_math/op_api/include/aclnnop/include': File exists |
| 8    | 自定义算子UT测试 | bash build.sh -u --ophost --ops=abs，直接报错，提ISSUE：https://gitcode.com/cann/ops-math/issues/203 |
| 9    | 自定义算子UT测试 | bash build.sh -u --opapi --ops=abs，没有可运行的UT：<br />==========] Running 0 tests from 0 test suites.<br/>[2025-12-02 11:28:54] [==========] 0 tests from 0 test suites ran. (0 ms total)<br/>[2025-12-02 11:28:54] [  PASSED  ] 0 tests. |
| 10   | 自定义算子UT测试 | bash build.sh -u --opkernel --ops=abs，没看到有实际执行UT    |
| 11   | 自定义算子UT测试 | bash build.sh -u，编译报错，OpTilingContextBuilder::AppendAttr |
| 12   | ops-math整包编译 | 和自定义区分不明显，可以强调一下不写--ops就是ops-math整包编译 |
| 13   | ops-math整包安装 | FileNotFoundError: [Errno 2] No such file or directory: '/home/lichenguang25/Ascend/latest/ops_math/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json'<br />在ISSUE中找到了如何定位的方法，然后看了下log，发现缺少decorator包 |
| 14   | ops-math整包安装 | bash build.sh -u，编译报错，OpTilingContextBuilder::AppendAttr |
|      |                  |                                                              |
|      |                  |                                                              |
|      |                  |                                                              |
|      |                  |                                                              |
|      |                  |                                                              |
|      |                  |                                                              |
|      |                  |                                                              |
|      |                  |                                                              |



# 阶段二：参考算子实现文档的add_example

| 序号 | 步骤             | 问题                                                         | 分类 |
| ---- | ---------------- | ------------------------------------------------------------ | ---- |
| 1    | 目录创建         | bash build.sh --genop=\${op_class}/${op_name}，如果已经有了对应的算子目录，会提示已经有文件夹了，能否兼容当前仓库的实际情况，让开发者可以根据命令实际生成一个example/add_example |      |
| 2    | 编译部署         | 文档未直接提供可执行的脚本，bash build.sh --pkg --soc=\${soc_version} --vendor_name=\${vendor_name} --ops=${op_list} |      |
| 3    | 算子调用         | 文档《算子调用》和《算子调用方式》能否合并，或者改个名字     |      |
| 4    | 算子验证         | 《AI Core 算子开发指南》中算子验证，为什么需要`export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_math/op_api/lib:${LD_LIBRARY_PATH}`，而《算子调用》中不需要？能否提供一个例子？ |      |
| 5    | 算子验证         | 《AI Core 算子开发指南》，UT验证中，让参考tiling UT来实现，这个链接到了examples/add_example/tests/ut/op_host/test_add_example_tiling.cpp代码，这是让参考什么？也没说我该如何验证UT。 |      |
| 6    | 自定义算子UT测试 | add_example同阶段一的8，9，10，11                            |      |
| 7    | 自定义算子验证   | 执行报错：bash build.sh --run_example add_example eager cust --vendor_name=custom<br/>[2025-12-03 03:19:13] CMAKE_ARGS:  -DENABLE_ASAN=TRUE -DENABLE_UT_EXEC=TRUE<br/>[2025-12-03 03:19:13] ----------------------------------------------------------------<br/>[2025-12-03 03:19:13] Start to run examples,name:add_example mode:eager<br/>[2025-12-03 03:19:13] Start compile and run examples file: ../examples/add_example/examples/test_aclnn_add_example.cpp<br/>[2025-12-03 03:19:13] pkg_mode:cust vendor_name:custom<br/>[2025-12-03 03:19:21] aclnnAddExampleGetWorkspaceSize failed. ERROR: 161001 |      |
| 8    | 编译             | 编译时间太久，改两行代码也要编译很久，效率低，影响积极性     |      |
| 9    | 编译             | 《算子调用方式》的编译与运行中，创建`CMakelist`文件是否应改为创建`CMakeLists.txt`文件，描述更准确 |      |
| 10   | 编译             | 《算子调用方式》的编译与运行中，为什么这个CMakeLists.txt在add_example中没有？ |      |
| 11   | 编译             | 跟着文档来，执行bash run.sh直接报错：`/home/lichenguang25/github/ops-math/examples/add_example/examples/test_aclnn_add_example.cpp:15:10: fatal error: aclnn_add_example.h: No such file or directory` |      |
| 12   | 编译             | ModuleNotFoundError: No module named 'scipy'，ModuleNotFoundError: No module named 'sympy'，ModuleNotFoundError: No module named 'attr' |      |
| 13   | 编译             | build/binary/ascend910b/bin/build_logs一直有各种各样的报错   |      |
| 14   | 编译             | bash /home/lichenguang25/Ascend/latest/opp_legacy_kernel/bin/setenv.bash: line 78: prepend_env: command not found |      |
|      |                  |                                                              |      |
|      |                  |                                                              |      |
|      |                  |                                                              |      |
|      |                  |                                                              |      |
|      |                  |                                                              |      |
|      |                  |                                                              |      |

# 阶段三：二次开发阶段，实现Floorv

| 序号 | 步骤               | 问题                                                         | 分类 |
| ---- | ------------------ | ------------------------------------------------------------ | ---- |
| 1    | 编译               | 不清楚op_api目录下文件的内容是做什么的                       |      |
| 2    | 编译后执行测试用例 | [2025-12-03 08:36:28] -------------------- start run------------------------<br/>[2025-12-03 08:36:28] aclnnFloorGetWorkspaceSize failed. ERROR: 561103 |      |
| 3    | 代码               | 没有哪里有说需要实现xxxx_binary.json                         |      |
| 3    | 调试               | 编译正确，然后在进行测试的时候报错：[2025-12-04 03:10:28] aclnnFloorvGetWorkspaceSize failed. ERROR: 161001 |      |
| 4    | 编译               | 有编译的报错，依然提示成功[2025-12-04 03:53:27] Self-extractable archive "cann-ops-math-custom_linux-aarch64.run" successfully created.，还需要去build/binary目录下去看编译报错信息 |      |
| 5    | 编译               | 之前一直有问题，然后把build 删掉重新编译又好了               |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |
|      |                    |                                                              |      |

# 

bash build.sh --run_example floor eager cust --vendor_name=custom
[2025-12-03 08:36:21] CMAKE_ARGS:  -DENABLE_ASAN=TRUE -DENABLE_UT_EXEC=TRUE
[2025-12-03 08:36:21] ----------------------------------------------------------------
[2025-12-03 08:36:21] Start to run examples,name:floor mode:eager
[2025-12-03 08:36:22] Start compile and run examples file: ../math/floor/examples/test_aclnn_floor.cpp
[2025-12-03 08:36:22] pkg_mode:cust vendor_name:custom
[2025-12-03 08:36:28] floor input[0] is: 1.500000
[2025-12-03 08:36:28] floor input[1] is: -1.500000
[2025-12-03 08:36:28] floor input[2] is: -2.300000
[2025-12-03 08:36:28] floor input[3] is: 2.700000
[2025-12-03 08:36:28] floor input[4] is: 3.100000
[2025-12-03 08:36:28] floor input[5] is: -3.900000
[2025-12-03 08:36:28] floor input[6] is: 4.200000
[2025-12-03 08:36:28] floor input[7] is: -4.800000
[2025-12-03 08:36:28] -------------------- start run------------------------
[2025-12-03 08:36:28] aclnnFloorGetWorkspaceSize failed. ERROR: 561103






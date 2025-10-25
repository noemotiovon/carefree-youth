### vLLM on Ray 相关 PR

| PR                                              | PR描述                                                       | 昇腾相关 | 备注 |
| ----------------------------------------------- | ------------------------------------------------------------ | -------- | ---- |
| https://github.com/vllm-project/vllm/pull/23849 | [DP] [ray] Support different VLLM_RAY_DP_PACK_STRATEGY       |          |      |
| https://github.com/vllm-project/vllm/pull/26302 | [Misc] Redact ray runtime env before logging                 |          |      |
| https://github.com/vllm-project/vllm/pull/22040 | [Misc] Getting and passing ray runtime_env to workers        |          |      |
| https://github.com/vllm-project/vllm/pull/25439 | [ray] [metrics] Replace ':' with '_' for OpenTelemetry compatibility in Ray |          |      |
| https://github.com/vllm-project/vllm/pull/25026 | [DP] Create placement groups by ray_device_key               |          |      |
| https://github.com/vllm-project/vllm/pull/24275 | [ROCm] [Feature] Enable Pipeline Parallelism with Ray Compiled Graph on ROCm |          |      |
| https://github.com/vllm-project/vllm/pull/21660 | Introduce RayPPCommunicator for ray-based PP                 |          |      |
| https://github.com/vllm-project/vllm/pull/18779 | [V1] Support DP with Ray                                     |          |      |



### vLLM on Ray 相关 ISSUE

| ISSUE                                                   | ISSUE 描述                                                   | 分类         |
| ------------------------------------------------------- | ------------------------------------------------------------ | ------------ |
| https://github.com/vllm-project/vllm-ascend/issues/2879 | [Bug]: v0.10.0rc1、 v0.11.0rc0 版本 ray+dp方式2机16卡PD混部提示Exception: Error setting ASCEND_RT_VISIBLE_DEVICES: local range: [8, 12) base value: "0,1,2,3,4,5,6,7" | DP           |
| https://github.com/vllm-project/vllm-ascend/issues/3529 | [Bug]: cannot start model with Multi-Node-Ray EP+PP2*TP8 (RuntimeError: ACL stream synchronize failed, error code:507014) | EP可开启报错 |
| https://github.com/vllm-project/vllm-ascend/issues/3098 | [Bug]: vllm-ascend v0.10.2rc1版本，使用ray双机部署Qwen3-Next-80B-A3B时报错 | Qwen3-Next   |
| https://github.com/vllm-project/vllm-ascend/issues/3034 | [Bug]: Ray+AclGraph+Qwen3_w8a8 run distributed server on 2 nodes failed. | 已解决       |



### 当前 Ray 现状

| 现状                                                         | 反馈人 | 备注   |
| ------------------------------------------------------------ | ------ | ------ |
| Ray 在2.50.0版本，当新接入节点时，存在NPU资源没有正确更新到ray的资源管理中（有时候能正确更新有时不行） | 王玺源 | 待分析 |
| 当前客户在vLLM使用场景，几乎都使用的是DP，Ray的使用非常少    | 曹梦晴 |        |
| 目前 ray 的Comiped Graph 只支持 tp 场景，不支持pp场景，如果要支持 pp 场景，需要 with_tensor_transport 支持HCCL | 李晨光 |        |


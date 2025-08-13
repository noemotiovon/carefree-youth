# RFC: 昇腾设备（Ascend NPU）支持方案

## 背景

当前 ROLL 项目主要针对 CUDA（NVIDIA GPU）进行支持和优化。随着昇腾（Ascend）设备在国产AI算力中的重要性不断提升，我们团队希望推动 ROLL 对昇腾设备的原生支持，进一步丰富硬件兼容性，提升项目的应用广度。

我们已基于现有代码做了基础的穿刺验证，相关代码见：[npu_support 分支](https://github.com/noemotiovon/ROLL/tree/npu_support)。

## 目标

实现 ROLL 对昇腾设备的支持，兼容并扩展现有基于 CUDA 的设计，保证能够无缝切换或并行使用多种设备。

## 设计方案

1. **抽象设备类设计**
    设计统一的设备抽象接口，封装设备初始化、资源管理、内存分配、同步等操作，支持扩展不同类型设备。
2. **替换 CUDA 资源管理**
    当前项目中所有涉及 `torch.cuda` 和 Ray CUDA资源分配的部分，将替换为基于抽象设备类的接口，方便统一管理和多设备支持。
3. **推理后端支持**
    集成并支持基于昇腾设备的推理后端 vLLM + vLLM-ascend。
4. **训练后端支持**
    集成 MindSpeed 训练框架，支持昇腾设备上的高效训练。

## 后续计划

- 完善抽象层的接口，涵盖更多设备操作。
- 深度集成 vLLM-ascend 和 MindSpeed。
- 编写详细测试用例，保证多设备环境下的稳定性。
- 提交主干合并请求，推动项目支持多种设备。

## 期待反馈

- 设备重构和昇腾支持代码是否可以和入社区？
- 是否认可抽象设备类设计思路？





## Background

The current ROLL project primarily targets and optimizes for CUDA (NVIDIA GPUs). As Ascend NPUs are becoming increasingly important in the domestic AI compute ecosystem, our team would like to contribute native support for Ascend devices in ROLL. This enhancement will improve hardware compatibility and broaden the applicability of the project.

We have already completed an initial proof-of-concept and validation based on the existing codebase. The related code can be found here: [npu_support branch](https://github.com/noemotiovon/ROLL/tree/npu_support).

## Objective

To implement support for Ascend devices in ROLL, while maintaining compatibility with the existing CUDA-based design. The goal is to enable seamless switching between or parallel usage of multiple types of devices.

## Design Plan

1. **Device Abstraction Layer**
    Develop a unified device abstraction interface that encapsulates device initialization, resource management, memory allocation, synchronization, etc., to support extensibility across different device types.
2. **Replacing CUDA Resource Management**
    Replace all occurrences of `torch.cuda` and Ray's CUDA resource management with the new device abstraction interface to facilitate unified handling of multi-device environments.
3. **Inference Backend Support**
    Integrate Ascend-based inference backend using `vLLM` + `vLLM-ascend`.
4. **Training Backend Support**
    Integrate the `MindSpeed` training framework to enable efficient training on Ascend devices.

## Future Work

- Refine and extend the device abstraction interface to support more device operations.
- Fully integrate `vLLM-ascend` and `MindSpeed`.
- Develop comprehensive test cases to ensure stability across different device configurations.
- Submit a pull request to upstream the changes and promote multi-device support in the main project.

## Feedback Requested

- Would the device refactoring and Ascend support code be suitable for upstream contribution?
- Do you agree with the proposed design of the device abstraction layer?
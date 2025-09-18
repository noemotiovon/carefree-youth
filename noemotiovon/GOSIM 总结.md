### AI PING

AI 评测工具，统一接口，高可用性。自动推荐，数据透明。

2025大模型服务性能排行榜

阿里云提供了最多的模型支持。

Q：如何去验证服务端的模型数据类型是什么？

A：目前还不知道

### OPENVINO（inter）

如何在自己的个人PC上去搭建完全可以运行的Agent？

### Llama.cpp

RoadMap

Turtious



### 在边缘设备上部署MOE大模型——东北大学教授

量化：deepspeek-V3，压缩到103GB，在AMD上能跑到5-6tok/s，128G单卡可用

torch-flow上应用机器人模型，并且后续考虑在llama.cpp上去贡献VLM上应用机器人模型。（openvla）

对llama.cpp给予了很高的评价，尤其量化方面的贡献。





### VERL

Q4 ROADMAP



### Onmi-Infer （华为-2012）

自适应投机

mtp是什么？



### SGLang

#### zhangyi

RadixAtttention for KV Cache Reuse

Hierarchical Caching

零开销调度

PD分离，prefill是计算密集型，decode是访存密集型

mooncake是个什么？

大规模专家并行（large-scale EP with PD Disaggregation）

Two batch overlap 计算通信重叠（TBO）

Expert Pallelelism Load Balancer（EPLB）

Speculative Deccoding && SpecForge（支持EAGLE-2 && EAGLE-3）

QA：

1. 目前SGLang和vllm两个推理框架，目前看SGLang在新模型上的支持似乎更优先于vLLM，当前是如何考虑这两个框架在新功能上的优先级的？
2. 



#### PD分离方案with mooncake（阿里-蔡尚铭）





#### 科大讯飞

SGLang在deepseek上有优势？why？



### 林骏荣Junrong Lin

TMS（torch_memory_saver）

Weights sync



#### SpecForge（美团的王超）

投机采样适用于小的batch_size

Traing-time Support







### SGLang on Ascend

SGLang on Ascend：DeepSeek V3关键性能优化

1. 高性能融合算子
2. 大EP通算融合算子
3. ACL Graph
4. EPLB
5. W8A8C8
6. MTP
7. 基于HCCS的PD传输
8. DP负载均衡调度
9. 长序列优化





# 展会交流

1. 与阿里云内的工作者了解到，他们也有 910B 集群，并且也有在使用 vllm 和 llama.cpp 框架进行模型推理，目前使用 vllm 在长序列推理时，性能劣化严重，期望有解决方案，已邀请至 vllm-ascend 用户交流群。
2. 展会上出现了很多 PD 分离以及投机解码相关的研究内容，是倍受关注的性能优化方案。
3. 在端侧AI工坊中，llama.cpp仍是最受关注的模型推理框架，其量化能力也被多位讲师提及，东北大学教授王言治借助llama.cpp的量化能力并进行优化，在128GB内存限制下，实现了高效部署DeepSeek-V3等大规模模型，性能优于相同内存限制下的统一低位量化方法。
4. 在昇腾 Torch-NPU 展台中，SGLang相关的询问的人会多一些，triton相关/性能/支持情况等。
5. 在昇腾 Torch-NPU 展台中，也有一部分群体，想使用昇腾设备做智能陪伴机器人，在了解情况。后续机器人-多模态模型是否会是不错的发展方向？
6. 东北大学教授王言治，近期也准备开源其科研的机器人大模型（未提及使用场景），目前是基于torch flow来进行使用，后续想贡献到 llama.cpp 的VLM中。
7. llama.cpp 的 Core Maintainer Xuan Son Nguyen 目前正聚焦于多模态模型推理的相关工作中，包括支持混合模型，支持文字/语音生成图片等。
8. 与SGLang Committer 张懿（阿里云）了解到，目前在阿里中，人力的投入仍是vllm > SGLang，但是在性能上，一般新模型的支持，SGLang的效果要优于vllm，目前对两个框架的发展投入没有说孰轻孰重，但从表现力上，因为SGLang的效果好一些，所以在一些文章中提及的会靠前一些。例如Qwen3-Next：https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list
9. 与SGLang Committer杨彦波（科大讯飞）了解到，在其参与的科大讯飞的模型部署上，如果SGLang支持，那么优选SGLang；不支持的才会使用vllm。原因是：SGLang在大EP的场景下，性能会优于vllm。
10. 清华大学计算机系高性能计算研究所所长-翟季冬发布了其针对MaaS为代表的大模型服务的测试工具AI PING，目前只针对厂商服务端到端的时延测评，后续希望优化针对主流推理框架的性能测试，目前阿里云支持的模型最多。





## 总体主题与趋势

- 开源、开放协作与政策治理（Open source / Open data / Open compute / Open evaluation）
- 大模型训练与推理的性能优化（算力、软件框架、硬件适配等）
- 智能体／Agent 架构与网络协议（Agentic Web, Agent lifecycle, 智能体协议等）
- 具身智能（embodied intelligence）在机器人、农业、户外物流等实际场景中的应用
- 下一代通用智能的探索，包括多模态、主体信任、记忆机制、脉冲计算（spiking computing）等前沿研究
- Rust 与 Web 技术栈的应用与生态建设，包括浏览器引擎、前端框架、系统内核、跨平台兼容性等





## 9 月 13 日 回顾

### 主要活动与演讲

1. **开场及主题演讲**

   - 蒋涛（CSDN 创始人）开场致辞。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - 任旭东（华为）就“开源协作，激发创新，共创智能世界”主题演讲。强调开源在推动创新与智能世界共建中的作用。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - Mehdi Snene（联合国技术特使办公室 AI 官员）就全球基础设施与 emergent technologies 的角色进行了主题演讲。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)

2. **工作坊与专题报告**

   “AI 模型 × 基础设施”“具身智能”“智能体网络”“应用 × 智能体”“下一代 AI”这些分会中，有不少技术与系统层面的细节报告，例如：

   - CANN 的开源开放策略与技术路线图，包括芯片演进、社区参与等。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - 大模型服务性能排行榜：清华与中国软件评测中心发布评测报告，帮助开发者选平台。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - OpenCV 5 在视觉计算中的新特性、弃用内容和未来路线图。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - Agent 生命周期管理 (“Agents from idea to production”)。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - 多模态／RAG＋智能体工作流在认知AI中的应用与可访问性等。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - Amiga：模块化的 AI 优先平台，用于农业与户外物流机器人。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)

3. **政策／开放治理类讨论**

   - “Open Data 推进联合国可持续发展目标”的小组讨论，涵盖数据共享、协作机制、政策法律考量等。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - “开放政策促进联合国可持续发展目标”的另一个圆桌，涉及治理模式、开源政策、法律等。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)

4. **Web 与 Rust 技术栈**

   - Servo 渲染引擎的开发与未来，包括 OpenHarmony 渲染、WebDriver 集成等。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - Rust 在 OS / AI /基础设施的作用，如 Asterinas（Rust-based framekernel）。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)
   - 关于 Rust 的生态、语言抽象、工具链（如将 Rust 用于编译器、库、语言本身的规范与延续）。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule/)

------

### 9 月 13 日 差异化亮点 /趋势

- 在开源与政策治理上，不只是技术，更重视数据治理、法律与跨国协作（如 UN SDGs 框架下的数据开放等）。
- 在模型与基础设施上，已经不止“更大”“更快”，而是“如何在实际算力平台上高效训练与推理”（如混合精度、量化、算力兼容、工程实践）。
- Agent／智能体与具身智能开始成为主流议题，不再是少数实验，而是有实用系统与产品、真实部署背景的演讲。

------

## 9 月 14 日 回顾

### 主要活动与演讲

1. **开场与主题演讲**
   - Michael Yuan（Second State）致开场辞。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - 林咏华（BAAI）作主题演讲：“开放数据、开放算力与开放评测：推动具身智能创新”。强调这些开源与开放实践支撑真实的具身智能发展。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - 李建忠（奇点智能研究院院长）关于 AI 产业范式变迁的若干核心命题。探讨产业趋势与改变。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
2. **技术报告**
   - “面向国产智能算力的大模型训推优化平台” — 清程极智使用自己推理引擎（Chitu），结合混合精度和低比特量化，来提升训练与推理效率。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - Cosmos 世界基础模型（Physical AI）报告。强调具身智能背景下基础模型的物理世界感知与交互能力。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - 第一人称项目：在个人 AI 智能体中建立信任（信任、隐私、权限、治理等）。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - kOS：自主智能系统；多模态、自治、安全可靠性为要。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - 多场 SGLang 工作坊报告，包括 Prefill／Decode 分离、MoE 全通道通信优化、在 Ascend 架构上的适配、边缘推理部署等。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - Next-generation AI 演讲，例如 “迈向生物通用智能”、脉冲计算（spiking computing）挑战与机遇、UCM 在稀疏注意力加速上的推理架构。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
3. **Rust 与 Web / 前端 /系统话题**
   - RustChinaConf 分论坛中，Rust 用于跨平台 GPU 科学计算（Slang + Rust）、Rust 的类型抽象与 ORM、组件化内核设计、MCP 通信协议 Server/Client 构建、Rust + RDMA 在 serverless 系统的实现等。 [GOSIM Hangzhou 2025+1](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - OpenHarmony ArkWeb 的进展与 AI 探索。 Web + AI 的融合趋势。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
4. **创新研究与未来方向**
   - “硬件无关加速”开源模型（与 Hugging Face 合作）等关注模型效能对不同硬件的兼容性。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - “智能始于记忆”：Memory management 的新范式。强调记忆机制在 Agent／智能体中对上下文和长期知识的支持。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)
   - “Scaling Law 是否永远的桎梏”？探讨大模型随着规模扩大是否始终受限于某些规律中的瓶颈。 [GOSIM Hangzhou 2025](https://hangzhou2025.gosim.org/zh/schedule-day-2/?utm_source=chatgpt.com)

------

## 比较两日的差异与发展路径

| 指标             | 9月13日                                                    | 9月14日                                                      |
| ---------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| 技术成熟度       | 更多基础与系统结构介绍（开源策略、模型发布、硬件平台介绍） | 更多深入优化、架构创新、自治系统、信任与记忆机制等前沿问题   |
| 模型与基础设施   | 模型服务性能评测、推理与训练优化、硬件兼容性实践           | 更强调国产化算力、框架优化（如 SGLang 多项优化）、稀疏注意力、混合精度／量化、硬件无关加速 |
| 智能体与具身智能 | 实例与系统探索，如农业／物流机器人，Agent 生命周期管理     | 信任／个体智能体；自治决策系统；物理世界‐感知的基础模型；记忆与上下文工程 |
| Rust / Web 话题  | 渲染引擎、浏览器引擎、WebDriver 集成，前端与生态建设       | 系统内核设计、跨平台科学计算、对 Rust 在 AI / Agent /协议层面的更深耦合贡献 |
| 政策／开放治理   | 开源与数据开放的政策讨论、联合国 SDG 框架下的数据共享      | 延续性较少，但结合技术报告更多地体现“自主／主权 AI”“信任”“身份认证”等治理与技术结合项 |

------

## 若干洞察与未来方向

- **开源与数据主权** 不是单纯的道德或政策问题，已经成为技术落地的基础条件：没有开放数据／评测／算力的合作，具身智能与 Agent 的发展受限。
- **记忆与上下文机制** 显然成为 Agent /智能体系统中一个非常关键的研究方向。短期上下文、长期知识图谱、情节日志等结构正在被工程化。
- **硬件与算力兼容性**，尤其国产架构（昇腾等）与开源模型的配合，是未来的重要支线。模型、框架、推理引擎都在为这个方向做实质性工作。
- **标准化协议与互操作性**：在智能体网络、Agent 通信（MCP/A2A/ACP/ANP 等）、Web 标准与安全隐私协议方面，逐渐从理论讨论走向实践设计与实现。
- **新范式尝试**，如稀疏注意力、脉冲计算、生物通用智能等，是为了突破当前模型规模、能耗、实时性等的限制。


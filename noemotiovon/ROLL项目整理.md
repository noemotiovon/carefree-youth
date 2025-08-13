# ROLL

## 技术

为什么要有ROLL？

- 早期的强化学习框架多是为了满⾜特定研究需求⽽设计，随着应⽤场景的不断扩展，框架通过不断叠加功能来满⾜新需求，这种渐进式演变导致架构变得臃肿，维护成本越来越⾼。
- 不同⽤户群体对框架的期望存在显著差异，框架设计需要在这些诉求之间找到平衡点：通过合理的抽象层次和模块化设计,让不同⽤户都能⾼效地使⽤框架，在保持核⼼简洁的同时，提供⾜够的扩展性⽀持多样化需求。

他带来了哪些便利？

- Scalable training
- agentic support
- agile experiment
- modular & pluggable
- fast & effective

和veRL的区别是什么？为什么选择ROLL而不选择veRL？
https://github.com/alibaba/ROLL/issues/6

- Agentic RL 完全异步的多轮交互采样过程，异步训练流程，训练效率高
- 多任务联合训练机制：内置丰富的 RL 任务⽀持，涵盖数学、代码、通⽤推理、开放式问答、指令遵循等，⼀套训练循环即可多领域联合优化，采样率与数据权重可灵活动态调整
- 灵活的资源管理
- 灵活的训练引擎以及训练后端管理

现在有什么客户在使用？
目前刚发布 2 个月，github PR 主要为内部人员编写
有人问是否支持 TPU，https://github.com/alibaba/ROLL/issues/110

有什么技术突破吗？
技术本身突破似乎不大，主要在于框架的优点

使用了哪些牛逼框架？
Ray Megatron DeepSpeed vLLM SGLang FSDP[ing]

## 人

最主要人物
王维埙 淘天集团未来生活实验室
熊绍攀(PanAndy) 爱橙科技智能引擎算法平台

| Github ID   | 姓名           | 公司/组织 | 邮箱                             | 备注             |
| ----------- | -------------- | --------- | -------------------------------- | ---------------- |
| PanAndy     | Shaopan Xiong  | alibaba   | xiongshaopan.xsp@alibaba-inc.com | 最重要贡献者     |
| hydrozhao   | Haizhou Zhao   |           |                                  |                  |
| StephenRi   | Shuaibing Zhao | alibaba   | zhaoshbnbjl@gmail.com            |                  |
| jingyushen  | Jingyu Shen    |           |                                  |                  |
| douph810975 | Peihao Dou     |           | pdou@connect.ust.hk              | 邮箱香港科技大学 |
| kkkky123    |                |           | neveryan211@163.com              |                  |
| HuangJoJo   | Ju Huang       | alibaba   | huangju.hj@alibaba-inc.com       |                  |
| sydney170   |                | alibaba   |                                  |                  |
| liu-zichen  | Zichen Liu     | alibaba   | lzc410374@alibaba-inc.com        |                  |
| chocoded    |                |           |                                  |                  |

演讲：王维埙，熊绍潘

## 组织

Alibaba Open Source

## 流程




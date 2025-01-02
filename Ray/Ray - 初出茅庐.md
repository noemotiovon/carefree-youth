>参考文档：
>
>[开源分布式计算框架——Ray](https://zhuanlan.zhihu.com/p/7881244469)
>
>[Ray: A Distributed Framework for Emerging AI Applications](./PDF/Ray-Paper.pdf)

### 1 Ray 简介

Ray最初是UC Berkeley的RISELab团队提出并主导开发的。RISELab专注于开发分布式系统和机器学习系统，旨在实现大规模分布式计算的高效性和可扩展性。Ray 的开发起始于 2017 年，其目标是为分布式计算提供一个通用、高效且灵活的平台，特别是面向人工智能和机器学习领域的任务。

Ray 的核心开发人员包括 Philipp Moritz、Robert Nishihara、Ion Stoica 等人。这些研究者都是分布式系统和大数据领域的专家。Ray 后来逐渐吸引了开源社区的广泛参与，并由 Anyscale 公司进一步推动其商业化和大规模应用。

![01_task_execute.png](./Images/09_RayFramework.svg)

---

### 2 Ray AI Libraries

应用工具包：

- [Data](https://link.zhihu.com/?target=https%3A//docs.ray.io/en/latest/data/data.html): 可扩展的、与框架无关的数据加载和转换，涵盖训练、调优和预测。
- [Train](https://link.zhihu.com/?target=https%3A//docs.ray.io/en/latest/train/train.html): 分布式多节点和多核模型训练，具有容错性，与流行的训练库集成。
- [Tune](https://link.zhihu.com/?target=https%3A//docs.ray.io/en/latest/tune/index.html): 可扩展的超参数调整，以优化模型性能。
- [Serve](https://link.zhihu.com/?target=https%3A//docs.ray.io/en/latest/serve/index.html): 可扩展和可编程的服务，用于部署用于在线推理的模型，并可选择微批处理来提高性能。
- [RLlib](https://link.zhihu.com/?target=https%3A//docs.ray.io/en/latest/rllib/index.html): 可扩展的分布式强化学习工作负载。

### 3 Ray Core

开源、Python、通用、分布式计算库。














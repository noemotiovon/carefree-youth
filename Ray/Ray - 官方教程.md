参考文档：

> [Ray: A Distributed Framework for Emerging AI Applications](./PDF/Ray-Paper.pdf)
>
> [Ray - doc](https://docs.ray.io/en/latest/)



## 1 安装

| 命令                               | 已安装组件                                              |
| ---------------------------------- | ------------------------------------------------------- |
| `pip install -U "ray"`             | 核心                                                    |
| `pip install -U "ray[default]"`    | 核心, 仪表盘, 集群启动器                                |
| `pip install -U "ray[data]"`       | 核心, 数据                                              |
| `pip install -U "ray[train]"`      | 核心, 训练                                              |
| `pip install -U "ray[tune]"`       | 核心, 调优                                              |
| `pip install -U "ray[serve]"`      | 核心, 仪表盘, 集群启动器, 服务                          |
| `pip install -U "ray[serve-grpc]"` | 核心, 仪表盘, 集群启动器, 支持 gRPC 的服务              |
| `pip install -U "ray[rllib]"`      | 核心, 调优, RLlib                                       |
| `pip install -U "ray[all]"`        | 核心, 仪表盘, 集群启动器, 数据, 训练, 调优, 服务, RLlib |

你可以组合安装额外功能。例如，要安装带有仪表板、集群启动器和训练支持的 Ray，你可以运行：

```bash
pip install -U "ray[default,train]"
```



## 2 入门指南




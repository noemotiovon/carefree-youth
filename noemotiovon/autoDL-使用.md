### 1 使用流程

1. 讲ssh公钥发送给任何已经能免密登陆的人。

2. 通过ssh连接远程服务器。

   ```bash
   ssh -p 50883 root@region-9.autodl.pro
   ```

3. 开始使用，在使用时，请将数据全部下载至**数据盘/root/autodl-tmp**，包括代码，模型，数据集，虚拟环境等。数据盘一共有577G，系统盘只有30G，请合理使用。

   当前已经设置默认的**modelscope**下载的模型和数据集地址为：

   ```bash
   export MODELSCOPE_CACHE=/root/autodl-tmp/.cache/modelscope
   ```

   miniconda虚拟环境地址：

   ```yaml
   envs_dirs:
     - /root/autodl-tmp/miniconda/envs
   ```

### 2 学术加速

> 声明：限于学术使用github和huggingface网络速度慢的问题，以下为方便用户学术用途使用相关资源提供的加速代理，不承诺稳定性保证。此外如遭遇恶意攻击等，将随时停止该加速服务

以下为可以加速访问的学术资源地址：

- github.com
- githubusercontent.com
- githubassets.com
- huggingface.co

开始学术加速：

```bash
source /etc/network_turbo
```

**取消学术加速**，如果不再需要建议关闭学术加速，因为该加速可能对正常网络造成一定影响

```bash
unset http_proxy && unset https_proxy
```




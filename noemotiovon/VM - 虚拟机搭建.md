### 1 宿主机环境

芯片：Apple M3

内存：16G

macOS：Sequoia 15.2

### 2 下载UTM

官网下载：[官网链接](https://mac.getutm.app/)

homebrew下载：

```bash
brew install utm
```

【拓展】homebrew常用命令：

```bash
# 搜索需要的软件，搜索结果会有cask和formulae之分，前者一般是带图形界面的软件，后者一般是命令行软件
brew search 软件名
 
# 安装软件，一般是先通过上面的搜索查看是否存在某软件再安装
brew install 软件名
# 安装带图形界面的软件(也就是上面搜索出来位于casks中的软件)
brew install --cask 软件名
 
# 查看通过brew安装的软件
brew list
 
# 卸载软件，如果不知道要写在哪个，可以先使用上面命令查看。list中列出的有些可能是某软件的依赖，不知道是否有用的软件经历不要卸载
brew uninstall 软件名
 
# 更新软件,如果不指定需要更新的软件，brew会更新所有需要更新的软件
brew upgrade [git]
 
# 更新brew
brew update
 
# 查看帮助
brew --help #查看brew可用的命令
brew 命令名 --help # 查看该命令可用的参数
```

### 3 下载Ubuntu镜像

下载地址 ：[Ubuntu 22.04镜像下载](https://cdimage.ubuntu.com/releases/22.04/release/)

Mac 选择 `64-bit ARM (ARMv8/AArch64) server install image`

### 4 
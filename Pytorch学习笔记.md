### 1. Tensor 张量

* shape是Tensor的维度元组

* 在 PyTorch 中，方法名称后加上下划线 _ 通常表示**该操作是 “in-place” 操作**

  ```python
  x = torch.tensor([1, 2, 3])
  y = x.add(1)  # 创建了一个新的张量 y
  print(x)  # 输出: tensor([1, 2, 3])
  print(y)  # 输出: tensor([2, 3, 4])
  ```

  ```python
  x = torch.tensor([1, 2, 3])
  x.add_(1)  # 在原张量 x 上进行加法操作
  print(x)  # 输出: tensor([2, 3, 4])，x 本身被修改了
  ```

  就地操作可以节省一些内存，但在计算导数时可能会出现问题，因为历史记录会立即丢失。因此，不鼓励使用它们。

* CPU 和 NumPy 数组上的张量可以共享其底层内存位置，并且更改其中一个将更改另一个。



### 2. QuickStart

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 从公开数据集下载训练数据。使用 FashionMNIST 数据集，将数据存储在 data 文件夹中。train=True 表示下载训练集，download=True 表示如果数据集不存在则下载，transform=ToTensor() 将数据转换为 PyTorch 张量格式。
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 下载测试数据集。train=False 表示下载测试集，其余参数与训练数据相同。
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 设置每个批次的大小为 64，这决定了训练过程中每次传递给模型的数据样本数量。
batch_size = 64

# 创建数据加载器。DataLoader 用于将数据分成小批次，并在训练期间自动加载这些数据，以提高效率。
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 迭代 test_dataloader 以获取一批数据（X 和 y），然后打印出 X 和 y 的形状。X 是图像数据，y 是标签。break 语句确保只打印第一批数据的形状。
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 根据可用的设备选择训练设备。如果有 CUDA 支持的 GPU 则使用 "cuda"，如果使用的是 Apple 的 Metal Performance Shaders (MPS) 则使用 "mps"，否则使用 CPU。
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 定义一个神经网络模型，继承自 nn.Module。__init__ 方法中初始化网络结构，首先将输入图像扁平化，然后定义一个包含多个全连接层和 ReLU 激活函数的顺序模块。
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
		#  定义前向传播方法，输入 x 先被扁平化，然后通过线性层和 ReLU 激活层，输出logits（预测结果）。
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 实例化模型并将其移动到指定的设备上。打印出模型结构。
model = NeuralNetwork().to(device)
print(model)

# 定义损失函数为交叉熵损失，用于多分类问题。定义优化器为随机梯度下降 (SGD)，学习率设为 0.001。
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 定义训练函数，接收数据加载器、模型、损失函数和优化器作为参数。size 是数据集的大小，调用 model.train() 将模型设置为训练模式。遍历数据加载器以获取每个批次的输入 X 和标签 y，并将它们移动到指定设备。
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算模型的预测值 pred，并使用损失函数计算损失 loss。
        pred = model(X)
        loss = loss_fn(pred, y)

        # 进行反向传播以计算梯度，使用优化器更新模型参数，并在每次迭代后清零梯度。
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
				
        # 每隔 100 个批次打印一次当前的损失值和当前处理的数据样本数。
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 定义测试函数，计算测试集的损失和准确率。初始化测试损失和正确预测的数量为 0。
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    # 用 torch.no_grad() 上下文管理器关闭梯度计算，以节省内存。在测试数据加载器中遍历每个批次，将数据移动到设备并进行预测。计算总测试损失和正确预测的数量。
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # 计算平均测试损失和准确率，并打印结果。
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 设置训练的轮数（epochs）为 20，并在每一轮中调用 train 和 test 函数进行训练和评估。打印出每一轮的状态。
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# 保存训练后的模型参数到文件 model.pth 中，以便后续加载和使用。
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

**Batch（批次）**

**定义**: Batch 是指在一次迭代中传递给模型的数据样本的数量。模型在处理数据时，会将整个数据集划分成多个小批次，每次处理一个批次。

**作用**: 使用批次的主要原因是内存限制和计算效率。通过将数据集分成多个小批次，可以在显存或内存的限制下进行训练。每次模型更新参数时，都是基于当前批次的数据计算得出的损失和梯度。

**示例**: 如果数据集有 1000 个样本，batch size 设置为 100，那么在训练过程中，会有 10 个批次（每个批次 100 个样本）。

**Epoch（轮次）**

**定义**: Epoch 是指对整个训练数据集进行一次完整的训练过程。一次 epoch 包括所有样本都被用来训练模型一次。

**作用**: 通过多次 epoch，可以让模型在数据集上反复学习，从而提高模型的性能和泛化能力。通常，模型在多个 epoch 上进行训练，以便充分调整其参数。

**示例**: 如果你设置训练 10 个 epochs，那么在训练过程中，整个 1000 个样本的数据集会被用来训练 10 次。
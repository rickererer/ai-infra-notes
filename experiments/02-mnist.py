"""
第二个练习：MNIST 手写数字识别。

和 01 的区别，多了四样东西：
    1. DataLoader，把数据切成 batch
    2. 多层网络 + 激活函数
    3. 交叉熵损失（分类不用 MSE）
    4. 准确率评估（光看 loss 不知道对错）

运行：
    D:/Anaconda/python.exe 02-mnist.py
"""

import os

# 下载数据集要走代理，而且不能用环境里那个默认的，得指定 7897
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

# 数据放在脚本同级的 data 目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------- 数据

# 原始图片是 0-255 的灰度，这里转成 0-1 的浮点，再按 MNIST 的统计量标准化
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)

# batch_size=128，意思是每次喂 128 张图给模型，而不是一次喂 60000 张
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

print(f"训练集 {len(train_ds)} 张，测试集 {len(test_ds)} 张")
print(f"每轮 {len(train_loader)} 个 batch")

# ---------------------------------------------------------------- 模型

model = nn.Sequential(
    nn.Flatten(),          # [128,1,28,28] -> [128,784]，把图拉平成一维
    nn.Linear(784, 128),   # 784 个像素 -> 128 个隐藏特征
    nn.ReLU(),             # 非线性。去掉它，两层线性层等于一层
    nn.Linear(128, 10),    # 128 -> 10，对应 0-9 十个类别
).to(DEVICE)

print(f"参数量: {sum(p.numel() for p in model.parameters())}")

# 分类用交叉熵，不用 MSE。MSE 会把错的预测惩罚得太温柔
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------- 训练

for epoch in range(3):
    model.train()
    t0 = time.perf_counter()
    loss_sum, seen = 0.0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item() * xb.size(0)
        seen += xb.size(0)

    # 评估阶段：不更新参数，也不算梯度
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)      # 10 个分数里取最大的那个当预测
            correct += (pred == yb).sum().item()

    acc = correct / len(test_ds) * 100
    print(f"epoch {epoch + 1}  train_loss {loss_sum / seen:.4f}  "
          f"test_acc {acc:.2f}%  {time.perf_counter() - t0:.1f}s")

# ---------------------------------------------------------------- 看一张

x, y = test_ds[0]
with torch.no_grad():
    logits = model(x.unsqueeze(0).to(DEVICE))[0]
    probs = logits.softmax(0)

print(f"\n第一张测试图，真实标签 {y}，预测 {logits.argmax().item()}")
print("十个类别的概率:")
print("  " + "  ".join(f"{i}:{p:.2f}" for i, p in enumerate(probs.tolist())))

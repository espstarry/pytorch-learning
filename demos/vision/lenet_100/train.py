import torch
from torch import nn

from model import LeNet


# 用随机数据只演示训练循环；真实学习时可以换成 MNIST。
x = torch.randn(16, 1, 32, 32)
y = torch.randint(0, 10, (16,))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(20):
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("input:", x.shape)
print("output:", model(x).shape)
print("loss:", loss.item())
print("device:", device)

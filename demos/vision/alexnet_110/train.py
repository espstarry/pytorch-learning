import torch
from torch import nn

from model import AlexNet


# 224×224 RGB 是经典 AlexNet 的输入尺寸。
x = torch.randn(1, 3, 224, 224)
y = torch.tensor([3])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
model = AlexNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

logits = model(x)
loss = loss_fn(logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("input:", x.shape)
print("output:", model(x).shape)
print("loss:", loss.item())
print("device:", device)

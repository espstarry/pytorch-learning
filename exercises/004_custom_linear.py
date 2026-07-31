import torch
from torch import nn


class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0]]))
        self.bias = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x):
        return x @ self.weight.T + self.bias


x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

model = MyLinear()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(2000):
    y_pred = model(x)
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 200 == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

print("weight:", model.weight.item())
print("bias:", model.bias.item())

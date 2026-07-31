import torch
from torch import nn


# Five input features and two output features.
# Output 1: 3*x1 + 5*x2 + 7*x3 + 11*x4 + 13*x5 + 17
# Output 2: 2*x1 + 4*x2 + 6*x3 + 8*x4 + 10*x5 + 19
x = torch.tensor([
    [1.0, 0.0, 2.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 2.0, 0.0],
    [2.0, 1.0, 0.0, 1.0, 3.0],
    [1.0, 2.0, 3.0, 0.0, 2.0],
    [3.0, 0.0, 1.0, 2.0, 1.0],
    [0.0, 3.0, 2.0, 1.0, 2.0],
])

y_true = torch.tensor([
    [47.0, 43.0],
    [53.0, 45.0],
    [78.0, 67.0],
    [77.0, 67.0],
    [68.0, 57.0],
    [83.0, 71.0],
])

model = nn.Linear(5, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

print("model:", model)
print("weight shape:", model.weight.shape)
print("bias shape:", model.bias.shape)

for step in range(10000):
    y_pred = model(x)
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 2000 == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

print("weight:", model.weight.detach())
print("bias:", model.bias.detach())

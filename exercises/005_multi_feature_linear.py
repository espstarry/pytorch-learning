import torch
from torch import nn


# Target function: y = 3*x1 + 5*x2 + 6.
x = torch.tensor([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0],
    [4.0, 3.0],
])
y_true = torch.tensor([
    [19.0],
    [17.0],
    [35.0],
    [33.0],
])

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(3000):
    y_pred = model(x)
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 500 == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

print("weight:", model.weight.detach())
print("bias:", model.bias.detach())
print("weight shape:", model.weight.shape)
print("bias shape:", model.bias.shape)

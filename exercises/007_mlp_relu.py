import torch
from torch import nn


# Five input features and two output features.
# Output 1: 3*x1 + 5*x2 + 7*x3 + 11*x4 + 13*x5 + 17
# Output 2: 2*x1 + 4*x2 + 6*x3 + 8*x4 + 10*x5 + 19
x = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
])

y_true = torch.tensor([
    [17.0, 19.0],
    [20.0, 21.0],
    [22.0, 23.0],
    [24.0, 25.0],
    [28.0, 27.0],
    [30.0, 29.0],
])

model = nn.Sequential(
    nn.Linear(5, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print("model:", model)
print("first weight shape:", model[0].weight.shape)
print("first bias shape:", model[0].bias.shape)
print("second weight shape:", model[2].weight.shape)
print("second bias shape:", model[2].bias.shape)

criterion = nn.MSELoss()

for step in range(5000):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 1000 == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

print("first layer weight:", model[0].weight.detach())
print("first layer bias:", model[0].bias.detach())
print("second layer weight:", model[2].weight.detach())
print("second layer bias:", model[2].bias.detach())

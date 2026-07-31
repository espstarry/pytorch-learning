import torch
from torch import nn


# Class 0: points near the origin.
# Class 1: points farther from the origin.
x = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [2.0, 3.0],
    [3.0, 2.0],
    [3.0, 3.0],
])

# CrossEntropyLoss expects class indices, not one-hot vectors.
labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(2000):
    logits = model(x)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 400 == 0:
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        print(
            f"step={step:04d}",
            f"loss={loss.item():.6f}",
            f"accuracy={accuracy.item():.2f}",
        )

with torch.no_grad():
    logits = model(x)
    predictions = logits.argmax(dim=1)

print("logits:")
print(logits)
print("predictions:", predictions)
print("labels:", labels)

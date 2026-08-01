import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 2),
        )

    def forward(self, x):
        return self.net(x)


x = torch.randn(8, 1, 16, 16)
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

model = TinyCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(100):
    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("input:", x.shape)
print("output:", model(x).shape)
print("loss:", loss.item())

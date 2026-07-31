import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class PointDataset(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.labels[index]


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
labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

dataset = PointDataset(x, labels)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()

    for batch_x, batch_labels in loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 20 == 0:
        print(f"epoch={epoch:03d} loss={loss.item():.6f}")

model.eval()
with torch.no_grad():
    logits = model(x)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).float().mean()

print("predictions:", predictions)
print("labels:", labels)
print("accuracy:", accuracy.item())

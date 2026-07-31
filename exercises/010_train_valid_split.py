import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


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

dataset = TensorDataset(x, labels)
train_dataset, valid_dataset = random_split(
    dataset,
    [6, 2],
    generator=torch.Generator().manual_seed(0),
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()

    for batch_x, batch_labels in train_loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for batch_x, batch_labels in valid_loader:
            logits = model(batch_x)
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

        valid_accuracy = correct / total

    if epoch % 20 == 0:
        print(
            f"epoch={epoch:03d}",
            f"train_loss={loss.item():.6f}",
            f"valid_accuracy={valid_accuracy:.2f}",
        )

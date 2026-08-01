import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import AlexNet


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--train-samples", type=int, default=1000)
parser.add_argument("--test-samples", type=int, default=200)
args = parser.parse_args()

# CIFAR-10 是 32×32 RGB；Resize 后符合经典 AlexNet 的 224×224 输入。
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
train_set = datasets.CIFAR10("data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform)
train_set = Subset(train_set, range(min(args.train_samples, len(train_set))))
test_set = Subset(test_set, range(min(args.test_samples, len(test_set))))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print("device:", device)


def accuracy(loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


for epoch in range(1, args.epochs + 1):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}: test_accuracy={accuracy(test_loader):.3f}")

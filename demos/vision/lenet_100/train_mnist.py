import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet
from export_tensors import export


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--export-tensors", action="store_true")
args = parser.parse_args()

# MNIST 原图是 28x28；LeNet-5 的经典输入尺寸是 32x32。
transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
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
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}: test_accuracy={accuracy(test_loader):.3f}")

if args.export_tensors:
    images, labels = next(iter(test_loader))
    export(model, images[:2].to(device), labels[:2])
    print("saved tensor_data.json with trained model outputs")

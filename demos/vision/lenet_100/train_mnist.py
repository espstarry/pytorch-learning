import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet
from export_tensors import export

sys.path.append(str(Path(__file__).resolve().parents[3]))
from loss_viewer.recorder import LossRecorder


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
losses = LossRecorder("LeNet")


def evaluate(loader):
    correct = 0
    total = 0
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    train_items = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
        train_items += labels.size(0)

    test_loss, test_accuracy = evaluate(test_loader)
    losses.add(epoch, train_loss=train_loss / train_items, test_loss=test_loss, test_accuracy=test_accuracy)
    print(f"epoch {epoch}: train_loss={train_loss / train_items:.4f} test_loss={test_loss:.4f} test_accuracy={test_accuracy:.3f}")

losses.save("loss_data.json")
print("saved loss_data.json")

if args.export_tensors:
    images, labels = next(iter(test_loader))
    export(model, images[:2].to(device), labels[:2])
    print("saved tensor_data.json with trained model outputs")

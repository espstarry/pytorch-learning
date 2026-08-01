import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from model import AlexNet

sys.path.append(str(Path(__file__).resolve().parents[3]))
from loss_viewer.recorder import LossRecorder


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--train-samples", type=int, default=1000)
parser.add_argument("--test-samples", type=int, default=200)
parser.add_argument("--source", choices=["torchvision", "huggingface"], default="torchvision")
args = parser.parse_args()

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print("CIFAR-10 labels:", dict(enumerate(class_names)))

# CIFAR-10 是 32×32 RGB；Resize 后符合经典 AlexNet 的 224×224 输入。
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

if args.source == "torchvision":
    train_set = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform)
else:
    from datasets import load_dataset

    class HFCIFAR10(Dataset):
        def __init__(self, split):
            self.data = load_dataset("uoft-cs/cifar10", split=split)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            sample = self.data[index]
            return transform(sample["img"]), sample["label"]

    train_set = HFCIFAR10("train")
    test_set = HFCIFAR10("test")
train_set = Subset(train_set, range(min(args.train_samples, len(train_set))))
test_set = Subset(test_set, range(min(args.test_samples, len(test_set))))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print("device:", device)
losses = LossRecorder("AlexNet")


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
        loss = loss_fn(model(images), labels)
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

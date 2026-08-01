import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet


transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
print("device:", device)

# 只训练一轮，目的是快速得到可以探测的真实模型。
model.train()
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    loss = loss_fn(model(images), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 找一张真实标签为 2 的测试图片。
for index in range(len(test_set)):
    image, label = test_set[index]
    if label == 2:
        original = image.unsqueeze(0).to(device)
        break

# 只擦掉最底部 5 行中实际有墨迹的像素，近似移除底部横线。
masked = original.clone()
ink_rows = (original[0, 0] > 0.1).any(dim=1).nonzero().flatten()
bottom_rows = ink_rows[-5:]
for row in bottom_rows:
    ink_columns = original[0, 0, row] > 0.1
    masked[0, 0, row, ink_columns] = 0

model.eval()
with torch.no_grad():
    original_logits = model(original)[0]
    masked_logits = model(masked)[0]


def step(name, tensor, description):
    values = tensor.detach().cpu()
    if values.ndim == 2:
        values = values.unsqueeze(1)
    return {
        "name": name,
        "shape": list(tensor.shape),
        "values": values.tolist(),
        "description": description,
    }


steps = [
    step("original_2", original, "原始 MNIST 数字 2"),
    step("original_logits", original_logits.unsqueeze(0), "原图的 10 个类别 logits"),
    step("bottom_erased", masked, f"只擦掉最底部 5 行的墨迹，行号是 {bottom_rows.tolist()}"),
    step("erased_logits", masked_logits.unsqueeze(0), "只遮挡底部笔画后的 10 个类别 logits"),
]

with open("occlusion_data.json", "w", encoding="utf-8") as file:
    json.dump({"model": "LeNet bottom-stroke probe", "steps": steps}, file)

print(f"true label: {label}")
for digit in (2, 7, 9):
    print(
        f"class {digit}: "
        f"original={original_logits[digit]:.3f}, "
        f"bottom_erased={masked_logits[digit]:.3f}"
    )
print("saved occlusion_data.json")

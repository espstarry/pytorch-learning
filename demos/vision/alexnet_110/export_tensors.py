import json

import torch

from model import AlexNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device).eval()
images = torch.randn(1, 3, 224, 224, device=device)
steps = []


def save(name, tensor, description):
    values = tensor.detach().cpu()
    if values.ndim == 4:
        values = values[:1, :8]
    elif values.ndim == 2:
        values = values[:1].unsqueeze(1)
    steps.append({
        "name": name,
        "shape": list(tensor.shape),
        "values": values.tolist(),
        "description": description,
    })


with torch.no_grad():
    save("input", images, "一张随机 RGB 图像")
    x = images
    for index, layer in enumerate(model.features):
        x = layer(x)
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
            save(f"features_{index}_{layer.__class__.__name__}", x, str(layer))
    x = torch.flatten(x, 1)
    save("flatten", x, "256×6×6 展平成 9216 维向量")
    x = model.classifier(x)
    save("logits", x, "10 个类别的 logits")

with open("tensor_data.json", "w", encoding="utf-8") as file:
    json.dump({"model": "AlexNet", "steps": steps}, file)

print("saved tensor_data.json")

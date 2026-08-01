import json

import torch
from torchvision import datasets, transforms

from model import LeNet


def export(model, images, labels, path="tensor_data.json"):
    model.eval()
    steps = []

    def save(name, tensor, description):
        values = tensor.detach().cpu()
        if values.ndim == 4:
            values = values[:2, :8]
        elif values.ndim == 2:
            values = values[:2].unsqueeze(1)
        steps.append({
            "name": name,
            "shape": list(tensor.shape),
            "values": values.tolist(),
            "description": description,
        })

    with torch.no_grad():
        save("input", images, f"真实 MNIST 图片，标签是 {labels.tolist()}")
        x = model.features[0](images)
        save("conv1", x, "第一层卷积：1 个 channel 变成 6 个 feature maps")
        x = model.features[1](x)
        save("tanh1", x, "Tanh 激活")
        x = model.features[2](x)
        save("pool1", x, "平均池化：空间尺寸减半")
        x = model.features[3](x)
        save("conv2", x, "第二层卷积：6 个 channel 变成 16 个 feature maps")
        x = model.features[4](x)
        save("tanh2", x, "Tanh 激活")
        x = model.features[5](x)
        save("pool2", x, "平均池化：得到 16×5×5")
        x = torch.flatten(x, 1)
        save("flatten", x, "16×5×5 展平成 400 维向量")
        x = model.classifier(x)
        save("logits", x, "10 个类别的 logits")

    with open(path, "w", encoding="utf-8") as file:
        json.dump({"model": "LeNet-5", "steps": steps}, file)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
    images = torch.stack([test_set[index][0] for index in range(2)]).to(device)
    labels = torch.tensor([test_set[index][1] for index in range(2)])
    export(LeNet().to(device), images, labels)
    print(f"saved tensor_data.json with real MNIST images on {device}")


if __name__ == "__main__":
    main()

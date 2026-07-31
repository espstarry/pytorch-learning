import torch


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_labels in loader:
            logits = model(batch_x)
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    return correct / total


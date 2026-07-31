import torch


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_labels in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_labels)

            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


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

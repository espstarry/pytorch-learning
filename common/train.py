def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_labels in loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = batch_labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

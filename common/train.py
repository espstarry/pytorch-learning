def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for batch_x, batch_labels in loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


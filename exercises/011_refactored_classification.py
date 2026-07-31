import torch
from torch import nn
from pathlib import Path

from common.data import make_point_loaders
from common.evaluate import evaluate_accuracy, evaluate_loss
from common.models import make_point_classifier
from common.train import train_one_epoch


SEED = 0
BATCH_SIZE = 2
LEARNING_RATE = 0.01
EPOCHS = 100
HIDDEN_FEATURES = 8
NUM_CLASSES = 2
PATIENCE = 10
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
LAST_CHECKPOINT_PATH = Path("checkpoints/last_checkpoint.pt")


def main():
    # Fix model initialization and DataLoader shuffle for reproducible runs.
    torch.manual_seed(SEED)

    train_loader, valid_loader = make_point_loaders(batch_size=BATCH_SIZE)
    model = make_point_classifier(
        hidden_features=HIDDEN_FEATURES,
        num_classes=NUM_CLASSES,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    best_valid_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        valid_loss = evaluate_loss(model, valid_loader, criterion)
        valid_accuracy = evaluate_accuracy(model, valid_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss,
                "best_epoch": best_epoch,
                "config": {
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "hidden_features": HIDDEN_FEATURES,
                    "num_classes": NUM_CLASSES,
                    "patience": PATIENCE,
                },
            },
            LAST_CHECKPOINT_PATH,
        )

        if epoch % 20 == 0:
            print(
                f"epoch={epoch:03d}",
                f"train_loss={train_loss:.6f}",
                f"valid_loss={valid_loss:.6f}",
                f"valid_accuracy={valid_accuracy:.2f}",
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"early stopping at epoch={epoch:03d}")
            break

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    print(f"best model loaded from: {CHECKPOINT_PATH}")
    print(f"last checkpoint saved at: {LAST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

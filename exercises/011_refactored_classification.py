import torch
from torch import nn

from common.data import make_point_loaders
from common.evaluate import evaluate_accuracy, evaluate_loss
from common.models import make_point_classifier
from common.train import train_one_epoch


def main():
    train_loader, valid_loader = make_point_loaders(batch_size=2)
    model = make_point_classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        valid_loss = evaluate_loss(model, valid_loader, criterion)
        valid_accuracy = evaluate_accuracy(model, valid_loader)

        if epoch % 20 == 0:
            print(
                f"epoch={epoch:03d}",
                f"train_loss={train_loss:.6f}",
                f"valid_loss={valid_loss:.6f}",
                f"valid_accuracy={valid_accuracy:.2f}",
            )


if __name__ == "__main__":
    main()

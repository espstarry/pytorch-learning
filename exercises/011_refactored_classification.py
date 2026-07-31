import torch
from torch import nn
from pathlib import Path
import json

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
HISTORY_PATH = Path("artifacts/history.json")
LOSS_CURVE_PATH = Path("artifacts/loss_curve.svg")
ACCURACY_CURVE_PATH = Path("artifacts/accuracy_curve.svg")


def save_curve(path, title, y_label, series, y_min=0.0, y_max=None):
    """Save a small dependency-free SVG line chart from experiment history."""
    width, height = 720, 420
    left, right, top, bottom = 70, 25, 45, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for values, _, _ in series for value in values]
    if y_max is None:
        y_max = max(all_values) * 1.1 if all_values else 1.0
    y_max = max(y_max, y_min + 1e-6)
    max_epoch = max(len(series[0][0]) - 1, 1)

    def point(index, value):
        x = left + plot_width * index / max_epoch
        y = top + plot_height * (y_max - value) / (y_max - y_min)
        return f"{x:.2f},{y:.2f}"

    lines = []
    for values, color, _ in series:
        points = " ".join(point(index, value) for index, value in enumerate(values))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    legend = "".join(
        f'<text x="{left + index * 150}" y="25" fill="{color}">{label}</text>'
        for index, (_, color, label) in enumerate(series)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="25" font-family="sans-serif" font-size="18">{title}</text>
{legend}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="black"/>
<text x="10" y="{top + plot_height / 2}" font-family="sans-serif" font-size="14" transform="rotate(-90 10,{top + plot_height / 2})">{y_label}</text>
<text x="{width / 2}" y="{height - 10}" font-family="sans-serif" font-size="14">epoch</text>
{''.join(lines)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


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
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        valid_loss = evaluate_loss(model, valid_loader, criterion)
        valid_accuracy = evaluate_accuracy(model, valid_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_accuracy,
            }
        )

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
    with HISTORY_PATH.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    valid_losses = [item["valid_loss"] for item in history]
    valid_accuracies = [item["valid_accuracy"] for item in history]
    save_curve(
        LOSS_CURVE_PATH,
        "Training and validation loss",
        "loss",
        [(train_losses, "#1f77b4", "train loss"), (valid_losses, "#d62728", "valid loss")],
    )
    save_curve(
        ACCURACY_CURVE_PATH,
        "Validation accuracy",
        "accuracy",
        [(valid_accuracies, "#2ca02c", "valid accuracy")],
        y_max=1.05,
    )

    print(f"best model loaded from: {CHECKPOINT_PATH}")
    print(f"last checkpoint saved at: {LAST_CHECKPOINT_PATH}")
    print(f"history saved at: {HISTORY_PATH}")
    print(f"loss curve saved at: {LOSS_CURVE_PATH}")
    print(f"accuracy curve saved at: {ACCURACY_CURVE_PATH}")


if __name__ == "__main__":
    main()

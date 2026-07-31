from dataclasses import asdict, dataclass

import torch
from torch import nn

from common.data import make_point_loaders
from common.experiment import ClassificationExperiment, seed_everything
from common.models import make_point_classifier


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "point_classifier_baseline"
    seed: int = 0
    batch_size: int = 2
    learning_rate: float = 0.01
    epochs: int = 100
    hidden_features: int = 8
    num_classes: int = 2
    patience: int = 10


CONFIG = ExperimentConfig()


def main():
    seed_everything(CONFIG.seed)
    train_loader, valid_loader = make_point_loaders(
        batch_size=CONFIG.batch_size,
        seed=CONFIG.seed,
    )
    model = make_point_classifier(
        hidden_features=CONFIG.hidden_features,
        num_classes=CONFIG.num_classes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    experiment = ClassificationExperiment(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        config=asdict(CONFIG),
    )
    summary = experiment.run(train_loader, valid_loader)

    print(f"run saved at: {experiment.run_dir}")
    print(f"best epoch: {summary['best_epoch']}")
    print(f"best validation accuracy: {summary['best_valid_accuracy']:.2f}")


if __name__ == "__main__":
    main()

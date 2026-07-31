from torch import nn


def make_point_classifier(hidden_features=8, num_classes=2):
    return nn.Sequential(
        nn.Linear(2, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, num_classes),
    )

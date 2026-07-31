import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def make_point_loaders(batch_size=2, seed=0):
    x = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [3.0, 3.0],
    ])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    dataset = TensorDataset(x, labels)
    train_dataset, valid_dataset = random_split(
        dataset,
        [6, 2],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader

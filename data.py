"""MNIST data loading and consistent training/inference normalization."""
from torch.utils.data import DataLoader, random_split
import torch
from torchvision import datasets, transforms

MEAN, STD = 0.1307, 0.3081


def transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])


def training_loaders(root, batch_size, seed, train_limit=None):
    dataset = datasets.MNIST(root, train=True, download=True, transform=transform())
    train, val = random_split(dataset, [55000, 5000],
                              generator=torch.Generator().manual_seed(seed))
    if train_limit is not None:
        if not 1 <= train_limit <= len(train):
            raise ValueError('--train-limit must be between 1 and 55000')
        train = torch.utils.data.Subset(train, range(train_limit))
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed)),
            DataLoader(val, batch_size=batch_size))


def test_loader(root, batch_size):
    return DataLoader(datasets.MNIST(root, train=False, download=True,
                                    transform=transform()), batch_size=batch_size)

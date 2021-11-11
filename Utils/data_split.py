import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def fashion_train_val_test_split(train_batch=128, val_batch=128, test_batch=256):
    transform_train = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
    )
    train_set = datasets.FashionMNIST(root='../Datasets/fashion', train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(train_set, batch_size=train_batch, shuffle=False)

    transform_test = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    val_test_set = datasets.FashionMNIST(root='../Datasets/fashion', train=False, download=False,
                                         transform=transform_test)
    val_test_size = len(val_test_set.targets)
    val_set, test_set = data.random_split(val_test_set,
                                          [int(val_test_size * 0.5), val_test_size - int(val_test_size * 0.5)],
                                          generator=torch.Generator().manual_seed(2021))
    val_loader = data.DataLoader(val_set, batch_size=val_batch, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=test_batch, shuffle=False)

    return train_set, train_loader, val_set, val_loader, test_set, test_loader


def cifar10_train_val_test_split(train_batch=128, val_batch=128, test_batch=256):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    train_set = datasets.CIFAR10(root='../Datasets/cifar10', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=train_batch, shuffle=False)
    val_test_set = datasets.CIFAR10(root='../Datasets/cifar10', train=False, download=False, transform=transform)
    val_test_size = len(val_test_set.targets)
    val_set, test_set = data.random_split(val_test_set,
                                          [int(val_test_size * 0.5), val_test_size - int(val_test_size * 0.5)],
                                          generator=torch.Generator().manual_seed(2021))
    val_loader = data.DataLoader(val_set, batch_size=val_batch, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=test_batch, shuffle=False)
    return train_set, train_loader, val_set, val_loader, test_set, test_loader


def cifar100_train_val_test_split(train_batch=128, val_batch=128, test_batch=256):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    train_set = datasets.CIFAR100(root='../Datasets/cifar100', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=train_batch, shuffle=False)

    val_test_set = datasets.CIFAR100(root='../Datasets/cifar100', train=False, download=False, transform=transform)
    val_test_size = len(val_test_set.targets)
    val_set, test_set = data.random_split(val_test_set,
                                          [int(val_test_size * 0.5), val_test_size - int(val_test_size * 0.5)],
                                          generator=torch.Generator().manual_seed(2021))
    val_loader = data.DataLoader(val_set, batch_size=val_batch, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=test_batch, shuffle=False)
    return train_set, train_loader, val_set, val_loader, test_set, test_loader


if __name__ == '__main__':
    train_set, train_loader, val_set, val_loader, test_set, test_loader = cifar10_train_val_test_split()
    print(len(train_set), len(val_set), len(test_set))
    train_set, train_loader, val_set, val_loader, test_set, test_loader = cifar100_train_val_test_split()
    print(len(train_set), len(val_set), len(test_set))

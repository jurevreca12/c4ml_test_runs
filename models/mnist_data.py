import torch
import torchvision
from torchvision.transforms import v2


def convert_unit_nums(x):
    return torch.round(x * 255).to(torch.float32)


def get_data_loaders(batch_size=256):
    trans = v2.Compose([v2.ToTensor(), v2.Lambda(convert_unit_nums)])
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=trans
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_set = torchvision.datasets.MNIST(root="./data", download=True, transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, test_loader

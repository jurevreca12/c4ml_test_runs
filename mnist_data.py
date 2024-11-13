import torch
import torchvision
import torchvision.transforms as transforms


def get_data_loaders(batch_size=256): 
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=trans
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4
    )
    test_set = torchvision.datasets.MNIST(
        root='./data', 
        download=True, 
        transform=trans
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4
    )
    return train_loader, test_loader
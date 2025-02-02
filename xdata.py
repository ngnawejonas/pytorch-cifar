import torchvision
import torchvision.transforms as transforms


def get_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=False, download=True, transform=transform_test)
    return trainset, testset
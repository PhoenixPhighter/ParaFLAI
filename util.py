import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

DATA_ROOT = "./dataset"


# Load CIFAR-10
def load_data():
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
    testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


# Partition Dataset
def load_partition(idx: int, total: int):
    assert idx in range(total)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / total)
    print(f"train size {n_train}")
    n_test = int(num_examples["testset"] / total)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)

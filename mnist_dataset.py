"""
Mnist dataset preprocess
"""

import torch
import torchvision
import matplotlib.pyplot as plt


def mnist_dataset(is_shuffle=True):
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./dataset', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./dataset', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=is_shuffle)

    return train_loader, test_loader


def print_dim(train_dataset, test_dataset):
    # print information
    print(train_dataset.dataset)
    print("----"*10)
    print(test_dataset.dataset)
    print("----" * 10)
    train_features, train_labels = next(iter(train_dataset))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")


def show_img(data):

    examples = enumerate(data)
    # example_data：64x1x28x28, example_targets: 64 label
    batch_idx, (example_data, example_targets) = next(examples)

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == '__main__':
    # load data
    train_dataset, test_dataset = mnist_dataset()
    # dataset dimension
    print_dim(train_dataset, test_dataset)
    # show img
    show_img(train_dataset)


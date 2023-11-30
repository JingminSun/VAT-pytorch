import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms

train_path = f'/notebooks/VAT-pytorch/data/CIFAR10/train/'
test_path = f'/notebooks/VAT-pytorch/data/CIFAR10/test/'



train_dataset = datasets.CIFAR10(train_path, download=True, train=True, transform=None)
test_dataset = datasets.CIFAR10(test_path, download=True, train=False, transform=None)

# a = train_dataset.train_data
print("b",b)

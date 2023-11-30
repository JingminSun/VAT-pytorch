import numpy as np
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms
from sklearn import  datasets as skdatasets


class SimpleDataset(Dataset):

    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img = self.x[index]
        if self.transform is not None:
             img = self.transform(img)


        target = self.y[index]
        return img, target

    def __len__(self):
        return len(self.x)


class InfiniteSampler(sampler.Sampler):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return None


def get_iters(
        dataset='CIFAR10', root_path='.', data_transforms=None,
        n_labeled=4000, valid_size=1000,
        l_batch_size=32, ul_batch_size=128, test_batch_size=256,
        workers=8, pseudo_label=None):
    
    train_path = f'{root_path}/data/{dataset}/train/'
    test_path = f'{root_path}/data/{dataset}/test/'

    if dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(train_path, download=True, train=True, transform=None)
        test_dataset = datasets.CIFAR10(test_path, download=True, train=False, transform=None)
    elif dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(train_path, download=True, train=True, transform=None)
        test_dataset = datasets.CIFAR100(test_path, download=True, train=False, transform=None)
    elif dataset == 'MNIST':
        train_dataset = datasets.MNIST(train_path, download=True, train=True, transform=None)
        test_dataset = datasets.MNIST(test_path, download=True, train=False, transform=None)
    elif dataset == 'moon':
        train_dataset = skdatasets.make_moons(n_samples=50000, noise=0.1)
        test_dataset = skdatasets.make_moons(n_samples=10000, noise=0.1)
    else:
        raise ValueError

    if data_transforms is None:
        if dataset == 'MNIST':
            data_transforms = {
                'train': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]),
                'eval': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]),
            }
        elif dataset == 'moon':
            data_transforms = {
                'train': None,
                'eval': None,
            }

        else:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
                'eval': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            }

    if dataset == 'moon':
        x_train, y_train = train_dataset
        x_test, y_test = test_dataset
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
    else:
        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

    randperm = np.random.permutation(len(x_train))
    labeled_idx = randperm[:n_labeled]
    validation_idx = randperm[n_labeled:n_labeled + valid_size]
    unlabeled_idx = randperm[n_labeled + valid_size:]

    x_labeled = x_train[labeled_idx]
    x_validation = x_train[validation_idx]
    x_unlabeled = x_train[unlabeled_idx]

    y_labeled = y_train[labeled_idx]
    y_validation = y_train[validation_idx]
    if pseudo_label is None:
        y_unlabeled = y_train[unlabeled_idx]
    else:
        assert isinstance(pseudo_label, np.ndarray)
        y_unlabeled = pseudo_label
    
    data_iterators = {
        'labeled': iter(DataLoader(
            SimpleDataset(x_labeled, y_labeled, data_transforms['train']),
            batch_size=l_batch_size, num_workers=workers,
            sampler=InfiniteSampler(len(x_labeled)),
        )),
        'unlabeled': iter(DataLoader(
            SimpleDataset(x_unlabeled, y_unlabeled, data_transforms['train']),
            batch_size=ul_batch_size, num_workers=workers,
            sampler=InfiniteSampler(len(x_unlabeled)),
        )),
        'make_pl': iter(DataLoader(
            SimpleDataset(x_unlabeled, y_unlabeled, data_transforms['eval']),
            batch_size=ul_batch_size, num_workers=workers, shuffle=False
        )),
        'val': iter(DataLoader(
            SimpleDataset(x_validation, y_validation, data_transforms['eval']),
            batch_size=len(x_validation), num_workers=workers, shuffle=False
        )),
        'test': iter(DataLoader(
            SimpleDataset(x_test, y_test, data_transforms['eval']),
            batch_size=test_batch_size, num_workers=workers, shuffle=False
        ))
    }

    traindata = {
        'x_labeled': x_labeled,
        'y_labeled': y_labeled,
        'x_unlabeled': x_unlabeled,
        'y_unlabeled': y_unlabeled
    }

    return data_iterators,traindata

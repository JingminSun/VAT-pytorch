import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from vat import VATLoss
import data_utils
import utils
import os
import logging
import random
from pathlib import Path


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        return x


def train(args, model, device, data_iterators, optimizer,PATH, newexp =True):
    iteration = 0
    if not newexp:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ce_losses = checkpoint['celoss']
        vat_losses = checkpoint['vatloss']
        prec1 = checkpoint['prec1']
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration'] + 1
        print(f'\nIteration: {iteration}\t'
              f'CrossEntropyLoss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
              f'VATLoss {vat_losses.val:.4f} ({vat_losses.avg:.4f})\t'
              f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})')

    model.train()

    model.train()
    epoch = 1
    for i in tqdm(range(iteration,args.iters)):
        
        # reset
        if i % args.log_interval == 0:
            ce_losses = utils.AverageMeter()
            vat_losses = utils.AverageMeter()
            prec1 = utils.AverageMeter()
        
        x_l, y_l = next(data_iterators['labeled'])
        x_ul, _ = next(data_iterators['unlabeled'])

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_ul = x_ul.to(device)

        optimizer.zero_grad()

        vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        cross_entropy = nn.CrossEntropyLoss()

        lds = vat_loss(model, x_ul)
        output = model(x_l)
        classification_loss = cross_entropy(output, y_l)
        loss = classification_loss + args.alpha * lds
        loss.backward()
        optimizer.step()

        acc = utils.accuracy(output, y_l)
        ce_losses.update(classification_loss.item(), x_l.shape[0])
        vat_losses.update(lds.item(), x_ul.shape[0])
        prec1.update(acc.item(), x_l.shape[0])

        if i % args.log_interval == 0:
            logging.info(f'\nIteration: {i}\t'
                  f'CrossEntropyLoss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                  f'VATLoss {vat_losses.val:.4f} ({vat_losses.avg:.4f})\t'
                  f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})')

        torch.save({
            'epoch': 1,
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'celoss': ce_losses,
            'vatloss': vat_losses,
            'prec1': prec1
        }, PATH)


def test(model, device, data_iterators):
    model.eval()
    correct = 0
    length = 0
    with torch.no_grad():
        for x, y in tqdm(data_iterators['test']):
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                outputs = model(x)
            correct += torch.eq(outputs.max(dim=1)[1], y).detach().cpu().float().sum()
            length += x.shape[0]
            # print(x.shape[0], length)
        test_acc = correct / length * 100.

    print(f'\nTest Accuracy: {test_acc:.4f}%\n')


def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=10000, metavar='N',
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT (default: 0.1)')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT (default: 1)')
    parser.add_argument('--workers', type=int, default=8, metavar='W',
                        help='number of CPU')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--exp-id', type=str, default="", metavar='EID',
                        help='experiment id')
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='GPU', help='dataset')
    return parser

def setup(device,args):


    new_exp = False
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # Create a Path object for the base directory
    base_dir = Path("checkpoint") / args.exp_id

    # Check if the experiment directory already exists
    if not base_dir.exists():
        # Create the directory, including any necessary parent directories
        base_dir.mkdir(parents=True, exist_ok=True)
        # Set new_exp to True if the directory was just created
        new_exp = True

    # The PATH variable is now a string representation of the base_dir Path object
    PATH = str(base_dir)
    # print(PATH)
    return model,optimizer,PATH, new_exp


    # test(model, device, data_iterators)

def valid_only(path,device):

    model = Net().to(device)
    model,_= utils.load_checkpoint(model, path, optimizer=None)
    test(model, device, data_iterators)



if __name__ == '__main__':
    parser = get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.exp_id == "":
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        args.exp_id = "".join(random.choice(chars) for _ in range(10))

    data_iterators = data_utils.get_iters(
        dataset=args.dataset,
        root_path='.',
        l_batch_size=args.batch_size,
        ul_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        workers=args.workers
    )

    model,optimizer,directory, newexp = setup(device,args)
    try:
        utils.set_logger(directory+ "/train.log")
    except Exception as e:
        print(f"Error setting up logger: {e}")

    train(args, model, device, data_iterators, optimizer, directory+'/vat_' + args.dataset + '.pth',newexp)
    valid_only(directory+directory+'/vat_' + args.dataset + '.pth',device)




#      VAT: (x_l, y_l)  (model) (x_ul)= y_ul vatloss  (y_ul)
#      AT:

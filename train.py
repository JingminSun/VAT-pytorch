import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from robustness import VATLoss, WassersteinLoss
import data_utils
import utils
import os
import logging
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from networks import Net, SimpleNet, ConvSmallCIFAR10, ConvSmallSVHN
from plot import plot_acc, plot_shown


def train(args, model, device, data_loader, optimizer, scheduler, directory):
    if args.method == 'vat':
        logging.info(f"Starting  VAT training for experiment {args.exp_id}")
    elif args.method == 'wrm':
        logging.info(f"Starting  WRM training for experiment {args.exp_id}")
    elif args.method == 'reg':
        logging.info(f"Starting  regular training for experiment {args.exp_id}")
    else:
        raise ValueError
    subdire = '/' + args.method + '_'
    PATH = directory + subdire + args.dataset + '.pth'
    iteration = -1
    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ce_losses = checkpoint['celoss']
        regularization_losses = checkpoint['regloss']
        prec1 = checkpoint['prec1']
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        logging.info(f'\nIteration: {iteration}\t'
                     f'Classification {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                     f'Regularization Loss {regularization_losses.val:.4f} ({regularization_losses.avg:.4f})\t'
                     f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})')

    model.train()
    epoch = 1

    data_iterators = {
        'labeled': iter(data_loader['labeled']),
        'unlabeled': iter(data_loader['unlabeled']),
    }
    for i in tqdm(range(iteration + 1, args.iters)):

        # reset
        if i % args.log_interval == 0:
            ce_losses = utils.AverageMeter()
            regularization_losses = utils.AverageMeter()
            prec1 = utils.AverageMeter()

        x_l, y_l = next(data_iterators['labeled'])
        x_ul, _ = next(data_iterators['unlabeled'])
        x_l, y_l = x_l.to(device), y_l.to(device)
        x_ul = x_ul.to(device)

        optimizer.zero_grad()
        cross_entropy = nn.CrossEntropyLoss()

        if args.method == 'vat':
            vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
            regulation_loss = vat_loss(model, x_ul) #lds

            # classification_loss = vat_loss(model, x_l, pred=y_l)#
        elif args.method == 'wrm':
            wrmloss = WassersteinLoss(xi=args.xi_wrm, eps=args.eps_wrm, ip=args.ip_wrm)
            regulation_loss = wrmloss(model, x_ul, cross_entropy)
            # classification_loss = wrmloss(model, x_l, cross_entropy, pred=y_l)
        else:
            regulation_loss = torch.tensor(0.0)
            # classification_loss = cross_entropy(model(x_l), y_l) #l()

        output = model(x_l)
        classification_loss = cross_entropy(output, y_l)

        loss = classification_loss + args.alpha * regulation_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = utils.accuracy(output, y_l)
        ce_losses.update(classification_loss.item(), x_l.shape[0])
        regularization_losses.update(regulation_loss.item(), x_ul.shape[0])
        prec1.update(acc.item(), x_l.shape[0])

        if args.plot_shown:
            if i in args.plot_iters:
                os.makedirs(directory + subdire + args.dataset+'/iteration' + str(i), exist_ok=True)

                save_path = directory + subdire + args.dataset+'/iteration' + str(i) + '/vat_'  + '.pth'

                arguments = {
                    'iteration': i,
                    'celoss': ce_losses,
                    'regloss': regularization_losses,
                    'prec1': prec1,
                    'val': False
                }

                utils.save_checkpoint(model, optimizer, scheduler, arguments, save_path)

                # torch.save({
                #         'epoch': 1,
                #         'iteration': i,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'celoss': ce_losses,
                #         'regloss': regularization_losses,
                #         'prec1': prec1
                #     },  directory+ subdire+ '/iteration'+ str(i) +args.dataset + '.pth')

        if i % args.log_interval == 0:
            model.eval()  # Set the model to evaluation mode
            val_ce_losses = utils.AverageMeter()
            val_prec1 = utils.AverageMeter()
            val_ce_losses_per = utils.AverageMeter()
            val_prec2 = utils.AverageMeter()



            for x_val, y_val in tqdm(iter(data_loader['val'])):
                with torch.no_grad():
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    x_val_perturb = x_val + torch.rand_like(x_val)
                    output_val = model(x_val)
                    output_val_per = model(x_val_perturb)
                    # print("xval", x_val[0,:], "xvalperturb", x_val_perturb[0,:], "outputval", output_val[0,:], "outputvalperturb", output_val_per[0,:])
                val_loss = cross_entropy(output_val, y_val)
                val_acc = utils.accuracy(output_val, y_val)
                val_loss_per = cross_entropy(output_val_per, y_val)
                val_ce_losses_per.update(val_loss_per.item(), x_val.size(0))
                val_acc_per = utils.accuracy(output_val_per, y_val)
                val_prec2.update(val_acc_per.item(), x_val.size(0))
                # print("valloss", val_loss, "valacc", val_acc, "vallossper", val_loss_per, "valaccper", val_acc_per)

                val_ce_losses.update(val_loss.item(), x_val.size(0))
                val_prec1.update(val_acc.item(), x_val.size(0))

            model.train()  # Set the model back to training mode
            logging.info(f'\nIteration: {i}\t'
                         f'Classification {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                         f'Regularization Loss {regularization_losses.val:.4f} ({regularization_losses.avg:.4f})\t'
                         f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})\t'
                         f'Validation - Loss: {val_ce_losses.avg:.4f}, Accuracy: {val_prec1.avg:.2f}%\t'
                         f'Validation - Loss: {val_ce_losses_per.avg:.4f}, Accuracy: {val_prec2.avg:.2f}%')

            if args.plot_acc:
                os.makedirs(directory + subdire + args.dataset+ '/iteration' + str(i), exist_ok=True)

                save_path = directory + subdire+ args.dataset + '/iteration' + str(i) + '/'+ str(args.method)  + '.pth'

                arguments = {
                    'iteration': i,
                    'celoss': ce_losses,
                    'regloss': regularization_losses,
                    'prec1': prec1,
                    'val_ce_losses': val_ce_losses,
                    'val_prec1': val_prec1,
                    'val_ce_losses_per': val_ce_losses_per,
                    'val_prec2': val_prec2,
                    'val': True
                }

                utils.save_checkpoint(model, optimizer, scheduler, arguments, save_path)

        arguments = {
                'iteration': i,
                'celoss': ce_losses,
                'regloss': regularization_losses,
                'prec1': prec1,
                'val': False
            }

        utils.save_checkpoint(model, optimizer, scheduler, arguments, PATH)

    # fig.text(0.5, 0.04, 'Common X-axis title', ha='center', va='center')
    # fig.text(0.04, 0.5, 'Common Y-axis title', ha='center', va='center', rotation='vertical')


def test(model, device, data_loader):
    model.eval()
    pred1 = utils.AverageMeter()
    pred2 = utils.AverageMeter()
    with torch.no_grad():
        for x, y in tqdm(iter(data_loader['test'])):
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                outputs_per = model(x + torch.rand_like(x))
            pred1.update(utils.accuracy(outputs, y).item(), x.shape[0])
            pred2.update(utils.accuracy(outputs_per, y).item(), x.shape[0])

    logging.info(f'\nTest Accuracy: {pred1.avg:.4f}%\n\t'
                 f'Test Accuracy Perturbed: {pred2.avg:.4f}%\n\t')





def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=48000, metavar='N',
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--decay', type=int, default=16800, metavar='N',
                        help='decay for scheduler (default: 10000)')
    parser.add_argument('--gamma', type=float, default=0.4, metavar='N',
                        help='gamma for scheduler (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument('--xi', type=float, default=1e-6, metavar='XI',
                        help='hyperparameter of VAT (default: 0.1)')
    parser.add_argument('--eps', type=float, default=6, metavar='EPS',
                        help='hyperparameter of VAT (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT (default: 1)')
    parser.add_argument('--xi_wrm', type=float, default=1.0, metavar='XI',
                        help='hyperparameter of wrm (default: 1.0)')
    parser.add_argument('--eps_wrm', type=float, default=6, metavar='EPS',
                        help='hyperparameter of wrm (default: 0.3)')
    parser.add_argument('--ip_wrm', type=int, default=1, metavar='IP',
                        help='hyperparameter of wrm (default: 1)')
    parser.add_argument('--workers', type=int, default=8, metavar='W',
                        help='number of CPU')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--exp-id', type=str, default="", metavar='EID',
                        help='experiment id')
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='GPU', help='dataset')
    parser.add_argument('--plot_shown', type=bool, default=False, metavar='plot', help='plot or not')
    parser.add_argument('--numplot', type=int, default=5, metavar='numplot', help='total number of plot')
    parser.add_argument('--valid_only', type=bool, default=False, metavar='valid_only', help='only validate')
    parser.add_argument('--method', type=str, default='vat', metavar='mathod', help='method for training')
    parser.add_argument('--plot_acc', type=bool, default=True, metavar='plot', help='plot  acc or not')
    return parser

def setup(device,args):



    # Create a Path object for the base directory
    base_dir = Path("checkpoint") / args.exp_id

    # Check if the experiment directory already exists
    if not base_dir.exists():
        # Create the directory, including any necessary parent directories
        base_dir.mkdir(parents=True, exist_ok=True)

    # The PATH variable is now a string representation of the base_dir Path object
    PATH = str(base_dir)
    # print(PATH)
    logging.info(f"Saving checkpoints to {PATH}")

    data_loader, traindata = data_utils.get_iters(
        dataset=args.dataset,
        root_path='.',
        l_batch_size=args.batch_size,
        ul_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        workers=args.workers
    )

    if args.dataset == 'CIFAR10':
        logging.info(f"Using CIFAR10 dataset")
        dim_input = 3
        dim_output = 10
    elif args.dataset == 'SVHN':
        logging.info(f"Using SVHN dataset")
        dim_input = 3
        dim_output = 10
    elif args.dataset == 'MNIST':
        logging.info(f"Using MNIST dataset")
        dim_input = 1
        dim_output = 10
    elif args.dataset == 'FashionMNIST':
        logging.info(f"Using FashionMNIST dataset")
        dim_input = 1
        dim_output = 10
    elif args.dataset == 'moon':
        logging.info(f"Using moon dataset")
        dim_input = 2
        dim_output = 2
    else:
        raise ValueError
    if args.dataset == 'moon':
        model = SimpleNet(dim_input,dim_output).to(device)
        args.iters = 1000
        args.plot_shown = True
        args.plot_iters = [0]
        for i in range(args.numplot):
            args.plot_iters.append(10 ** (i+1)-1)
            if 10 ** (i+1)-1 > args.iters:
                args.numplot = i+1
                break
    elif args.dataset == 'CIFAR10':
        model = ConvSmallCIFAR10(dim_input,dim_output).to(device)
    elif args.dataset == 'SVHN':
        model = ConvSmallSVHN(dim_input,dim_output).to(device)
    else:
        model = Net(dim_input,dim_output).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lambda_function = lambda update: 1 - args.gamma * max(0, update - (args.iters - args.decay)) / args.decay
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_function)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay, gamma=args.gamma)
    return model,optimizer,scheduler,PATH,data_loader,traindata


    # test(model, device, data_loader)

def valid_only(args, directory,device,model,data_loader):
    if args.method == 'vat':
        logging.info(f"Starting  VAT training for experiment {args.exp_id}")
    elif args.method == 'wrm':
        logging.info(f"Starting  WRM training for experiment {args.exp_id}")
    elif args.method == 'reg':
        logging.info(f"Starting  regular training for experiment {args.exp_id}")
    else:
        raise ValueError
    subdire = '/' + args.method + '_'
    path = directory + subdire + args.dataset + '.pth'

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model, device, data_loader)


def main(device,args):
    model, optimizer, scheduler,directory, data_loader, traindata = setup(device, args)
    try:
        utils.set_logger(directory + "/train_" + args.dataset + "_" + str(args.method)+ ".log")
    except Exception as e:
        print(f"Error setting up logger: {e}")

    if not args.valid_only:
        train(args, model, device, data_loader, optimizer, scheduler, directory)
    valid_only(args, directory, device, model, data_loader)

    if args.plot_shown:
        plot_shown(args, model, device, traindata, directory)

    if args.plot_acc:
        plot_acc(args, directory)


if __name__ == '__main__':
    parser = get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.exp_id == "":
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        args.exp_id = "".join(random.choice(chars) for _ in range(10))



    main(device,args)


#      VAT: (x_l, y_l)  (model) (x_ul)= y_ul vatloss  (y_ul)
#      AT:

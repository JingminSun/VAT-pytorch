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
from networks import Net, SimpleNet


def train(args, model, device, data_iterators, optimizer,scheduler, directory):
    if args.method == 'vat':
        logging.info(f"Starting  VAT training for experiment {args.exp_id}")
    elif args.method == 'wrm':
        logging.info(f"Starting  WRM training for experiment {args.exp_id}")
    elif args.method == 'reg':
        logging.info(f"Starting  regular training for experiment {args.exp_id}")
    else:
        raise ValueError
    subdire = '/' + args.method + '_'
    PATH = directory+subdire + args.dataset + '.pth'
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
                     f'CrossEntropyLoss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                     f'Regularization Loss {regularization_losses.val:.4f} ({regularization_losses.avg:.4f})\t'
                     f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})')

    model.train()
    epoch = 1
    for i in tqdm(range(iteration + 1 ,args.iters)):
        
        # reset
        if i % args.log_interval == 0:
            ce_losses = utils.AverageMeter()
            regularization_losses = utils.AverageMeter()
            prec1 = utils.AverageMeter()
        
        x_l, y_l = next(data_iterators['labeled'])
        x_ul, _ = next(data_iterators['unlabeled'])\

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_ul = x_ul.to(device)

        optimizer.zero_grad()
        cross_entropy = nn.CrossEntropyLoss()

        if args.method == 'vat':
            vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
            lds = vat_loss(model, x_ul)

            classification_loss = vat_loss(model, x_l, pred=y_l)
        elif args.method == 'wrm':
            wrmloss = WassersteinLoss(xi=args.xi_wrm, eps=args.eps_wrm, ip=args.ip_wrm)
            lds = wrmloss(model, x_ul, cross_entropy)
            classification_loss = wrmloss(model, x_l, cross_entropy, pred=y_l)
        else:
            lds = torch.norm(model(x_ul) - model(x_ul + 1e-8 * torch.randn_like(x_ul)), dim=1).mean()
            classification_loss = cross_entropy(model(x_l), y_l)

        output = model(x_l)

        loss = classification_loss + args.alpha * lds
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = utils.accuracy(output, y_l)
        ce_losses.update(classification_loss.item(), x_l.shape[0])
        regularization_losses.update(lds.item(), x_ul.shape[0])
        prec1.update(acc.item(), x_l.shape[0])

        if args.plot:
            if  i in args.plot_iters:


                os.makedirs(directory+ subdire+ '/iteration'+ str(i), exist_ok=True)

                save_path = directory+ subdire+ '/iteration'+ str(i) +'/vat_' + args.dataset + '.pth'

                arguments = {
                    'iteration': i,
                    'celoss': ce_losses,
                    'regloss': regularization_losses,
                    'prec1': prec1,
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
            logging.info(f'\nIteration: {i}\t'
                  f'CrossEntropyLoss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                  f'Regularization Loss {regularization_losses.val:.4f} ({regularization_losses.avg:.4f})\t'
                  f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})')

        arguments = {
            'iteration': i,
            'celoss': ce_losses,
            'regloss': regularization_losses,
            'prec1': prec1,
        }

        utils.save_checkpoint(model, optimizer, scheduler, arguments, PATH)



    # fig.text(0.5, 0.04, 'Common X-axis title', ha='center', va='center')
    # fig.text(0.04, 0.5, 'Common Y-axis title', ha='center', va='center', rotation='vertical')


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

    logging.info(f'\nTest Accuracy: {test_acc:.4f}%\n')


def plot(args,model, device, traindata,directory):
    def desaturate_color(color, factor):
        # Convert color to a NumPy array for easy manipulation
        color = np.array(color)
        # Blend the color with gray
        desaturated_color = color * factor + (1 - factor) * np.array([0.5, 0.5, 0.5])
        return desaturated_color

    # Original colors (in RGB)
    purple = (0.5, 0, 0.5)
    gray = (0.5, 0.5, 0.5)
    green = (0, 0.5, 0)

    # Desaturation factor (0: no change, 1: fully desaturated)
    factor = 0.9

    # Desaturate colors
    desaturated_purple = desaturate_color(purple, factor)
    desaturated_gray = desaturate_color(gray, factor)
    desaturated_green = desaturate_color(green, factor)
    colors = [desaturated_purple,desaturated_gray,desaturated_green]  # Colors from 0 to 1
    n_bins = 100  # Number of bins in the colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_bins)
    fig, axs = plt.subplots(2, args.numplot, figsize=(15, 6))
    x_la = traindata['x_labeled']
    y_la = traindata['y_labeled']
    index = np.random.randint(0, x_la[y_la == 0].shape[0], size = (3,))
    index = np.concatenate((index,np.random.randint(0, x_la[y_la == 1].shape[0], size = (3,))))

    if args.method == 'vat':
        reg_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
    elif args.method == 'wrm':
        reg_loss = WassersteinLoss(xi=args.xi_wrm, eps=args.eps_wrm, ip=args.ip_wrm)
    else:
        raise ValueError
    x_ula = traindata['x_unlabeled']
    for i in range(args.numplot):
        iteration = args.plot_iters[i]
        if iteration == 0:
            pred = np.ones((x_ula.shape[0],2)) * 0.5
        else:
            checkpoint = torch.load(directory+ '/iteration'+ str(iteration) +'/vat_' + args.dataset + '.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            pred = model(torch.from_numpy(x_ula).to(device)).detach().cpu().numpy()
        ldsa = np.zeros((x_ula.shape[0],))
        if i > 0:
            for ii in range(int(x_ula.shape[0] / args.batch_size)):
                ldsa[ii] = (reg_loss(model, torch.from_numpy(x_ula[ii:ii+1,:]).to(device)).detach().cpu().numpy())
        ldsa = np.abs(ldsa) / np.min(np.abs(ldsa) + 1e-8) + 1e-8
        ldsa = np.log(ldsa)
        sorted_indices = np.argsort(ldsa)  # Get the indices that would sort the array
        top_indices = sorted_indices[-100:]  # Get the indices of the top 10 values





        # print("max ldsa", np.max(ldsa), "min ldsa", np.min(ldsa), "std ldsa", np.std(ldsa))
        axs[0, i].scatter(x_ula[:, 0], x_ula[:, 1], c=pred[:, 1], cmap=cmap, alpha=0.5,vmin=0, vmax=1, label='Unlabeled data')
        axs[0, i].scatter(x_la[index, 0], x_la[index, 1], c=y_la[index], cmap=cmap, alpha=0.5, marker = 'D',vmin=0, vmax=1, edgecolors='black',
                          linewidths=1, s = 100, label='Labeled data')
        axs[1, i].scatter(x_ula[:, 0], x_ula[:, 1], c=ldsa, cmap='Blues', alpha=0.5)
        axs[1, i].scatter(x_ula[top_indices, 0], x_ula[top_indices, 1], c=ldsa[top_indices], cmap='Blues', alpha=0.5)
        axs[1, i].scatter(x_la[index, 0], x_la[index, 1], c=y_la[index], cmap=cmap, alpha=0.5, marker = 'D',vmin=0, vmax=1,edgecolors='black',
                          linewidths=1, s = 100)

        if i == 0:
            coltitle = ["Initial"]
        else:
            coltitle.append("After Iteration " + str(iteration + 1) + " of training" )
    axs[0, 0].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.1))


    for ax, col_title in zip(axs[0], coltitle):
        ax.set_title(col_title)

    row_titles = [r'$P(y|x,\theta)$', r'LDS$(x,\theta)$']
    for ax, row_title in zip(axs[:, 0], row_titles):
        ax.set_ylabel(row_title, rotation=90, size='large')

    plt.tight_layout()
    plt.savefig(directory +'/plot.png', dpi=300, bbox_inches='tight')
    plt.show()



def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=48000, metavar='N',
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--decay', type=int, default=1600, metavar='N',
                        help='decay for scheduler (default: 10000)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='N',
                        help='gamma for scheduler (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument('--xi', type=float, default=1e-6, metavar='XI',
                        help='hyperparameter of VAT (default: 0.1)')
    parser.add_argument('--eps', type=float, default=8, metavar='EPS',
                        help='hyperparameter of VAT (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT (default: 1)')
    parser.add_argument('--xi_wrm', type=float, default=1.0, metavar='XI',
                        help='hyperparameter of wrm (default: 1.0)')
    parser.add_argument('--eps_wrm', type=float, default=8, metavar='EPS',
                        help='hyperparameter of wrm (default: 0.3)')
    parser.add_argument('--ip_wrm', type=int, default=1, metavar='IP',
                        help='hyperparameter of wrm (default: 1)')
    parser.add_argument('--workers', type=int, default=8, metavar='W',
                        help='number of CPU')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--exp-id', type=str, default="", metavar='EID',
                        help='experiment id')
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='GPU', help='dataset')
    parser.add_argument('--plot', type=bool, default=False, metavar='plot', help='plot or not')
    parser.add_argument('--numplot', type=int, default=5, metavar='numplot', help='total number of plot')
    parser.add_argument('--valid_only', type=bool, default=False, metavar='valid_only', help='only validate')
    parser.add_argument('--method', type=str, default='vat', metavar='mathod', help='method for training')
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

    data_iterators, traindata = data_utils.get_iters(
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
    elif args.dataset == 'CIFAR100':
        logging.info(f"Using CIFAR100 dataset")
        dim_input = 3
        dim_output = 100
    elif args.dataset == 'MNIST':
        logging.info(f"Using MNIST dataset")
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
        args.plot = True
        args.plot_iters = [0]
        for i in range(args.numplot):
            args.plot_iters.append(10 ** (i+1)-1)
            if 10 ** (i+1)-1 > args.iters:
                args.numplot = i+1
                break
    else:
        model = Net(dim_input,dim_output).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lambda_function = lambda update: 1 - max(0, update - (args.iters - args.decay)) / args.decay
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_function)
    return model,optimizer,scheduler,PATH,data_iterators,traindata


    # test(model, device, data_iterators)

def valid_only(args, directory,device,model,data_iterators):
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
    test(model, device, data_iterators)


def main(device,args):
    model, optimizer, scheduler,directory, data_iterators, traindata = setup(device, args)
    try:
        utils.set_logger(directory + "/train_" + args.dataset + "_" + str(args.method)+ ".log")
    except Exception as e:
        print(f"Error setting up logger: {e}")

    if not args.valid_only:
        train(args, model, device, data_iterators, optimizer, scheduler, directory)
    valid_only(args, directory, device, model, data_iterators)

    if args.plot:
        plot(args, model, device, traindata, directory)


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

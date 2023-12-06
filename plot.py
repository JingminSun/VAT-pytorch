import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import argparse
from pathlib import Path
def plot_acc(args,directory):
    subdire = '/' + args.method + '_'
    PATH = directory + subdire + args.dataset + '.pth'
    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH)
        total_iters = checkpoint['iteration']
    else:
        raise ValueError

    xaxis = np.arange(0,total_iters+1,args.log_interval)
    yaxis = np.zeros((7,xaxis.shape[0]))
    for i in range(xaxis.shape[0]):
        checkpoint = torch.load(directory + subdire+  args.dataset + '/iteration' + str(xaxis[i]) + '/'+ str(args.method) + '.pth')
        yaxis[0,i] = checkpoint['celoss'].val
        yaxis[1,i] = checkpoint['regloss'].val
        yaxis[2,i] = checkpoint['prec1'].val
        yaxis[3,i] = checkpoint['val_ce_losses'].val
        yaxis[4,i] = checkpoint['val_prec1'].val
        yaxis[5,i] = checkpoint['val_ce_losses_per'].val
        yaxis[6,i] = checkpoint['val_prec2'].val

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].plot(xaxis,yaxis[0,:],label='Classification Loss')
    axs[0].plot(xaxis,yaxis[3,:],label='Validation Loss')
    axs[0].plot(xaxis,yaxis[5,:],label='Validation Loss Perturbed')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss in Training and Validation')

    if args.method != 'reg':
        print("regularization loss")
        ax2 = axs[0].twinx()
        ax2.plot(xaxis,yaxis[1,:],label='Regularization Loss',color='red')
        ax2.legend(loc='upper left')
        ax2.set_ylabel('Regularization Loss')

    axs[1].plot(xaxis,yaxis[2,:],label='Training Accuracy')
    axs[1].plot(xaxis,yaxis[4,:],label='Validation Accuracy')
    axs[1].plot(xaxis,yaxis[6,:],label='Validation Accuracy Perturbed')
    axs[1].legend(loc='lower right')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy in Training and Validation')
    axs[1].yaxis.set_label_position("right")


    fig.suptitle(str(args.method) + ' on ' + args.dataset)
    plt.savefig(directory + subdire + args.dataset +'/plot_acc_' + args.dataset + str(args.method) + '.png', dpi=300, bbox_inches='tight')
    # print(directory + subdire + args.dataset +'/plot_acc.png')
    # plt.show()




def plot_shown(args,model, device, traindata,directory):
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



def plot_all(args, directory):
    methods = ['vat', 'wrm', 'reg']
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))  # Change to a 2x3 grid

    for m, method in enumerate(methods):
        args.method = method
        subdire = '/' + args.method + '_'
        PATH = directory + subdire + args.dataset + '.pth'

        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            total_iters = checkpoint['iteration']
        else:
            raise ValueError("Checkpoint file not found.")

        xaxis = np.arange(0, total_iters + 1, args.log_interval)
        yaxis = np.zeros((7, xaxis.shape[0]))

        for i in range(xaxis.shape[0]):
            checkpoint = torch.load(directory + subdire + args.dataset + '/iteration' + str(xaxis[i]) + '/' + str(args.method) + '.pth')
            yaxis[0, i] = checkpoint['celoss'].val
            yaxis[1, i] = checkpoint['regloss'].val
            yaxis[2, i] = checkpoint['prec1'].val
            yaxis[3, i] = checkpoint['val_ce_losses'].val
            yaxis[4, i] = checkpoint['val_prec1'].val
            yaxis[5, i] = checkpoint['val_ce_losses_per'].val
            yaxis[6, i] = checkpoint['val_prec2'].val

        col = m % 3
        row = 0

        axs[row, col].plot(xaxis, yaxis[0, :], label='Classification Loss')
        axs[row, col].plot(xaxis, yaxis[3, :], label='Validation Loss')
        axs[row, col].plot(xaxis, yaxis[5, :], label='Validation Loss Perturbed')
        if col == 0:
            axs[row, col].legend(loc='upper right')

        if args.method != 'reg':
            ax2 = axs[row, col].twinx()
            ax2.plot(xaxis, yaxis[1, :], label='Regularization Loss', color='red')
            if col == 0:
                ax2.legend(loc='upper left')
                ax2.set_ylabel('Regularization Loss')

        axs[1, col].plot(xaxis, yaxis[2, :], label='Training Accuracy')
        axs[1, col].plot(xaxis, yaxis[4, :], label='Validation Accuracy')
        axs[1, col].plot(xaxis, yaxis[6, :], label='Validation Accuracy Perturbed')
        if col == 0:
            axs[1, col].legend(loc='lower right')

    for col, col_title in enumerate(['VAT', 'WRM', 'REG']):
        axs[0, col].set_title(col_title)

    for ax, row_title in zip(axs[:, 0], ['Loss', 'Accuracy']):
        ax.set_ylabel(row_title, rotation=90, size='large')

    plt.tight_layout()
    plt.savefig(directory + args.dataset + '.png')
    print('Plot saved to ' + directory + args.dataset + '.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument('--dataset', default='FashionMNIST', type=str, help='dataset = [CIFAR10, SVHN, MNIST]')
    parser.add_argument('--method', default='vat', type=str, help='vat, wrm, reg')
    parser.add_argument('--log_interval', default=100, type=int, help='log interval')
    parser.add_argument('--exp_id', default='allfinal', type=str, help='experiment id')
    args = parser.parse_args()
    directory = str(Path("checkpoint") / args.exp_id)
    plot_all(args, directory)

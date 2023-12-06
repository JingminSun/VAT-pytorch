from collections import OrderedDict
import logging
import logzero
from logzero import logger
from pathlib import Path
from tensorboardX import SummaryWriter
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


def save_checkpoint(model, optimizer, scheduler,arguments, path):
    if arguments['val']:
        torch.save({
            'epoch': 1,
            'iteration': arguments['iteration'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'celoss': arguments['celoss'],
            'regloss': arguments['regloss'],
            'prec1': arguments['prec1'],
            'val_ce_losses': arguments['val_ce_losses'],
            'val_prec1': arguments['val_prec1'],
            'val_ce_losses_per': arguments['val_ce_losses_per'],
            'val_prec2': arguments['val_prec2'],
        }, path)
    else:
        torch.save({
            'epoch': 1,
            'iteration': arguments['iteration'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'celoss': arguments['celoss'],
            'regloss': arguments['regloss'],
            'prec1': arguments['prec1'],
        }, path)


def set_logger(log_file_path, log_level=logging.INFO, tf_board_path=None):
    # Create directory for log file if it doesn't exist
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')

    # Create and add file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create and add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # TensorBoard Writer Setup
    writer = None
    if tf_board_path is not None:
        Path(tf_board_path).parent.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tf_board_path)

    return writer
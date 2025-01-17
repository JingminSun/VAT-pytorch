import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, pred=None):
        with torch.no_grad():
            predd = model(x)
            num_classes = predd.shape[1]
            if pred is None:
                pred = F.softmax(predd, dim=1)
            else:
                pred = F.one_hot(pred, num_classes=num_classes).float()
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')
        return lds


#      vat distance (model(x), model(xperturb)); at distance(y, model(xperturb))


class WassersteinLoss(nn.Module):

    def __init__(self, xi = 0.3 ,eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of Wasserstein (default: 0.3)
        :param eps: hyperparameter of Wasserstein (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(WassersteinLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, model_loss, pred=None):
        with torch.no_grad():
            if pred is None:
                pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        x_adv = (x + self.eps * d).requires_grad_()
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                loss = model_loss(model(x_adv), pred)
                grad = torch.autograd.grad(self.xi * loss, x_adv)[0]
                grad2 = torch.autograd.grad(torch.norm(x_adv-x, p=2), x_adv)[0]
                grad -= grad2
                grad = _l2_normalize(grad)
                x_adv = x_adv + self.eps *  grad.detach()
                model.zero_grad()

            # calc LDS
            pred_hat = model(x_adv)
            lds = model_loss(pred_hat, pred)
        return lds


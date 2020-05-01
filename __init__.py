import itertools as it
import warnings

import torch.optim as optim
from torch.optim.optimizer import Optimizer

from .AdaMod import *
from .DeepMemory import *
from .diffGrad import *
from .diffMod import *
from .RAdam import *
from .Ranger import *

warnings.filterwarnings("once")


def get_optimizer(optimizer: str = 'Adam',
                  lookahead: bool = False,
                  model=None,
                  separate_head: bool = True,
                  lr: float = 1e-3,
                  lr_e: float = 1e-3):
    """
    # https://github.com/lonePatient/lookahead_pytorch/blob/master/run.py
    :param optimizer:
    :param lookahead:
    :param model:
    :param separate_head:
    :param lr:
    :param lr_e:
    :return:
    """

    if separate_head:
        params = [
            {'params': model.encoder.parameters(), 'lr': lr_e},
        ]
        for key in model.heads:
            params.append({'params': model.heads[key].parameters(), 'lr': lr})
    else:
        params = [{'params': model.parameters(), 'lr': lr}]

    if optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=lr)
    if optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=lr)
    elif optimizer == 'RAdam':
        optimizer = RAdam(params, lr=lr)
    elif optimizer == 'Ralamb':
        optimizer = Ralamb(params, lr=lr)
    elif optimizer == "Ranger":
        optimizer = Ranger(params, lr=lr)
    elif optimizer == "DeepMemory":
        optimizer = DeepMemory(params, lr=lr)
    elif optimizer == 'diffGrad':
        optimizer = diffGrad(params, lr=lr)
    elif optimizer == 'diffRGrad':
        optimizer = diffRGrad(params, lr=lr)
    else:
        raise ValueError('unknown base optimizer type')

    if lookahead:
        optimizer = Lookahead(base_optimizer=optimizer, k=5, alpha=0.5)

    return optimizer


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                             for group in self.param_groups]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p, q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)
        return loss

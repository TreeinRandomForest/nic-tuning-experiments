import sys
sys.path.append('../bayesopt')

import read_agg_data
import torch
import torch.nn as nn
import torch.autograd as auto
import torch.optim as optim
import math

import numpy as np
import matplotlib.pylab as plt

import pdb

plt.ion()

def inference(n_iter, lr, print_freq=10):
    x = torch.rand(1, requires_grad=True)

    func_tensor = torch.Tensor([1] * 10)
    # min_tensor = torch.Tensor([-1] * 10)

    # criterion = nn.MSELoss()
    optimizer = optim.Adam([x], lr=lr)

    for _ in range(n_iter):
        # func = 5 * x
        func = torch.Tensor(math.sin(x))
        pdb.set_trace()
        # loss = criterion(min_tensor, func)
        
        optimizer.zero_grad()
        func.backward()
        optimizer.step()

        if _ % print_freq == 0:
            print(x.item())

    return x

def run(n_iter=2000, 
        lr=1e-1):

    pred = inference(n_iter, lr, print_freq=500)

    return pred
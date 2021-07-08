import torch
import torch.nn as nn
import torch.autograd as auto
import torch.optim as optim

import numpy as np
import matplotlib.pylab as plt

plt.ion()

def synthetic(A, B, p, N):
    d = []
    for i in range(N):
        x = np.random.uniform()
        y = A + B*(x**p)
        d.append([x,y])

    return torch.tensor(d)

def inference(d, n_iter, lr, print_freq=10):
    A = torch.tensor([0.], requires_grad=True)
    B = torch.tensor([0.], requires_grad=True)
    p = torch.tensor([0.], requires_grad=True)
    
    x = d[:,0]
    y = d[:,1]
    print(type(d))
    print(type(x))

    criterion = nn.MSELoss()
    optimizer = optim.Adam([A, B, p], lr=lr)

    for _ in range(n_iter):
        pred = A + B*(x**p)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _ % print_freq == 0:
            print(A, B, p, loss)

    return A + B*(x**p)

    
def run(n_iter=2000):
    A = 2
    B = 1
    p = 3
    N = 100

    d = synthetic(A, B, p, N)
    plt.plot(d[:,0], d[:,1], 'p')

    pred = inference(d, n_iter, 1e-1, print_freq=100)
    plt.plot(d[:,0], pred.detach().numpy(), 'p')
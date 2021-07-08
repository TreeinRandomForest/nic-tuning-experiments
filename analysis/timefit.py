import sys
sys.path.append('../bayesopt')

import read_agg_data
import torch
import torch.nn as nn
import torch.autograd as auto
import torch.optim as optim

import numpy as np
import matplotlib.pylab as plt

import pdb

plt.ion()

def inference(d, n_iter, lr, workload, sys, print_freq=10):
    #starts randomly
    #max_time = torch.tensor(torch.Tensor(1,1).uniform_(10, 500), requires_grad=True)
    #alpha = torch.tensor(torch.Tensor(1,1).uniform_(-2, 2), requires_grad=True)
    
    log_max_time = torch.rand(1, requires_grad=True)
    alpha = torch.rand(1, requires_grad=True)

    t_latency = d[:,0]
    itr = d[:,1]
    dvfs = d[:,2]

    criterion = nn.MSELoss()
    optimizer = optim.Adam([log_max_time, alpha], lr=lr)

    print(f'---------------{workload} {sys} lr = {lr}---------------')

    for _ in range(n_iter):
        max_time = torch.exp(log_max_time)
        pred = itr + max_time/dvfs**(1+alpha)
        loss = criterion(pred, t_latency)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _ % print_freq == 0:
            print(max_time, alpha, loss)

    return pred
    
def run(n_iter=2000, lr=1e-1):
    #read linux_mcd.csv
    for workload in ['mcd']:
        df_comb, _, _ = read_agg_data.start_analysis(workload) #DATA
        df_comb['dvfs'] = df_comb['dvfs'].apply(lambda x: int(x, base=16))
        df_comb = df_comb[(df_comb['itr']!=1) | (df_comb['dvfs']!=65535)] #filter out linux dynamic
        df_comb['dvfs'] = df_comb['dvfs'].astype(float) / df_comb['dvfs'].min()
        df_comb = df_comb[df_comb['QPS'] == 400000]

        for sys in ['ebbrt_tuned']:
            df = df_comb[(df_comb['sys']==sys)].copy()
            df = df[['read_99th_mean','itr', 'dvfs']]
            d = df.values
            d = torch.tensor(d)

            pred = inference(d, n_iter, lr, workload, sys, print_freq=500)
            df[f'prediction lr={lr}'] = pred.detach().numpy()

            fig, ax = plt.subplots()
            plt.title(f'workload={workload} system={sys} lr={lr} QPS=400000')
            plt.xlabel(u"predictions")
            plt.ylabel(u"actual values")

            scatter = ax.scatter(pred.detach().numpy(), d[:,0], marker = 'o', s = d[:,1], c = d[:,2], alpha=0.3)

            legend1 = ax.legend(*scatter.legend_elements(),loc="upper left", title="dvfs")
            ax.add_artist(legend1)
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
            legend2 = plt.legend(handles, labels, loc="lower right", title="itr")
            ax.add_artist(legend2)

            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_ylim(0, ax.get_ylim()[1])
            ax.grid()

            ax.plot(np.arange(0, int(ax.get_xlim()[1])), np.arange(0, int(ax.get_xlim()[1])))

            plt.savefig(f'plots/timefit/{workload}_{sys}_{lr}.png')
            plt.close()
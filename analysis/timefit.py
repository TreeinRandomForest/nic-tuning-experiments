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
    itr_suppress = torch.rand(1, requires_grad=True)

    t_latency = d[:,0]
    itr = d[:,1]
    dvfs = d[:,2]

    criterion = nn.MSELoss()
    optimizer = optim.Adam([log_max_time, alpha, itr_suppress], lr=lr)

    for _ in range(n_iter):
        max_time = torch.exp(log_max_time)
        pred = itr_suppress*itr + max_time/dvfs**(1+alpha)
        loss = criterion(pred, t_latency)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _ % print_freq == 0:
            if _==0:
                print(f'{"max_time":^10} {"alpha":^10} {"itr_suppress":^10} {"loss":^10}')
            
            print(f'{max_time.item():^10.3f} {alpha.item():^10.3f} {itr_suppress.item():^10.3f} {loss.item():^10.3f}')

    return pred, {'max_time': max_time.item(), 'alpha': alpha.item(), 'itr_suppress': itr_suppress.item(), 'sqrt_loss': np.sqrt(loss.item())}
    
def run_all(n_iter=1000, lr=1e-2, qps=200000):
    #for qps in [200000,400000, 60000]:
    data = []
    for sys in ['linux_tuned', 'ebbrt_tuned']:
        for target_col in [f'read_{i}th_mean' for i in [5, 10, 50, 90, 99]]:
            d = run(n_iter=n_iter, lr=lr, sys=sys, qps=qps, target_col=target_col)
            data.append(d)

    return data


def run(n_iter=2000, 
        lr=1e-1, 
        target_col='read_99th_mean',
        sys='ebbrt_tuned',
        qps=400000):

    #read linux_mcd.csv
    for workload in ['mcd']:
        #read raw data: TODO check all preprocessing is correct
        df_comb, _, _ = read_agg_data.start_analysis(workload)
        df_comb['dvfs'] = df_comb['dvfs'].apply(lambda x: int(x, base=16))
        df_comb = df_comb[(df_comb['itr']!=1) | (df_comb['dvfs']!=65535)] #filter out linux dynamic
        df_comb['dvfs'] = df_comb['dvfs'].astype(float) / df_comb['dvfs'].min()
        df_comb = df_comb[df_comb['QPS'] == qps]

        #filter to system
        df = df_comb[(df_comb['sys']==sys)].copy()
        df = df[[target_col,'itr', 'dvfs']]
        d = df.values
        d = torch.tensor(d)

        #fitting
        print(f'----------{workload} {sys} QPS={qps} {target_col}-------------')
        if df.shape==0:
            raise ValueError('Empty Dataframe')
        pred, params = inference(d, n_iter, lr, workload, sys, print_freq=500)
        df[f'prediction lr={lr}'] = pred.detach().numpy()

        #plotting
        fig, ax = plt.subplots()
        plt.title(f"{workload} {sys} {qps} {target_col}\n maxtime={params['max_time']:.2f} alpha={params['alpha']:.2f} itr_suppress={params['itr_suppress']:.2f} loss={params['sqrt_loss']:.2f}")
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

        plt.savefig(f'plots/timefit/{workload}_{sys}_{target_col}_{qps}_{lr}.png')
        plt.close()

        return {**params, 'sys': sys, 'workload': workload, 'qps': qps, 'target_col': target_col}
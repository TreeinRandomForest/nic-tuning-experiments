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
    # p_busy_min = 20
    p_static = {
        'c1':1.5, 
        'c3':0.5,
        'c4':0.25,
        'c7':0,
        'busy': 10
    }
    chosen_sleep = 'c7'

    # p_q = p_static[chosen_sleep]
    # p_detect = p_static[chosen_sleep]

    #starts randomly
    max_time = torch.tensor(torch.Tensor(1,1).uniform_(10, 500), requires_grad=True)
    alpha = torch.tensor(torch.Tensor(1,1).uniform_(-2, 2), requires_grad=True)
    beta = torch.tensor(torch.Tensor(1,1).uniform_(-2, 2), requires_grad=True)
    p_static_busy = torch.tensor(torch.Tensor(1,1).uniform_(0, 35), requires_grad=True)
    p_detect = torch.tensor(torch.Tensor(1,1).uniform_(0, 35), requires_grad=True)
    p_q = torch.tensor(torch.Tensor(1,1).uniform_(0, 35), requires_grad=True)
    p_busy_min = torch.tensor(torch.Tensor(1,1).uniform_(0, 35), requires_grad=True)
    itr_suppress = torch.rand(1, requires_grad=True)
    
    qps = d[:,3]
    energy = d[:,0]/(qps*20)
    itr = d[:,1]
    dvfs = d[:,2]
    time = d[:,4]
    interarrival_time = 1/qps*10**6

    current_loss_time = -100
    fixed_max_time = -100
    fixed_alpha = -100

    criterion = nn.MSELoss()
    optimizer_time = optim.Adam([max_time, alpha], lr=lr)
    optimizer_energy = optim.Adam([max_time, alpha, beta, p_detect, p_q], lr=lr)
    # optimizer = optim.Adam([max_time, alpha, beta, p_detect, p_q], lr=lr)

    print(f'---------------FOR TIME LOSS {workload} {sys} lr = {lr}---------------')

    for _ in range(n_iter):
        p_busy = (p_static_busy + p_busy_min*dvfs**(2+beta))
        t_busy = (max_time / dvfs**(1+alpha))
        pred_time = itr_suppress*itr + t_busy
        loss_time = criterion(pred_time/time, torch.ones((1,pred_time.shape[1])).double())
        # loss = loss_energy + loss_time

        optimizer_time.zero_grad()
        loss_time.backward(retain_graph=True)
        optimizer_time.step()

        if(current_loss_time == -100):
            current_loss_time = loss_time.item()
        else:
            if(current_loss_time >= loss_time.item()):
                current_loss_time = loss_time.item()
                fixed_max_time = max_time.item()
                fixed_alpha = alpha.item()

        if _ % print_freq == 0:
            print(max_time.item(), alpha.item(), itr_suppress.item(), loss_time.item())

    print(f'---------------FOR ENERGY LOSS {workload} {sys} lr = {lr} max_time = {fixed_max_time} alpha = {fixed_alpha}---------------')

    for _ in range(n_iter):
        t_busy_energy = (fixed_max_time / dvfs**(1+fixed_alpha))
        t_q_energy = (interarrival_time - itr - t_busy_energy)
        pred_energy = (p_detect * itr_suppress*itr) + (p_busy * t_busy_energy) + (p_q * t_q_energy)
        loss_energy = criterion(pred_energy/energy, torch.ones((1,pred_energy.shape[1])).double())

        optimizer_energy.zero_grad()
        loss_energy.backward(retain_graph=True)
        optimizer_energy.step()

        if _ % print_freq == 0:
            print(max_time.item(), alpha.item(), beta.item(), p_detect.item(), p_q.item(), itr_suppress.item(), loss_energy.item())

    return pred_energy, pred_time
    
def run(n_iter=2000, lr = 1e-2):
    #read linux_mcd.csv
    for workload in ['mcd']:
        df_comb, _, _ = read_agg_data.start_analysis(workload) #DATA
        df_comb['dvfs'] = df_comb['dvfs'].apply(lambda x: int(x, base=16))
        df_comb = df_comb[(df_comb['itr']!=1) | (df_comb['dvfs']!=65535)] #filter out linux dynamic
        df_comb['dvfs'] = df_comb['dvfs'].astype(float) / df_comb['dvfs'].min()
        df_comb = df_comb[df_comb['QPS'] == 400000]

        for sys in ['ebbrt_tuned']:
            df = df_comb[(df_comb['sys']==sys)].copy()
            df = df[['joules_mean','itr', 'dvfs', 'QPS', 'read_99th_mean']]
            d = df.values
            d = torch.tensor(d)
            plt.plot(d[:,0], d[:,1], 'p')

            for lr in [lr]:
                pred_energy, pred_time = inference(d, n_iter, lr, workload, sys, print_freq=1000)
                df[f'pre_energy lr={lr}'] = pred_energy.view(245, 1).detach().numpy()
                df[f'pre_time lr={lr}'] = pred_time.view(245, 1).detach().numpy()
                
                for pred_name in ['energy', 'time']:
                    if pred_name == 'energy':
                        pred = pred_energy
                        qps = d[:,3]
                        yvalue = d[:,0]/(qps*20)
                    else:
                        pred = pred_time
                        yvalue = d[:,4]
                    fig, ax = plt.subplots()
                    plt.title(f'predict:{pred_name} workload={workload} system={sys} lr={lr} QPS=400000')
                    plt.xlabel(u"predictions")
                    plt.ylabel(pred_name)
                    scatter = ax.scatter(pred.detach().numpy()[0], yvalue, marker = 'o', s = d[:,1], c = d[:,2], alpha=0.3)
                    legend1 = ax.legend(*scatter.legend_elements(),loc="upper left", title="dvfs")
                    ax.add_artist(legend1)
                    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
                    legend2 = plt.legend(handles, labels, loc="lower right", title="itr")
                    ax.add_artist(legend2)
                    plt.savefig(f'plots/energy_time_fit/randomp_{pred_name}_{workload}_{sys}_{lr}.png')
                    plt.close()
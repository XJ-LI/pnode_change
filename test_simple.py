# import 
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import copy
import random
import argparse

sys.path.append('/content/petsc/arch-linux-opt/lib')

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--niters', type=int, default=2000)
args, unknown = parser.parse_known_args()

import petsc4py
sys.argv = [sys.argv[0]] + unknown
petsc4py.init(sys.argv)
from petsc4py import PETSc
from pnode import petsc_adjoint

import scipy.io
data = scipy.io.loadmat('/content/petsc/pnode/database_double_pendulum.mat')  # may need to change the dir
print(data.keys())
L = data['L']
M = data['M']
z = data['Z']
dt = data['dt']
t0 = data['t_vec'][0,0]
tN = data['t_vec'][0,-1]

# try a generic neural ode setting
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ODEfunc(nn.Module):

    def __init__(self, layers):
        super(ODEfunc, self).__init__()
        self.act = nn.ELU(inplace=False)
        self.net_len = len(layers)
        # layers need to have correct input and ouput dim
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(self.net_len - 1)])

    def forward(self, t, x):
        for i in range(self.net_len - 2):
            x = self.act(self.linears[i](x))
        x = self.linears[-1](x)  # no activation for last layer

        return x
        
        

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print('device: ', device)

## Hyperparameters
dim = 10
nhidden = 50
layers = [dim, nhidden, nhidden, nhidden, dim]
lr = 0.005
step_size = 0.6

z = torch.tensor(z).to(device).double()
z0 = z[:,:,0]
z0_size = z0[0,:]
integration_times = torch.linspace(t0, tN, z.shape[2]).to(device)


func = ODEfunc(layers).to(device).double()
optimizer = optim.AdamW(func.parameters(), lr=lr)

ode = petsc_adjoint.ODEPetsc()
ode.setupTS(z0_size, func, step_size=step_size, implicit_form=True, method='cn')
loss_func = nn.MSELoss(reduction='mean')

print('number of parameters: ', count_parameters(func))
loss_arr = np.zeros(args.niters)
start_time = time.time()

min_loss = 1.0
for itr in range(1, args.niters + 1):
    iter_start_time = time.time()
    optimizer.zero_grad()
    idx = random.randrange(0, z0.shape[0]-1)

    rand_z0 = z0[idx,:]
    rand_z = z[idx,:,:]

    pred_z = ode.odeint_adjoint(rand_z0, integration_times)
    pred_z = pred_z.permute(1,0)
    loss = loss_func(pred_z, rand_z)
    loss.backward()
    optimizer.step()
    iter_end_time = time.time()

    loss_arr[itr-1] = loss
    if itr % 20 == 0:
        print('Iter: {}, running MSE: {:.4f}, time per iteration: {:.4f}'.format(itr, loss, iter_end_time-iter_start_time))
    if loss  < min_loss:
        min_loss = loss
        torch.save(func.state_dict(), 'node_model.pth')


end_time = time.time()
print('total time: ', end_time - start_time)

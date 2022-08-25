########################################
# Example of usage:
#   python3 Pendulum_DAE.py --double_prec --implicit_form -ts_trajectory_type memory
#######################################
# problem DAE of index 1
import os
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/content/petsc/arch-linux-opt/lib')  # edit path to petsc location
import copy
import scipy
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=22)
matplotlib.rc('axes', titlesize=22)
matplotlib.use('Agg')

parser = argparse.ArgumentParser('Chemical Akzo Nobel DAE')
parser.add_argument('--pnode_method', type=str, choices=['beuler', 'cn'], default='cn')
parser.add_argument('--data_size', type=int, default=5)
parser.add_argument('--steps_per_data_point', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--activation', type=str, choices=['gelu', 'tanh'], default='gelu')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--implicit_form', action='store_true')
parser.add_argument('--use_dlpack', action='store_true')
parser.add_argument('--double_prec', action='store_true')
parser.add_argument('--train_dir', type=str, metavar='PATH', default='./train_results' )
parser.add_argument('--petsc_ts_adapt', action='store_true')
args, unknown = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# device = 'cpu'
# data generation
theta_0=np.pi/2
theta_dot0=0.
m=1
r0=1
g=9.81
x0 =  r0*np.sin(theta_0)
y0 = -r0*np.cos(theta_0)
vx0 = r0*theta_dot0*np.cos(theta_0)
vy0 = r0*theta_dot0*np.sin(theta_0)
lbda_0 = (m*r0*theta_dot0**2 +  m*g*np.cos(theta_0))/r0 # equilibrium along the rod's axis
true_y0 = np.array([x0,y0,vx0,vy0,lbda_0])
true_y0 = torch.from_numpy(true_y0).double()
t = torch.linspace(0., 0.5, args.data_size+1, dtype=torch.float64)

if not args.petsc_ts_adapt:
    unknown.append('-ts_adapt_type')
    unknown.append('none') # disable adaptor in PETSc
    t_traj = torch.linspace(start=0, end=0.5, steps=args.data_size+1+(args.data_size)*(args.steps_per_data_point-1))
    step_size = t_traj[1] - t_traj[0]
else:
    step_size = 1e-2

class Lambda(nn.Module):
    def forward(self, t, y):
        f = torch.clone(y)
        f[0] = y[2]
        f[1] = y[3]
        f[2] = -y[0]*y[4]
        f[3] = -y[1]*y[4] - g
        f[4] = y[4]*(y[0]**2 + y[1]**2) + g*y[1] - (y[2]**2 + y[3]**2)
        return f

M = torch.eye(5); M[-1,-1] = 0.

import petsc4py
sys.argv = [sys.argv[0]] + unknown
petsc4py.init(sys.argv)
from pnode import petsc_adjoint_mod

ode0 = petsc_adjoint_mod.ODEPetsc()
ode0.setupTS(true_y0, Lambda(), step_size=step_size, enable_adjoint=False, use_dlpack=False, implicit_form=True, method='cn', M=M)
true_y = ode0.odeint(true_y0, t)

if not args.double_prec:
    true_y = true_y.float()
    t = t.float()
true_y = true_y.to(device)
t = t.to(device)
true_y0 = true_y[0]

# delete the last dimension? algebraic variable should not enter data

def get_batch():
    s = torch.from_numpy(np.sort(np.random.choice(np.arange(1, args.data_size+1, dtype=np.int64), args.batch_size, replace=False)))
    s = torch.cat((torch.tensor([0]), s))
    batch_t = t[s]
    batch_y = true_y[s]
    return batch_t, batch_y

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    makedirs(os.path.join(args.train_dir, 'png'))
    # import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 12), facecolor='white')
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    plt.show(block=False)

#marker_style1 = dict(marker='o', markersize=8, mfc='None')
#marker_style2 = dict(marker='x', markersize=8, mfc='None')
marker_style1 = {}
marker_style2 = {}
lw = 2.5

def visualize(t, true_y, pred_y, odefunc, itr, name):
    if args.viz:
        ax1.cla()
        ax1.set_xlabel('t')
        ax1.set_ylabel(r'$x$')
        ax1.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0], 'g-', linewidth=lw, label='Ground Truth', **marker_style1)
        ax1.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0], 'b--', linewidth=lw, label=name, **marker_style2)
        ax1.legend()

        ax2.cla()
        ax2.set_xlabel('t')
        ax2.set_ylabel(r'$y$')
        ax2.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 1], 'g-', linewidth=lw, **marker_style1)
        ax2.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 1], 'b--', linewidth=lw, **marker_style2)

        ax3.cla()
        ax3.set_xlabel('t')
        ax3.set_ylabel(r'$v_x$')
        ax3.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 2], 'g-', linewidth=lw, **marker_style1)
        ax3.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 2], 'b--', linewidth=lw, **marker_style2)

        ax4.cla()
        ax4.set_xlabel('t')
        ax4.set_ylabel(r'$v_y$')
        ax4.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 3], 'g-', linewidth=lw, **marker_style1)
        ax4.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 3], 'b--', linewidth=lw, **marker_style2)

        ax5.cla()
        ax5.set_xlabel('t')
        ax5.set_ylabel(r'$Lambda$')
        ax5.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 4], 'g-', linewidth=lw, **marker_style1)
        ax5.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 4], 'b--', linewidth=lw, **marker_style2)

        fig.tight_layout()
        plt.savefig(os.path.join(args.train_dir, 'png')+'/{:03d}'.format(itr)+name)
        plt.draw()
        plt.pause(0.001)

# network setup
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        if args.activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Tanh()
        if args.double_prec:
            self.net = nn.Sequential(
                nn.Linear(5, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 4, bias=False).double(),
            ).to(device)
            self.net2 = nn.Sequential(
                nn.Linear(5, 10, bias=False).double(),
                self.act.double(),
                nn.Linear(10, 5, bias=False).double(),
                self.act.double(),
                nn.Linear(5, 1, bias=False).double(),
            ).to(device)
        else:
            self.net = nn.Sequential(
                nn.Linear(5, 10, bias=False),
                self.act,
                nn.Linear(10, 10, bias=False),
                self.act,
                nn.Linear(10, 10, bias=False),
                self.act,
                nn.Linear(10, 10, bias=False),
                self.act,
                nn.Linear(10, 5, bias=False),
            ).to(device)
            self.net2 = nn.Sequential(
                nn.Linear(5, 10, bias=False),
                self.act,
                nn.Linear(10, 5, bias=False),
                self.act,
                nn.Linear(5, 1, bias=False),
            ).to(device)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
        for m in self.net2.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        f = torch.clone(y)
        f[:-1] = self.net(y)
        # f[-1] = self.net2(y)
        f[-1] = y[4]*(y[0]**2 + y[1]**2) + g*y[1] - (y[2]**2 + y[3]**2)
        return f

if __name__ == '__main__':

    ii = 0
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)

    func_PNODE = ODEFunc().to(device)
    ode_PNODE = petsc_adjoint_mod.ODEPetsc()
    ode_PNODE.setupTS(true_y0, func_PNODE, step_size=step_size, method=args.pnode_method, enable_adjoint=True, implicit_form=args.implicit_form, use_dlpack=args.use_dlpack, M=M)
    optimizer_PNODE = optim.AdamW(func_PNODE.parameters(), lr=args.lr)
    ode_test_PNODE = petsc_adjoint_mod.ODEPetsc()
    ode_test_PNODE.setupTS(true_y0, func_PNODE, step_size=step_size, method=args.pnode_method, enable_adjoint=False, implicit_form=args.implicit_form, use_dlpack=args.use_dlpack, M=M)

    loss_PNODE_array = []
    curr_iter = 1
    best_loss = float('inf')

    loss_save = torch.zeros(args.niters)
    deviation_save = torch.zeros(args.niters)

    loss_func = nn.MSELoss()
    start_PNODE = time.time()
    for itr in range(curr_iter, args.niters + 1):

        batch_t, batch_y = get_batch()
        optimizer_PNODE.zero_grad()
        pred_y_PNODE = ode_PNODE.odeint_adjoint(true_y0, batch_t)
        nfe_f_PNODE = func_PNODE.nfe
        func_PNODE.nfe = 0
        loss_PNODE = loss_func(pred_y_PNODE, batch_y)
        # loss_PNODE = torch.mean(torch.abs(pred_y_PNODE - batch_y))
        loss_PNODE.backward()
        optimizer_PNODE.step()

        loss_save[itr-1] = loss_PNODE
        # deviation_save[itr-1] = torch.mean(torch.abs(func_PNODE.net(pred_y_PNODE)[:,-1]))
        deviation_save[itr-1] = torch.mean(torch.abs(func_PNODE.net2(pred_y_PNODE)))

        nfe_b_PNODE = func_PNODE.nfe
        func_PNODE.nfe = 0

        if itr % args.test_freq == 0:
            with torch.no_grad():
                end_PNODE = time.time()
                test_t = t
                test_y = true_y
                pred_y_PNODE = ode_test_PNODE.odeint_adjoint(true_y0, test_t)
                print('check equality constraint, deviation: ', deviation_save[itr-1])
                loss_PNODE_array= loss_PNODE_array + [loss_PNODE.item()] + [torch.mean(torch.abs(pred_y_PNODE - test_y)).cpu()]
                print('PNODE: Iter {:05d} | Time {:.6f} | Total Loss {:.6f} | NFE-F {:04d} | NFE-B {:04d}'.format(itr, end_PNODE-start_PNODE, loss_PNODE_array[-1], nfe_f_PNODE, nfe_b_PNODE))

                if loss_PNODE_array[-1] < best_loss:
                    best_loss = loss_PNODE_array[-1]
                    new_best = True
                else:
                    new_best = False
                if new_best:
                    visualize(test_t, test_y, pred_y_PNODE, func_PNODE, ii, 'PNODE')
                    ckpt_path = os.path.join(args.train_dir, 'best.pth')
                    torch.save({
                        'iter': itr,
                        'ii': ii,
                        'best_loss': best_loss,
                        'func_state_dict': func_PNODE.state_dict(),
                        'optimizer_state_dict': optimizer_PNODE.state_dict(),
                    }, ckpt_path)
                    print('Saved new best results (loss={}) at Iter {}'.format(best_loss,itr))
                ii += 1
                start_PNODE = time.time()

    np.save('loss_save.npy', loss_save.detach().cpu().numpy())
    np.save('deviation_save.npy', deviation_save.detach().cpu().numpy())

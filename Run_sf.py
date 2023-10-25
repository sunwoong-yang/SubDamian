import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from surrogate_model.MLP import MLP
from surrogate_model.DE import DeepEnsemble
from surrogate_model.GPR import GPR
from surrogate_model.GPRs import GPRs
from surrogate_model.MFDNN import MFDNN
from data_mining.DM import DM


from data_mining.optimize import optimize
import matplotlib.pyplot as plt
from PrePost.PrePost import *
import os

project_dir = "./projects/231025"
os.chdir(project_dir)

class ToyProblem(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.x = torch.rand(size, 1)
        self.y = torch.sin(2 * torch.pi * self.x)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def draw_fig(test_x, test_y, file_name):
    Y_de, std, alea_std, epis_std = DE.predict(test_x, return_var=True)
    fig = plt.figure(figsize=(32,16))
    for y_idx in range(N_out):
        plt.subplot(4,8,y_idx+1)

        plt.scatter(test_y[:,y_idx], Y_de[:,y_idx], c='b', ls='--', label=f"DE {out_header[0]}")
        plt.plot((min(test_y[:,y_idx]),max(test_y[:,y_idx])), (min(test_y[:,y_idx]),max(test_y[:,y_idx])), c='k', ls='--')

        plt.title(f"Y{y_idx+1}")
    plt.savefig(f'{file_name}.png')
    plt.close()
# N_inp, N_out = 1, 2
# inp_header, out_header, train_x, train_y = csv2Num(N_inp=N_inp, dir="ToyProblem3.csv")
N_inp, N_out = 40, 31
inp_header, out_header, train_x, train_y, test_x, test_y = csv2Num(N_inp=N_inp, dir="231025.csv", train_size=240)

# train_y, train_y_scaler = normalize(train_y)

# mlp = MLP(N_inp, [15, 15, 15], "GELU", N_out)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# mlp = mlp.to(device)
#
# criterion_ = nn.MSELoss()
# optimizer_ = optim.Adam(mlp.parameters(), lr=1e-3)
#
# mlp.fit(train_x, train_y, 50000, criterion_, optimizer_)


DE = DeepEnsemble(N_inp, [60, 60, 60], "GELU", N_out, num_models=5)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
DE = DE.to(device)

criterion_ = nn.MSELoss()

optimizer_ = optim.Adam(DE.parameters(), lr=1e-2)
DE.fit(train_x, train_y, 500, optimizer_)
draw_fig(train_x, train_y, "train1")
draw_fig(test_x, test_y, "test1")

optimizer_ = optim.Adam(DE.parameters(), lr=1e-3)
DE.fit(train_x, train_y, 2000, optimizer_)
draw_fig(train_x, train_y, "train2")
draw_fig(test_x, test_y, "test2")

# optimizer_ = optim.Adam(DE.parameters(), lr=1e-4)
# DE.fit(train_x, train_y, 1000, optimizer_)
# draw_fig(train_x, train_y, "train3")
# draw_fig(test_x, test_y, "test3")

# gpr = GPR(n_restarts_optimizer=10, normalize_y=True)
# gpr.fit(train_x, train_y)
#
# gprs = GPRs(n_restarts_optimizer=10, normalize_y=True)
# gprs.fit(train_x, train_y)


# Y_de, std, alea_std, epis_std = DE.predict(test_x, return_var=True)
# Y_mlp = mlp.predict(test_x)
# Y_gpr, STD_gpr = gprs.predict(test_x, True)


# plt.scatter(gprs.models[1].train_x, gprs.models[1].train_y, c='k', label=f'Train data {out_header[1]}')
# plt.plot(X_test, Y_gpr[:,1], c='r', label=f"GPR {out_header[1]}")
# plt.plot(X_test, Y_mlp[:,1], c='g', label=f"MLP {out_header[1]}")
# plt.plot(X_test, Y_de[:,1], c='b', label=f"DE {out_header[1]}")
# plt.fill_between(X_test.flatten(), Y_gpr[:,1]-30*STD_gpr[:,1], Y_gpr[:,1]+30*STD_gpr[:,1], color='r', alpha=.5)
# plt.fill_between(X_test.flatten(), Y_de[:,1]-30*epis_std[:,1], Y_de[:,1]+30*epis_std[:,1], color='b', alpha=.5)
# plt.legend()
# plt.title("Y2")
# plt.show()
# function to sum to numbers





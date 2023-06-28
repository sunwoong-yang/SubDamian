import torch
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, Sum, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

import numpy as np

from surrogate_model.MLP import MLP
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim



result_DOE_feasible = pd.read_csv("./result_DOE_feasible_2.csv")

X=result_DOE_feasible.loc[:,"R_mr":"l_rod"]

Y=result_DOE_feasible.loc[:,"GW":"Max_dBA"]

X, Y = X.to_numpy(), Y.to_numpy()

# scaler = preprocessing.MinMaxScaler().fit(X)
# X_scaled=scaler.transform(X)
#
# scaler = preprocessing.MinMaxScaler().fit(Y)
# Y_scaled=scaler.transform(Y)
X_scaled, Y_scaled = X, Y[:,[0]]

X_train, X_valid, Y_train, Y_valid = train_test_split(X_scaled, Y_scaled, test_size=0.1, random_state=42)

# regr = MLPRegressor(hidden_layer_sizes=[10, 10, 10], activation='relu', solver='adam',
#                     learning_rate='adaptive', max_iter=10000,
#                     tol=1e-20, verbose=True, warm_start=True, n_iter_no_change=1000).fit(X_train, Y_train)


mlp = MLP(X_scaled.shape[1], [8,8,8,8,4], "LeakyReLU", Y_scaled.shape[1])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
mlp = mlp.to(device)

criterion_ = nn.MSELoss()
optimizer_ = optim.Adam(mlp.parameters(), lr=1e-3)

mlp.fit(X_train, Y_train, 4000, criterion_, optimizer_)



Y_pred=mlp.predict(X_valid)

Y_pred=pd.DataFrame(Y_pred)


X_aux=np.linspace(0,5000,2)
Y_aux=np.linspace(0,5000,2)

R2=np.zeros(Y.shape[1])


for i in range(Y_scaled.shape[1]):

    plt.figure(figsize=(10,10))
    plt.plot(pd.DataFrame(Y_valid).iloc[:,i],Y_pred.iloc[:,i])
    R2[i]=r2_score(pd.DataFrame(Y_valid).iloc[:,i],Y_pred.iloc[:,i])
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    xymin = min(xmin, ymin)
    xymax = max(xmax, ymax)
    plt.close()

    fig, ax = plt.subplots()

    plt.scatter(pd.DataFrame(Y_valid).iloc[:,i],Y_pred.iloc[:,i],facecolors='none', edgecolors='k')
    ax.set_aspect('equal')
    plt.xlabel('Calculated value',fontsize=15)
    plt.ylabel('Predicted value',fontsize=15)
    plt.title(pd.DataFrame(Y_valid).columns[i])

    plt.plot(X_aux,Y_aux,'r--')

    plt.axis('square')

    plt.xlim(xymin, xymax)
    plt.ylim(xymin, xymax)
    plt.xticks(np.arange(xymin, xymax+xymax*0.0001, (xymax-xymin)/4))
    plt.yticks(np.arange(xymin, xymax+xymax*0.0001, (xymax-xymin)/4))
    plt.tick_params(axis='x',direction='in', pad=10, width=1)
    plt.tick_params(axis='y',direction='in', pad=10, width=1)


    #plt.show()


    plt.savefig("./"+str(pd.DataFrame(Y_valid).columns[i])+".png",dpi=600)
    plt.close()

print(R2)
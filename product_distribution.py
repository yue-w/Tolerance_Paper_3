# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:06:33 2021

@author: wyue
"""
import numpy as np
import matplotlib.pyplot as plt

def assembly(X):
    X1=X[0,:]
    X2=X[1,:]
    X3=X[2,:]
    dividV = np.divide((X1+X2),(X3-X2))
    dividV = dividV[np.logical_and(dividV>=-1,dividV<=1)]
    Y=np.arccos(dividV)
    #Y=np.arccos(np.divide((X1+X2),(X3-X2))) 
    return Y

miuX= np.array([55.29, 22.86, 101.69])
sigmaX = np.array([0.08, 0.042, 0.105])

NSample = 100000
#np.random.uniform(miu_low,miu_high,1)
X1 = np.random.uniform(miuX[0]-3*sigmaX[0], miuX[0]+3*sigmaX[0], NSample)
X2 = np.random.uniform(miuX[1]-3*sigmaX[1], miuX[1]+3*sigmaX[1], NSample)
X3 = np.random.uniform(miuX[2]-3*sigmaX[2], miuX[2]+3*sigmaX[2], NSample)
X = np.array([X1,X2,X3])

Y = assembly(X)

num_bins = 5


fig, ax = plt.subplots(1, ncols=1)
num_bins = 50
ax.hist(Y, num_bins, density=False)
ax.set_title('Y')

fig.tight_layout()
plt.show()
#fig.savefig(fname='hist',dpi=300)
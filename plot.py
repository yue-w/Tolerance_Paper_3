# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:23:47 2021

@author: wyue
"""
import numpy as np
import matplotlib.pyplot as plt

## Design nominal
X0 = 0
sigma = 1
## Lower limit and upper limit
LS = -6
US = 6

## Case 1: miu follows an uniform distribution
## Lower and upper limits of the drifted miu
miu_low = -6
miu_high = 6


## miu is shifted for M times
M = 1000
## N samples are generated
N = 10000
x_vec = np.zeros((M,N))
miu = np.zeros(M)
for i in range(M):
    miu[i] = np.random.uniform(miu_low,miu_high,1)
    x_vec[i][:] = np.random.normal(miu[i],sigma,N)

x_vec = x_vec.reshape(-1)
## inspect/truncat
x_vec = x_vec[np.logical_and(x_vec>=LS,x_vec<=US)]

# Tweak spacing to prevent clipping of ylabel
num_bins = 50


fig, ax = plt.subplots(nrows=3, ncols=1)

ax[0].hist(miu, 10, density=False)
#ax[0].set_ylabel('miu')
ax[0].set_title('miu')

# the histogram of the data
ax[1].hist(x_vec, num_bins, density=False)

#ax.set_xlabel('x')
#ax.set_ylabel('Probability density')
ax[1].set_title(r'Dimension')

a = np.random.uniform(LS,US,len(x_vec))
ax[2].hist(a, num_bins,range=[-6, 6], density=False)
ax[2].set_title('Uniform distributiom')

# =============================================================================
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.hist(x_vec, num_bins, density=False)
# ax.set_xlabel('x')
# ax.set_ylabel('Probability density')
# ax.set_title(r'Dimension')
# =============================================================================

fig.tight_layout()
plt.show()
fig.savefig(fname='hist',dpi=300)
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:41:16 2020

@author: wyue
"""


import matplotlib.pyplot as plt
import numpy as np

#Cost-Rate function. The ith component is the ith component in the returned array
def Cprocess(A,B,r):
    return np.add(A,np.divide(B,r))

r = np.arange(3,20,0.1).reshape((1, -1))

A = np.array([2.4, 2.6, 6.15, 4.9, 7.0]).reshape((5, 1))
B = np.array([1.91, 1.8, 3.2, 2.14, 3.8]).reshape((5, 1))

Cp = Cprocess(A,B,r)


fig, ax = plt.subplots()
line1 = ax.plot(r.T, Cp[0], label=r'$X_1$',color='r')
line2 = ax.plot(r.T, Cp[1], label=r'$X_2$', color='g')
line3 = ax.plot(r.T, Cp[2], label=r'$X_3$', color='b')
line4 = ax.plot(r.T, Cp[3], label=r'$X_4$', color='black')
line5 = ax.plot(r.T, Cp[4], label=r'$X_5$', color='yellow')

ax.set_xlabel(r'$\mathcal{r}$',fontsize=11)
ax.set_ylabel(r'$C_P$',fontsize=11)
ax.legend()
plt.show()
#fig.savefig(fname='fig',dpi=300)
fig.savefig('Cp_r.tif',dpi=300)
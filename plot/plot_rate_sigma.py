# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:36:44 2020

@author: wyue
"""
import matplotlib.pyplot as plt
import numpy as np

r = np.arange(5,15,0.1)

E = np.array([0.04, 0.0267, 0.0667]) * 0.8 
F = np.array([0.0020*2, 0.0020, 0.0020*3.0]) * 0.1

sigma0 = E[0] +  F[0]*np.power(r,2)
sigma1 = E[1] +  F[1]*np.power(r,2)
sigma2 = E[2] +  F[2]*np.power(r,2) 

fig, ax = plt.subplots()
line1 = ax.plot(r,sigma0,label='Hub',color='r')
line2 = ax.plot(r,sigma1,label='Roller',color='g')

line3 = ax.plot(r,sigma2,label='Cage',color='b')

ax.set_xlabel(r'$\mathcal{r}$',fontsize=11)
ax.set_ylabel(r'$\sigma$',fontsize=11)
ax.legend()
plt.show()
#fig.savefig(fname='fig',dpi=300)
fig.savefig('E_F_sigma.tif',dpi=300)
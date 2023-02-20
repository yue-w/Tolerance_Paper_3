# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:08:16 2021

@author: wyue
"""

import helpers as hp
import numpy as np
import matplotlib.pyplot as plt


"""
X = np.array([55.30, 22.86, 101.60])
sigmaX = np.array([0.080, 0.042, 0.105])

## miu is shifted for M times
M = 500
## N samples are generated
N = 10000

x = np.zeros((3,M*N))
for i in range(3):
    
    x[i] = np.random.normal(X[i],sigmaX[i],M*N)

y = hp.assembly(x)
#hp.plot_hist(y,30)
#hp.normaltest(Y)
#hp.probability_plot(Y)

_, sigmaY_simulation = hp.stat_parameters_simulation(y)
## Analytical values
sigmaY_Taylor = hp.stat_parameters_analysis(X,sigmaX)  
error = (sigmaY_simulation - sigmaY_Taylor)/ sigmaY_simulation
print("error: ",error)

A = 500
B = 300
C = 375
D = 750
E = 125
U = 992.910

ans = B + np.sqrt(D*D - 0.5*C*C - np.power(A-E-np.sqrt(2)/2*C,2))
print(ans)
A = 500
B = 300
C = 375
D = 750
E = 125
U = 992.910

ans = B + np.sqrt(D*D - 0.5*C*C - np.power(A-E-np.sqrt(2)/2*C,2))
print(ans)

x = np.linspace(0, 8, 500)
G = 0.5
V = 1
H = 0.3
y = G + H * np.exp(-x*V)

fig, ax = plt.subplots()

line, = ax.plot(x, y)
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('Maintenance cost ($)')
plt.show()

"""
#%%
import numpy as np
A = np.array([0.48, 0.52, 1.23, 0.98, 1.4]) * 5
# %%
print(A)
# %%
(997712-997069)/997712
# %%
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 3, 10, endpoint=True)
y1 = 0.05 * x**2
y2 = -1 *y1

def plot_compare_U(x, U1, U2, x_label, y_label,fname="compare_U.tif"):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(x, U1, color='cyan', linestyle='solid',marker='o', label='U1')
    lns2 = ax2.plot(x, U2, color='green',linestyle='solid', marker='s', label="U2")

    ax1.set_xlabel(r'$\it{\sigma}_{Y}$')
    ax1.set_ylabel(f'$x_{i}$') 
    ax2.set_ylabel(r'an equation: $E=mc^2$, test') 
    fig.savefig(fname,dpi=300)
    # ax1.legend(p,[p_.get_label() for p_ in p], loc= 'best', fontsize='small', ncol=2,)
    # fig.legend(loc="upper right")
    # ax1.legend(loc=0)
    # ax2.legend(loc=0)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs,loc= 'best')
    ax1.grid(True)   
    plt.show()

plot_compare_U(x, y1, y2, "x", "y1")

# %%

"""
Plot normal distributions with different sigma.
Pleot normal distributions with fixed sigma and shifting mean.
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


#%%
#### (a)
miu = 0
vars = np.array([1, 2, 3, 4])
sigmas = np.sqrt(vars)
M = 5
xmin = -5 
xmax = 5
x = np.arange(xmin, xmax, 0.05)

fig, ax = plt.subplots()
for sigma in sigmas:
    y = stats.norm.pdf(x, miu, sigma)
    ax.plot(x, y, linewidth=3)

ax.axis('off')
plt.savefig('fig2_a'+".tif", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


#%%
#### (b)
mius = np.arange(3, 7, 0.5)
sigma = 2
M = 5
xmin = -6
xmax = 16
x = np.arange(xmin, xmax, 0.05)

fig, ax = plt.subplots()
for miu in mius:
    y = stats.norm.pdf(x, miu, sigma)
    ax.plot(x, y, linewidth=3,linestyle = 'dashed')

ax.axis('off')
plt.savefig('fig2_b'+".tif", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
# %%

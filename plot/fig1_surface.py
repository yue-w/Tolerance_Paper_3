
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# X = np.arange(-5, 5, 0.01)
# Y = np.arange(-5, 5, 0.01)
# X, Y = np.meshgrid(X, Y)
# #Z = Y*np.sin(X) - X * np.cos(Y)
# Z = Y*np.sin(X) - X * np.cos(0.8*Y)
# # Z = np.sin(X)*np.cos(0.9*Y)
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
                       

# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# #fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()


# Make data.
X = np.arange(0, 5, 0.01)
Y = np.arange(0, 5, 0.01)
X, Y = np.meshgrid(X, Y)

# Z = np.sin(X)*np.cos(Y)
Z = - np.sin(1.2*X)*np.cos(0.9*Y)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
                       

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

# ax.set_ylabel('Variation of shaft')
# ax.set_xlabel('Variation of hole')
# ax.set_zlabel('Total cost')

# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
plt.savefig('fig1_surface'+".tif", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


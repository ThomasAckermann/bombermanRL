import numpy as np
import matplotlib.pyplot as plt

the = np.load('theta_zwischenstand.npy')

plt.imshow(the, cmap = 'gray_r')
plt.colorbar()
plt.show()

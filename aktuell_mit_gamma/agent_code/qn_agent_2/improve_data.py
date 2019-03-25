
import numpy as np

theta_mit_bomben = np.load('thetas/{}.npy'.format('theta_q_mit_bomben'))
#plt.figure(figsize = (18, 5))
#plt.imshow(theta2-theta, cmap = 'gray_r')
#plt.colorbar(shrink = 0.6)
#plt.savefig('{}_diff.png'.format(file), format= 'png')

theta_mit_bomben[5, 32] = 3
#theta2[4, 31] = -10
np.save('thetas/theta_q_mit_bomben.npy', theta_mit_bomben)

import numpy as np

theta = np.load('theta_q.npy')
print('Theta hat shape', np.shape(theta), 'und lautet:')
print(theta)

q_data = np.load('q_data.npy')
print('q_data hat shape', np.shape(q_data), 'und lautet:')
print(q_data[-1])

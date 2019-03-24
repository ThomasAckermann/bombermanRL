import numpy as np 

np.save('q_data.npy', np.array([np.r_[-1, np.zeros(23)]]))  # ,(16,1)))
np.save('all_data.npy', np.array([np.r_[-1, np.zeros(23)]]))  # ,(16,1)))


# np.save('theta_q.npy', np.zeros((6, 19)))
np.save('y_data.npy', np.array([0]))

theta_alt = np.load('theta_q.npy')
print(np.shape(theta_alt), theta_alt)
tmp = theta_alt[:, -1]
tmp = np.reshape(tmp, (6, 1))
theta_alt = np.append(theta_alt[:,:-1], np.zeros((6, 4)), axis = -1)
theta_alt = np.append(theta_alt, tmp, axis = -1)
print(np.shape(theta_alt), theta_alt)
# np.save('theta_q.npy', theta_alt)



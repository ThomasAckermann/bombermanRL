import numpy as np 

np.save('q_data.npy', np.array([np.r_[-1, np.zeros(15)]]))# ,(16,1)))
#np.save('q_data.npy', np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1]]))
np.save('theta_q.npy', np.zeros((6,43)))
np.save('y_data.npy', np.array([0]))




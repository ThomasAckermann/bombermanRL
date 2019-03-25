import numpy as np
danger = 1.6
bomb = [0, 0, 0]
print(np.max([(0.633-0.133*bomb[2]), danger]))
q = np.load('q_data/q_data.npy')
print(q[-5:])

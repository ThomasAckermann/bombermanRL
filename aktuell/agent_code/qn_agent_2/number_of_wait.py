mport numpy as np 

data = np.load('all_data.npy')[-18000:]
shape = np.shape(data)
print('shape of data: ', shape)
for i in range(6):
    print(i)
    n_wait = len(data[:,0][data[:,0] == i])
    print(n_wait)
    print(n_wait / shape[0])



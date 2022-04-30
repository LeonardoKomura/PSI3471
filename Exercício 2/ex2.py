import numpy as np
import pandas as pd

data_df = pd.read_csv("circles_and_squares.csv")
data_df.head()
data = data_df.to_numpy()
data_df.shape
#print(data)

Xd = data[:800, :]
dt = Xd[:, -1]
d = np.reshape(dt, [-1,1]) 
Xd = np.delete(Xd, -1, axis=1)
#print(Xd)

Xd_test = data[800:, :]
yt_test = Xd_test[:, -1]
y_test = np.reshape(yt_test, [-1,1])
Xd_test = np.delete(Xd_test, -1, axis=1)
#print(Xd_test)

mb_size = 200           # mini-batch size
ne = int(800/mb_size)   # número de épocas
W = np.ones((401, 1))/5
novoW = np.ones((401, 1))/5
coluna1 = np.ones((mb_size,1))
#print(coluna1)

eta = 1

for i in range(ne):
    xt_incomplete = Xd[((i*mb_size)):(i+1)*mb_size, :] # 200x400
    xt = np.append(coluna1, xt_incomplete, axis = 1) # 200x401
    #print(xt)
    x = np.transpose(xt) # 401x200
    #print("X: \n", x)
    d_mb = d[i*mb_size, 0]
    
    for j in range(mb_size):
        W = novoW
        x1 = x[:, j]
        v = np.matmul(np.transpose(x1), W)
        #print(v)
        y = (np.exp(v)-np.exp(-v))/(np.exp(v)+np.exp(-v))
        #print(y)
        e = np.subtract(d_mb, y)
        exp2v = np.exp(2*v)
        exp2v_1 = (exp2v+1)**2
        d_phi = exp2v/exp2v_1
        #print(d_phi)
        delta = np.dot(np.matrix.flatten(d_phi), np.matrix.flatten(e))
        Delta = delta*xt
        novoW = W + eta*Delta

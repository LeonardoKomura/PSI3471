# --------------------------------------------------------------
# André Lucas Pierote Rodrigues Vasconcelos - NUSP: 11356540
# Leonardo Isao Komura                      - NUSP: 11261656
# --------------------------------------------------------------

import numpy as np
import pandas as pd

Nb = 20            # mini-batch size
Ne = 2000          # número de épocas
eta = 0.005

data_df = pd.read_csv("circles_and_squares.csv")
data_df.head()
data = data_df.to_numpy()
data_df.shape
print(data)

coluna1 = np.ones((800,1))
Xd = data[:800, :]
print(Xd)

Xd_test = data[800:, :]
y_test = Xd_test[:, -1]
#y_test = np.reshape(yt_test, [-1,1])
Xd_test = np.delete(Xd_test, -1, axis=1)
coluna = np.ones((200,1))
x_test = np.append(coluna, Xd_test, axis = 1)
#x_test = np.transpose(xt_test)
print(x_test)

w = np.zeros(401)
v = np.zeros(800)
y = np.zeros(800)
e = np.zeros(800)

for i in range(1, Ne):
    np.random.permutation(Xd)[:]
    #x = np.transpose(xt)
    #print("X: \n", x)
    #print(x[j])
    
    d = Xd[:, -1]
    #d = np.reshape(dt, [-1,1]) 
    
    x = np.delete(Xd, -1, axis=1)
    x = np.append(coluna1, x, axis = 1) # 100x401
    
    for j in range(int(800/Nb)-1):
        v = np.matmul(x[j], w)
        #print(j)
        #print(v)
        y = np.sign(v)
        #print(y)
        e = d-y
        #print(e)
        w = np.add(w, (eta/Nb)*np.matmul(np.transpose(x), e))

result = np.zeros(200)
n_erros = 0
for i in range(200):
    result[i] = np.sign(np.matmul(x_test[i], w))
    erro = y_test[i] - result[i]
    if erro != 0: 
        n_erros = n_erros + 1
print(n_erros)

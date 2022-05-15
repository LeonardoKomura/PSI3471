#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[15]:


Nb = 100            # mini-batch size
Ne = 1000           # número de épocas
eta = 0.05


# In[16]:


#Função a) (9900)
data_a_csv = pd.read_csv("dados_a_train.csv")
data_a_csv.head()
data_a = data_a_csv.to_numpy()
data_a.shape
#print(data_a)

coluna_a = np.ones((9899,1))
x_a = np.append(coluna_a, data_a, axis = 1)

#Dados de teste (1000)
test_a_csv = pd.read_csv("dados_a_test.csv")
test_a_csv.head()
test_a = test_a_csv.to_numpy()
test_a.shape

y_a = test_a[:, -1]
x_a_test = np.delete(test_a, -1, axis=1)
coluna_ta = np.ones((999,1))
x_a_test = np.append(coluna_ta, x_a_test, axis = 1)


# In[17]:


#Função b) (9000)
data_b_csv = pd.read_csv("dados_b_train.csv")
data_b_csv.head()
data_b = data_b_csv.to_numpy()
data_b.shape
#print(data_b)

coluna = np.ones((8999,1))
x_b = np.append(coluna, data_b, axis = 1)

#Dados de teste (1800)
test_b_csv = pd.read_csv("dados_b_test.csv")
test_b_csv.head()
test_b = test_b_csv.to_numpy()
test_b.shape

y_b = test_b[:, -1]
x_b_test = np.delete(test_b, -1, axis=1)
coluna_t = np.ones((1799,1))
x_b_test = np.append(coluna_t, x_b_test, axis = 1)


# In[18]:


#Função c) (9000)
data_c_csv = pd.read_csv("dados_c_train.csv")
data_c_csv.head()
data_c = data_c_csv.to_numpy()
data_c.shape
#print(data_c)

x_c = np.append(coluna, data_c, axis = 1)

#Dados de teste (1800)
test_c_csv = pd.read_csv("dados_c_test.csv")
test_c_csv.head()
test_c = test_c_csv.to_numpy()
test_c.shape

y_c = test_c[:, -1]
x_c_test = np.delete(test_c, -1, axis=1)
x_c_test = np.append(coluna_t, x_c_test, axis = 1)


# In[19]:


#Função d) (9000)
data_d_csv = pd.read_csv("dados_d_train.csv")
data_d_csv.head()
data_d = data_d_csv.to_numpy()
data_d.shape
#print(data_d)

x_d = np.append(coluna, data_d, axis = 1)

#Dados de teste (1800)
test_d_csv = pd.read_csv("dados_d_test.csv")
test_d_csv.head()
test_d = test_d_csv.to_numpy()
test_d.shape

y_d = test_d[:, -1]
x_d_test = np.delete(test_d, -1, axis=1)
x_d_test = np.append(coluna_t, x_d_test, axis = 1)


# In[20]:


def phi(x):
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y
def phi_l(x):
    y = 1 - (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))**2
    return y


# In[ ]:


# a)
def neural(x_train, Ne, Nb, eta, n, x_test, d_test):
    w = np.zeros((1,n+1))
    w1 = np.zeros((n,2))
    size = len(x_train)
    size_test = len(x_test)
    
    for i in range(Ne):
        np.random.permutation(x_train)[:]
        dt = x_train[:,-1]
        d = np.reshape(dt, [-1,1]) 
        xt = np.delete(x_train, -1, axis=1)
        x = np.transpose(xt)
        
        for j in range(Nb):
            x1 = np.reshape(x[:, j], [-1,1])
            #print("x1 ", x1)
            v1 = np.matmul(w1, x1)
            #print("v1 ", v1)
            y1 = phi(v1)
            #print("y1 ", y1)
            d_phi1 = phi_l(v1)
            #print("d_phi1 ", d_phi1)
            
            xout = np.vstack([[1],y1])
            #print("xout ", xout)
            
            v = np.matmul(w, xout)
            #print("v ", v)
            y = phi(v)
            #print("y ", y)
            d_phi = phi_l(v)
            #print("d_phi ",d_phi)
            
            e = d[j] - y
            #print("e ", e)
            delta = d_phi*e
            #print("delta ", delta)
            Delta = delta*np.transpose(xout)
            #print("Delta ", Delta)
            w = w + (eta * Delta)
            #print("w ", w)
            
            delta1 = np.multiply(d_phi1,(delta*w.T[1:, :]))
            #print("delta1 ", delta1)
            Delta1 = np.matmul(delta1, np.transpose(x1))
            #print("Delta1 ", Delta1)
            w1 = w1 + (eta*Delta1)
        
    xtest = np.transpose(x_test)
    erro = 0
    for i in range(size_test):
        test = xtest[:, j]
        v1_test = np.matmul(w1, test)
        y1_test = phi(v1_test)
        xout_test = np.append([1], y1_test, axis = 0)
        v_test = np.matmul(w, xout_test)
        y_test = phi(v)
        erro = erro + ((d_test[i]-y_test)**2)
    print("n = ", n, " : Erro quadrático: ", erro/size_test)


# In[ ]:


#3, 4, 5, 10, 15, 20, 50, 100
print("a) f(x) = 1/x")
erro_a = neural(x_a, Ne, Nb, eta, 3, x_a_test, y_a)
erro_a = neural(x_a, Ne, Nb, eta, 5, x_a_test, y_a)
erro_a = neural(x_a, Ne, Nb, eta, 10, x_a_test, y_a)
erro_a = neural(x_a, Ne, Nb, eta, 15, x_a_test, y_a)
erro_a = neural(x_a, Ne, Nb, eta, 20, x_a_test, y_a)
erro_a = neural(x_a, Ne, Nb, eta, 50, x_a_test, y_a)
erro_a = neural(x_a, Ne, Nb, eta, 100, x_a_test, y_a)

print("b) f(x) = log10(x)")
erro_b = neural(x_b, Ne, Nb, eta, 3, x_b_test, y_b)
erro_b = neural(x_b, Ne, Nb, eta, 5, x_b_test, y_b)
erro_b = neural(x_b, Ne, Nb, eta, 10, x_b_test, y_b)
erro_b = neural(x_b, Ne, Nb, eta, 15, x_b_test, y_b)
erro_b = neural(x_b, Ne, Nb, eta, 20, x_b_test, y_b)
erro_b = neural(x_b, Ne, Nb, eta, 50, x_b_test, y_b)
erro_b = neural(x_b, Ne, Nb, eta, 100, x_b_test, y_b)

print("a) f(x) = exp(-x)")
erro_c = neural(x_c, Ne, Nb, eta, 3, x_c_test, y_c)
erro_c = neural(x_c, Ne, Nb, eta, 5, x_c_test, y_c)
erro_c = neural(x_c, Ne, Nb, eta, 10, x_c_test, y_c)
erro_c = neural(x_c, Ne, Nb, eta, 15, x_c_test, y_c)
erro_c = neural(x_c, Ne, Nb, eta, 20, x_c_test, y_c)
erro_c = neural(x_c, Ne, Nb, eta, 50, x_c_test, y_c)
erro_c = neural(x_c, Ne, Nb, eta, 100, x_c_test, y_c)

print("a) f(x) = sen(x)")
erro_d = neural(x_d, Ne, Nb, eta, 3, x_d_test, y_d)
erro_d = neural(x_d, Ne, Nb, eta, 5, x_d_test, y_d)
erro_d = neural(x_d, Ne, Nb, eta, 10, x_d_test, y_d)
erro_d = neural(x_d, Ne, Nb, eta, 15, x_d_test, y_d)
erro_d = neural(x_d, Ne, Nb, eta, 20, x_d_test, y_d)
erro_d = neural(x_d, Ne, Nb, eta, 50, x_d_test, y_d)
erro_d = neural(x_d, Ne, Nb, eta, 100, x_d_test, y_d)


# In[ ]:





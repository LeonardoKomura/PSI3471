#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("vehicles_cleaned_train.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


x = df.drop(['price'], axis = 1).values
y = df['price'].values


# In[6]:


print(x)


# In[7]:


print(y)


# In[8]:


matriz_dumb = pd.get_dummies(df)
print(matriz_dumb)


# In[9]:


x1 = matriz_dumb.drop(['price'], axis=1).values
print(x1)


# In[44]:


#Adicionei coluna de 1´s à esquerda
matriz_unitaria = np.ones((8338,1))
matriz_unitaria


# In[23]:


matriz_X = np.append(matriz_unitaria, x1, axis = 1)
matriz_X


# In[24]:


matriz_X_transposta = matriz_X.transpose()
matriz_X_transposta


# In[27]:


XtX = np.matmul(matriz_X, matriz_X_transposta, out = None)
XtX


# In[42]:


#Primeiro método
#Você pode tentar inverter a matriz usando o np.linalg.inv(), só que vai dar erro pq o determinante vai dar 0
#determinante = np.linalg.det(primeiro_produto)
#print(determinante)
#Se você fizer isso provavelmente verá que o det é igual a 0
#No entanto, se por algum motivo não der, só continua a usar aquela primeira fómula: w = (X^t*X)^-1*X^t*d
#matriz_inversa = np.linalg.inv(primeiro_produto)
#segundo_produto = np.matmul(matriz_inversa, matriz_X_transposta)
#matriz_d = y
#terceiro_produto = npmatmul(segundo_produto, y)


#Segundo método
#Podemos driblar a inversa
#Aqui usei o método que eles descrevem no texto da parte de "Regressão Linear"
#Para isso, notamos que X^t*X*w = X^t*d
#Então, teremos um sistema ax = b
#Para solucioná-lo, usamos a função np.linalg.solve(a,b) que retornará a solução X, ou W no nosso caso
# O valor de "a" já foi calculado; Ele é igual ao nosso primeiro_produto
# Para calcularmos "b" faremos o seguinte: 
# Estou usando a premissa que "d" são os valores de price da entrada train
#Nesse caso, "d" seria igual à matriz y feita anteriormente
segundo_produto = np.matmul(matriz_X_transposta, y)
segundo_produto


# In[43]:


#Agora já tenho "a" e "b", só faltando calcular "x" (W0)
w = np.linalg.solve(primeiro_produto, segundo_produto)


# In[33]:


#De fato as matrizes o número de colunas de matriz da primeira é diferente do número de linhas da segunda
#Assim, realmente não dá pra fazer a multiplicação
#No entanto, eu segui a fórmula que eles passaram, então não sei pq isso ocorreu 


# In[ ]:





# In[ ]:





# In[ ]:





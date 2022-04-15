import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("vehicles_cleaned_train.csv")
df.head()
df.columns

x = df.drop(['price'], axis = 1).values
dt = np.array(df['price'].values)
print("dt: ", dt)
##d = np.transpose(dt)
d = np.reshape(dt, [-1,1])
print("d: ", d)

matriz_dumb = pd.get_dummies(df)
x1 = matriz_dumb.drop(['price'], axis=1).values
coluna1 = np.ones((8338,1))
matriz_X = np.append(coluna1, x1, axis = 1)
print("\nX: ", matriz_X)
Xt = np.transpose(matriz_X)
print("Xt: ", Xt)

XtX = np.matmul(Xt, matriz_X)
XtXinv = np.linalg.inv(XtX)
XtXinv_Xt = np.matmul(XtXinv, Xt)

w = np.matmul(XtXinv_Xt, d)
print("\nw: ", w)



d_test = pd.read_csv("vehicles_cleaned_test.csv")
d_test.head()
d_test.columns
##print("\n\nDados de teste: ", d_test)
x_test = d_test.drop(['price'], axis = 1).values
yt = d_test['price'].values
y = np.transpose(yt)
test_dumb = pd.get_dummies(d_test)
teste_inc = test_dumb.drop(['price'], axis=1).values
coluna1_test = np.ones((2084,1))
teste = np.append(coluna1_test, teste_inc, axis = 1)
print("Teste: ", teste)
prices = np.matmul(teste, w)
print("Prices: ", prices)
error = np.subtract(y,prices)
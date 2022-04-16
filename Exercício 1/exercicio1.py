import numpy as np
import pandas as pd

df = pd.read_csv("vehicles_cleaned_train.csv")
df.head()
df.columns

dt = np.array(df['price'].values)
##print("dt: ", dt)
d = np.reshape(dt, [-1,1])
##print("d: ", d)

df.drop(['price', 'cylinders', 'fuel', 'transmission'], axis = 1).values
x1 = pd.get_dummies(df)
coluna1 = np.ones((8338,1))
matriz_X = np.append(coluna1, x1, axis = 1)
##print("\nX: ", matriz_X)
Xt = np.transpose(matriz_X)
##print("Xt: ", Xt)

XtX = np.matmul(Xt, matriz_X)
XtXinv = np.linalg.inv(XtX)
XtXinv_Xt = np.matmul(XtXinv, Xt)

w = np.matmul(XtXinv_Xt, d)
##print("\nw: ", w)



d_test = pd.read_csv("vehicles_cleaned_test.csv")
d_test.head()
d_test.columns
##print("\n\nDados de teste: ", d_test)
yt = d_test['price'].values
y = np.reshape(yt, [-1, 1])

d_test.drop(['price', 'cylinders', 'fuel', 'transmission'], axis = 1).values
test_dumb = pd.get_dummies(d_test)
coluna1_test = np.ones((2084,1))
##print(coluna1_test)
teste = np.append(coluna1_test, test_dumb, axis = 1)
##print("Teste: ", teste)
prices = np.matmul(teste, w)
print("Prices: ", prices)
error = np.subtract(y,prices)
error2 = np.linalg.norm(error)
print("Erro quadrático: ", error2)
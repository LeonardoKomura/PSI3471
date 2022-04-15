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


df2 = pd.read_csv("vehicles_cleaned_test.csv")
df2
#X_train = df_train_dummies.drop(['Price'], axis=1).values
#y_train = df_train_dummies['Price'].values
#X_test = df_test_dummies.values


# In[10]:


matriz_dumb2 = pd.get_dummies(df2)
print(matriz_dumb2)


# In[18]:


X_train = matriz_dumb.drop(['price'], axis=1).values
y_train = matriz_dumb['price'].values
X_test = matriz_dumb2.drop(['price'], axis=1).values


# In[13]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
reg = RandomForestRegressor(criterion='mse')


# In[14]:


reg =RandomForestRegressor(criterion='mse', max_depth=13, n_estimators=20)


# In[15]:


reg.fit(X_train, y_train)


# In[16]:


reg.score(X_train, y_train)


# In[19]:


p = reg.predict(X_test)


# In[20]:


p


# In[21]:


y_test = matriz_dumb2['price'].values
y_test


# In[23]:


plt.scatter(y_test, p)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# In[ ]:





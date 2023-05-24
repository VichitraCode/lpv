#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout

import warnings
warnings.filterwarnings('ignore')


# In[46]:


col_names=["a","b","c","d","e","rm","g","h","i","j","ptratio","l","lstat","PRICE"]


# In[50]:


df=pd.read_csv("./housing.csv",header=None,delimiter=r"\s+",names=col_names)


# In[51]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.head()


# In[10]:


import seaborn as sns


# In[18]:


sns.boxplot(df.PRICE)


# In[25]:


sns.distplot(df.PRICE)


# In[96]:


import matplotlib.pyplot as plt
plt.subplots(figsize=(20,15))
corr=df.corr()
sns.heatmap(corr,square=True,annot=True)


# In[36]:


import matplotlib.pyplot as plt

x=df.rm
y=df.PRICE
plt.scatter(x,y)
plt.title("variation of house price")
plt.xlabel("rm")
plt.ylabel("price in $1000")



# In[53]:


x=df.ptratio
y=df.PRICE
plt.scatter(x,y)
plt.title("variation of house price")
plt.xlabel("rm")
plt.ylabel("price in $1000")


# In[52]:


x=df.lstat
y=df.PRICE
plt.scatter(x,y)
plt.title("variation of house price")
plt.xlabel("rm")
plt.ylabel("price in $1000")


# In[92]:


x=df.iloc[:,:-1].values
y=df.PRICE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# In[86]:


import keras

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout

model=Sequential()

model.add(Dense(128,activation='relu',input_dim=13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')
data=model.fit(x_train,y_train,epochs=1000)


# In[83]:


y_pred=model.predict(x_test)


# In[91]:


pd.DataFrame(data.history).plot(figsize=(6, 4), xlabel="Epochs", ylabel="Loss", title='Loss Curves')
plt.show()


# In[84]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[85]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)


# In[ ]:





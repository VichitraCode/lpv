#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import warnings  
warnings.filterwarnings('ignore') # to ignore the warnings


# In[7]:


training = pd.read_csv("./Google_Stock_Price_Train.csv")
training.head()


# In[8]:


real_stock_price_train = training.iloc[:, 1:2].values     # creates a 2D array having observation and feature


# In[20]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training2 = sc.fit_transform(real_stock_price_train)


# In[21]:


training2.shape


# In[22]:


# hence in the input we take
X_train = training2[0:1257]  # all but last observation as we don't have the output value for it
y_train = training2[1:1258]  # values shifted by 1


# In[23]:


X_train = np.reshape(X_train, (1257, 1, 1))
# (1257, 1, 1) the 2nd argument is no. of features and 3rd argument is the time step


# In[28]:


# importing libraries
from keras.models import Sequential  # initialize NN as a sequnce of layers
from keras.layers import Dense ,LSTM # to add fully connected layers

rnn_regressor = Sequential()

rnn_regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(1, 1)))
rnn_regressor.add(Dense(units=1))

rnn_regressor.compile(optimizer='adam', loss='mean_squared_error')
rnn_regressor.fit(X_train, y_train, batch_size=32, epochs=100)


# In[29]:


# predicting the training results
predicted_stock_price_train = rnn_regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)


# In[46]:


# visualizing the training results
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(real_stock_price_train, color = 'red', label='Real Google Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
#plt.show()


# In[ ]:





# In[31]:


testing = pd.read_csv("./Google_Stock_Price_Test.csv")
testing.head()


# In[32]:


# taking the column of "open" value of stock price
real_stock_price_test = testing.iloc[:, 1:2].values


# In[33]:


# feature Scaling
inputs = sc.transform(real_stock_price_test)


# In[35]:


# reshaping
inputs = np.reshape(inputs, (20, 1, 1))     # only 20 observations in testing set


# In[36]:


# predicting the stock price (for the year 2017)
predicted_stock_price_test = rnn_regressor.predict(inputs)     # but these are the scaled values


# In[37]:


predicted_stock_price_test = sc.inverse_transform(predicted_stock_price_test)


# In[38]:


# visualizing the results for testing
plt.figure(figsize=(20,10))
plt.plot(real_stock_price_test, color = 'red', label='Real Google Stock Price')
plt.plot(predicted_stock_price_test, color = 'blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction (Test Set)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[39]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(real_stock_price_test, predicted_stock_price_test))
print('The RMSE value is', rmse)


# In[41]:


from sklearn.metrics import r2_score
print(r2_score(real_stock_price_test, predicted_stock_price_test))


# In[ ]:





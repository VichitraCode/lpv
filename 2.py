#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# from sklearn import preprocessing
# from yellowbrick.classifier import ConfusionMatrix


# In[41]:


df = pd.read_csv("letter-recognition.data", sep = ",")


# In[42]:


names = ['letter_Class',
         'x-box',
         'y-box',
         'width',
         'high',
         'onpix',
         'x-bar',
         'y-bar',
         'x2bar',
         'y2bar',
         'xybar',
         'x2ybr',
         'xy2br',
         'x-ege',
         'xegvy',
         'y-ege',
         'yegvx']


# In[43]:


df.columns = names


# In[44]:


df.head(10)


# In[45]:


X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


# In[46]:


onehot_encoder = OneHotEncoder(categories='auto')
y = onehot_encoder.fit_transform(y.reshape(-1, 1)).toarray()


# In[47]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[49]:


model = Sequential()
model.add(Dense(64, input_shape=(16,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax')) 


# In[50]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[36]:


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[51]:


score = model.evaluate(X_test, y_test)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')


# In[52]:


# print(confusion_matrix(Y_test, predictions))
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)


# In[59]:


print(classification_report(y_true, y_pred))


# In[54]:


# create a new input with 16 feature values
new_input = [[4,2,5,4,4,8,7,6,6,7,6,6,2,8,7,10]]

# standardize the input using the scaler object
new_input = scaler.transform(new_input)

# make a prediction
prediction = model.predict(new_input)

# print the predicted letter
val=np.argmax(prediction)

print(chr(ord('A')+val))


# In[ ]:





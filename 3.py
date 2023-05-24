#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split


# In[2]:


fashion_train_df = pd.read_csv('fashion-mnist_train.csv', sep=',')
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep=',')


# In[3]:


fashion_train_df.shape   # Shape of the dataset


# In[4]:


fashion_train_df.columns   # Name of the columns of the DataSet.


# So we can see that the 1st column is the label or target value for each row.
# 
# Now Lets find out how many distinct lables we have.

# In[5]:


print(set(fashion_train_df['label']))


# So we have 10 different lables. from 0 to 9. 
# 
# Now lets find out what is the min and max of values of in the other columns.

# In[6]:


print([fashion_train_df.drop(labels='label', axis=1).min(axis=1).min(), 
      fashion_train_df.drop(labels='label', axis=1).max(axis=1).max()])


# So we have 0 to 255 which is the color values for grayscale. 0 being white and 255 being black.
# 
# Now lets check some of the rows in tabular format

# In[7]:


clothing = {0 : 'T-shirt/top',
            1 : 'Trouser',
            2 : 'Pullover',
            3 : 'Dress',
            4 : 'Coat',
            5 : 'Sandal',
            6 : 'Shirt',
            7 : 'Sneaker',
            8 : 'Bag',
            9 : 'Ankle boot'}


# In[33]:


fig, axes = plt.subplots(3, 4, figsize = (15,15))
for row in axes:
    for axe in row:
        index = np.random.randint(60000)
        img = fashion_train_df.drop('label', axis=1).values[index].reshape(28,28)
        cloths = fashion_train_df['label'][index]
        axe.imshow(img)
        axe.set_title(clothing[cloths])
        #axe.set_axis_off()



# In[32]:


fashion_train_df.head()


# So evry other things of the test dataset are going to be the same as the train dataset except the shape.

# In[33]:


fashion_test_df.shape


# So here we have 10000 images instead of 60000 as in the train dataset.
# 
# Lets check first few rows.

# In[34]:


fashion_test_df.head()


# In[15]:


training = np.asarray(fashion_train_df, dtype='float32')
X_train = training[:, 1:].reshape([-1,28,28,1])
X_train = X_train/255   
y_train = training[:, 0]

testing = np.asarray(fashion_test_df, dtype='float32')
X_test = testing[:, 1:].reshape([-1,28,28,1])
X_test = X_test/255    
y_test = testing[:, 0]


# In[16]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=12345)    # TODO : change the random state to 5


# In[17]:


print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)


# In[38]:


cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(filters=64, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
cnn_model.add(layers.MaxPooling2D(pool_size = (2,2)))
cnn_model.add(layers.Dropout(rate=0.3))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(units=32, activation='relu'))
cnn_model.add(layers.Dense(units=10, activation='sigmoid'))


# **compile the model**

# In[39]:


cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()


# **Train the model**

# In[40]:


cnn_model.fit(x=X_train, y=y_train, batch_size=512, epochs=1, validation_data=(X_val, y_val))


# In[41]:


eval_result = cnn_model.evaluate(X_test, y_test)
print("Accuracy : {:.3f}".format(eval_result[1]))


# In[42]:


y_pred = cnn_model.predict(x=X_test)


# In[43]:


print(y_pred[1])


# In[44]:


print(clothing[np.argmax(y_pred[1])])


# In[46]:


plt.figure()
plt.imshow(X_test[1].reshape(28,28))
plt.show()
cloths = y_test[1]
print(clothing[cloths])


# In[ ]:





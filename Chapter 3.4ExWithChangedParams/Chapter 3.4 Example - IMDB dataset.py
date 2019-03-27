
# coding: utf-8

# In[1]:


# tensorflow is a library for fast tensor manipulation
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# num_words means you'll only keep the top 10,000 most frequently occuring words in the training data
# rare words will be discarded
# this allows to work with vector data of manageable size


# In[2]:


train_data[0]


# In[3]:


train_labels[0]


# In[4]:


max([max(sequence) for sequence in train_data])


# In[5]:


word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]])


# In[6]:


decoded_review


# In[7]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[8]:


x_train[0]


# In[9]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[10]:


# Building the network
from keras import models
from keras import layers

model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[11]:


# Tells the network what optimizer to use
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[12]:


x_train.shape


# In[13]:



x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[14]:


model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['acc'])

# Batch_size is the number of times it runs through the batch and adds it back
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))


# In[15]:


history_dict = history.history
history_dict.keys()
[u'acc', u'loss', u'val_acc', u'val_loss']


# In[16]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[17]:


# This serves as a marker for the originial parameters


# In[18]:


# Overtraining/overfeeding is where it memorizes the pattern that was set for the network
# It memorizes the small patterns and if something falls into its pattern then it
# gets labeled as that thing
# You can add random noise to fix this problem. It creates a more broad outline so it doesn't
# memorize patterns 
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[19]:


plt.clf() # clears the figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[20]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[21]:


results


# In[22]:


model.predict(x_test)


# In[23]:


# First set of changes in parameters
# Epochs increased to 10


# In[24]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[25]:


results


# In[26]:


model.predict(x_test)


# In[27]:


# Second set of changes in parameters
# Batch size increased to 1000


# In[28]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=1000)
results = model.evaluate(x_test, y_test)


# In[29]:


results


# In[30]:


model.predict(x_test)


# In[31]:


# Third set of changes in parameters
# Batch size back to 512. Epochs equal to 4
# sgd optimizer used instead of rmsprop


# In[32]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'sgd',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[33]:


results


# In[34]:


model.predict(x_test)


# In[35]:


# Third set of changes in parameters
# Batch size back to 512. Epochs equal to 4
# sgd optimizer used 
# mean_squared_error loss function used instead of binary_crossentropy


# In[36]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'sgd',
             loss = 'mean_squared_error',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[37]:


results


# In[38]:


model.predict(x_test)


# In[39]:


# Fourth set of changes uses the originial parameters 
# except uses the mse loss function instead o binary_crossentropy


# In[40]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'mse',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[41]:


results


# In[42]:


model.predict(x_test)


# In[43]:


# Fifth set of changes uses the originial parameters 
# except uses the mse loss function instead of binary_crossentropy
# and uses the tanh activation instead of relu


# In[44]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'mse',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[45]:


results


# In[46]:


model.predict(x_test)


# In[47]:


# Sixth set of changes uses the originial parameters 
# and uses the tanh activation instead of relu


# In[48]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[49]:


results


# In[50]:


model.predict(x_test)


# In[51]:


# Seventh set of changes uses the originial parameters 
# and uses the tanh activation instead of relu because that seems to produce better
# accuracy. I also have increased the epochs to 10 and lowered the batch size to 412


# In[52]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=412)
results = model.evaluate(x_test, y_test)


# In[53]:


results


# In[54]:


model.predict(x_test)


# In[55]:


plt.clf() # clears the figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[56]:


# Eighth set of changes uses the originial parameters 
# and uses the tanh activation instead of relu because that seems to produce better
# accuracy. I also have increased the epochs to 6 and lowered the batch size to 512


# In[57]:


model.predict(x_test)


# In[58]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=6, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[59]:


results


# In[60]:


# Eighth set of changes uses the originial parameters 
# and uses the tanh activation instead of relu because that seems to produce better
# accuracy. I also have increased the hidden units used by the layers to 32


# In[61]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(32, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(32, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[62]:


results


# In[63]:


model.predict(x_test)


# In[64]:


# Nineth set of changes uses the originial parameters 
# and uses the mse loss function instead of binary_crossentropy
# I also have increased the hidden units used by the layers to 32
# Am trying to beat [0.08531691110730172, 0.88416] for results


# In[65]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'mse',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[66]:


results


# In[67]:


model.predict(x_test)


# In[68]:


# Tenth set of changes uses the originial parameters 
# and uses the mse loss function instead of binary_crossentropy
# I also have increased the hidden units used by the layers to 64
# Am trying to beat [0.08531691110730172, 0.88416] for results


# In[69]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'mse',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[70]:


results


# In[71]:


model.predict(x_test)


# In[72]:


# Tenth set of changes uses the originial parameters 
# and uses the mse loss function instead of binary_crossentropy
# I also have decreasesd the batch size to 412
# Am trying to beat [0.08531691110730172, 0.88416] for results


# In[73]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'mse',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=412)
results = model.evaluate(x_test, y_test)


# In[74]:


results


# In[75]:


model.predict(x_test)


# In[76]:


# 11th changed loss to mse and epochs to 3
# Am trying to beat [0.08531691110730172, 0.88416] for results


# In[77]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'mse',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=412)
results = model.evaluate(x_test, y_test)


# In[78]:


results


# In[79]:


model.predict(x_test)


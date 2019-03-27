
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


train_data[0]


# In[4]:


train_labels[0]


# In[5]:


max([max(sequence) for sequence in train_data])


# In[6]:


word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]])


# In[7]:


word_index = imdb.get_word_index()
#word_index
#type (train_data)


# In[8]:


decoded_review


# In[9]:


type (decoded_review)


# In[10]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[11]:


x_train[0]


# In[12]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[13]:


# Building the network
from keras import models
from keras import layers

model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[14]:


# Tells the network what optimizer to use
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[15]:


x_train.shape


# In[16]:



x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[17]:


model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['acc'])

# Batch_size is the number of times it runs through the batch and adds it back
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))


# In[18]:


history_dict = history.history
history_dict.keys()
[u'acc', u'loss', u'val_acc', u'val_loss']


# In[19]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[20]:


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


# In[21]:


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


# In[22]:


model = models.Sequential() # Sequential is important
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[23]:


results


# In[24]:


model.predict(x_test)


# In[25]:


#from keras.preprocessing.text import text_to_word_sequence
#file = open("review1.txt", "r")
#review = file.readlines()
#review1 = ' '.join(review)
#review1 = text_to_word_sequence(review)
#review1 = ''.join(review1)


# In[26]:


#turns list into string 
#data = ''.join(review)

#data = data.lower()
#result = re.sub(r'[^a-zA-Z]', " ", data)
#result


# In[27]:


# gets rid of punctuation and makes it lower case
#for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
#    data = data.replace(char,' ')
#data = data.strip("\ufeff")
#data = data.lower()
#data


# In[28]:


#from numpy import array
#words = []

#words = data.split(" ")
#type (words)
#type (train_data)

# converts to numpy.ndarray
#words = array([data.split(" ")])


# In[29]:


#from keras.preprocessing.text import hashing_trick
#testt_data = hashing_trick(data, round(500), hash_function='md5')
#testt_data


# In[30]:


#import numpy as np

#def vectorize_sequences(sequences, dimension=10000):
    #results = np.zeros((len(sequences), dimension))
    #for i, sequence in enumerate(sequences):
    #    results[i, sequence] = 1
    #return results

#x_testt = vectorize_sequences(testt_data)


# In[31]:


#model.predict(x_testt)


# In[32]:


# Below begins where I use the above model with the passed in data of
# reviews of the Samsung Galaxy s10


# In[33]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review1.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[34]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review2.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[35]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review3.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[36]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review4.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[37]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review5.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[38]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review6.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[39]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review7.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[40]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review8.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[41]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review9.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


# In[42]:


import numpy as np
from keras.preprocessing.text import hashing_trick

# reads in text as list 
file = open("review10.txt", "r")
review = file.readlines()

#turns list into string 
data = ''.join(review)

# gets rid of punctuation and makes it lower case
for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\n':
    data = data.replace(char,' ')
data = data.strip("\ufeff")
data = data.lower()

# turns string array into integer array
testt_data = hashing_trick(data, 500, hash_function='md5')

# vectorizes integer array
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_testt = vectorize_sequences(testt_data)

# produces resulting data
model.predict(x_testt)


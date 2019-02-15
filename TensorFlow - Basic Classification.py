
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[2]:


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


train_images.shape


# In[4]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[5]:


len(train_labels)


# In[6]:


train_labels


# In[7]:


len(test_labels)


# In[8]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[9]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# In[10]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[11]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[12]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[13]:


model.fit(train_images, train_labels, epochs=5)


# In[14]:


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# In[15]:


predictions = model.predict(test_images)


# In[16]:


predictions[0]


# In[17]:


np.argmax(predictions[0])


# In[18]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[19]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# In[20]:


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# In[21]:


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)


# In[22]:


# Grab an image from the test dataset
img = test_images[0]

print(img.shape)


# In[23]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[24]:


predictions_single = model.predict(img)

print(predictions_single)


# In[25]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# In[26]:


np.argmax(predictions_single[0])


# In[60]:


from skimage.io import imread
img=imread('/home/mike/Documents/img1.png')
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()


# In[61]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[63]:


predictions_single = model.predict(img)

print(predictions_single)


# In[47]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# In[51]:


from skimage.io import imread
img=imread('/home/mike/Documents/img2.png')
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()


# In[52]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[53]:


predictions_single = model.predict(img)

print(predictions_single)


# In[54]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# In[56]:


from skimage.io import imread
img=imread('/home/mike/Documents/img3.png')
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()


# In[57]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[58]:


predictions_single = model.predict(img)

print(predictions_single)


# In[59]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# In[64]:


from skimage.io import imread
img=imread('/home/mike/Documents/img4.png')
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()


# In[65]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[66]:


predictions_single = model.predict(img)

print(predictions_single)


# In[67]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# In[68]:


from skimage.io import imread
img=imread('/home/mike/Documents/img5.png')
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()


# In[69]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[70]:


predictions_single = model.predict(img)

print(predictions_single)


# In[71]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# In[80]:


from skimage.io import imread
img=imread('/home/mike/Documents/img6.png')
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()


# In[81]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[83]:


predictions_single = model.predict(img)

print(predictions_single)


# In[84]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


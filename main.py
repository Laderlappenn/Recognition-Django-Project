import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sheaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
plt.grid(False)

x_train = x_train / 255
x_test = x_test / 255

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)

plt.figure(figsize=(10,10))
for i in range (25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(x_train[i])
  plt.xlabel(class_names[y_train[i]])

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

predictions = model.predict(x_train)

predictions[12]

np.argmax(predictions[12])

y_train[12]

plt.figure()
plt.imshow(x_train[55])
plt.colorbar()
plt.grid(False)

class_names[np.argmax(predictions[55])]

shirts = []
for i in range(len(predictions)):
  if class_names[np.argmax(predictions[i])] == 'Shirt':
    shirts.append(i)
print(shirts)

import random

random_element = random.choice(shirts)
print(random_element)
plt.figure()
plt.imshow(x_train[random.choice(shirts)])
plt.colorbar()
plt.grid(False)
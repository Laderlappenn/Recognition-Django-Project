import random
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

shirt_top = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'T-shirt/top':
        shirt_top.append(i)
print(shirt_top)

trouser = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'Trouser':
        trouser.append(i)
print(trouser)

pullover = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'Pullover':
        pullover.append(i)
print(pullover)

dress = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'Dress':
        dress.append(i)
print(dress)

coat = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'Coat':
        coat.append(i)
print(coat)

sandal = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'Sandal':
        sandal.append(i)
print(sandal)

sheaker = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'Sheaker':
        sheaker.append(i)
print(sheaker)

bag = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'Bag':
        bag.append(i)
print(bag)

ankle_boot = []
for i in range(len(predictions)):
    if class_names[np.argmax(predictions[i])] == 'Ankle boot':
        ankle_boot.append(i)
print(ankle_boot)


def image(list, count = 1):

    for i in range(count):

        plt.figure()
        plt.imshow(x_train[random.choice(list)])
        plt.colorbar()
        plt.grid(False)

    return None


image(ankle_boot, 3)
image(shirts, 3)
image(bag, 3)
image(sheaker, 3)
image(sandal, 3)
image(coat, 3)
image(dress, 3)
image(pullover, 3)
image(trouser, 3)
image(shirt_top, 3)

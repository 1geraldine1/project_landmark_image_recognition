import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

# with np.load('./img_data_crop2.npz',allow_pickle=True) as data:
#     X_train = data['X_train']
#     Y_train = data['Y_train']
#     X_test = data['X_test']
#     Y_test = data['Y_test']
#
# print(X_train)
# X_train = np.asarray(X_train).astype(np.ndarray)
# Y_train = np.asarray(Y_train).astype(np.ndarray)
#
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y_test))

X_train, X_test, Y_train, Y_test = np.load('./img_data_crop.npy',allow_pickle=True)

print(X_train[0])
print(Y_train[0])

print(X_train.shape, X_test.shape)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

img_height = 180
img_width = 180
num_classes = 4




model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'])

model.summary()

epochs = 10
history = model.fit(X_train,Y_train, batch_size=32, epochs=epochs)

model.save('noncrop.h5')
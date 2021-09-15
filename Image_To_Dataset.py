import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
data_dir = os.path.join(BASE_DIR, "landmark")
json_dir = os.path.join(BASE_DIR, "landmark_json")
categories = os.listdir(data_dir)
num_classes = len(categories)

def make_ds():
    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = 4


    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def create_dataset_noncrop():
    image_size = 128

    X = []
    Y = []

    for idx, category in enumerate(categories):
        label = [0 for i in range(num_classes)]
        label[idx] = 1
        category_dir = os.path.join(data_dir, category)

        for top, dir, f in os.walk(category_dir):
            for filename in f:
                img_dir = os.path.join(category_dir, filename)
                print(img_dir)
                ff = np.fromfile(img_dir, np.uint8)
                img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, None, fx=image_size / img.shape[1], fy=image_size / img.shape[0])
                X.append(img / 256)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    xy = (X_train, X_test, Y_train, Y_test)

    np.savez("./img_data_noncrop.npz", xy)


def create_dataset_crop():
    image_size = 128

    X = []
    Y = []

    for idx, category in enumerate(categories):
        label = [0 for i in range(num_classes)]
        label[idx] = 1
        category_dir = os.path.join(data_dir, category)
        category_json_dir = os.path.join(json_dir, category)

        for top, dir, f in os.walk(category_dir):
            for filename in f:
                img_dir = os.path.join(category_dir, filename)
                img_json_dir = os.path.join(category_json_dir, filename[:-4] + '.json')
                with open(img_json_dir, "r", encoding='UTF-8') as j:
                    img_json = json.load(j)
                # lx < rx, ly < ry
                lx, ly, rx, ry = img_json['regions'][0]['boxcorners']
                print(img_dir)
                ff = np.fromfile(img_dir, np.uint8)
                img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
                crop_img = img[ly:ry, lx:rx]
                # 예외처리. 가끔 lx > rx, ly > ry인 데이터 존재.
                # 해당 상황에서 crop_img.shape가 [0,0,3]이 되는 현상 발견
                if crop_img.shape[0] == 0:
                    crop_img = img[ry:ly, rx:lx]
                print(crop_img.shape)
                img = cv2.resize(crop_img, None, fx=image_size / crop_img.shape[1], fy=image_size / crop_img.shape[0])
                X.append(img / 256)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    xy = (X_train, X_test, Y_train, Y_test)

    np.save('./img_data_crop.npy',xy)

# create_dataset_noncrop()
# create_dataset_crop()

make_ds()
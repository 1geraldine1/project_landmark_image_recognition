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


def create_dataset_noncrop_np():
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


def create_dataset_crop_np():
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

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    np.savez('./img_data_crop_4.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
    # np.savez('./img_data_crop.npz',xy=xy)


def image_data_generator_test():
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20, validation_split=0.2)
    feature_shape = (None,256,256,3)
    label_shape = (None,)

    train_ds = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(
            data_dir, classes=categories,
            class_mode='sparse', subset='training', seed=123
        ),
        output_signature=(
            tf.TensorSpec(shape=tf.TensorShape(feature_shape),dtype=tf.float32),
            tf.TensorSpec(shape=tf.TensorShape(label_shape),dtype=tf.float32)
        )
    )
    test_ds = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(
            data_dir, classes=categories,
            class_mode='sparse', subset='validation', seed=123
        ),
        output_signature=(
            tf.TensorSpec(shape=tf.TensorShape(feature_shape), dtype=tf.float32),
            tf.TensorSpec(shape=tf.TensorShape(label_shape), dtype=tf.float32)
        )
    )

    print(train_ds.element_spec)

    num_classes = 4

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    model.fit(
        train_ds,
        epochs=10,
        batch_size=512
    )

    model.evaluate(test_ds, batch_size=512)


def load_npz():
    label_to_index = {}
    for i in len(categories):
        label_to_index[categories[i]] = i

    filename = 'img_data_crop.npz'
    filepath = os.path.join(BASE_DIR, filename)
    with np.load(filepath) as data:
        X_train = data['X_train']
        X_test = data['X_test']
        Y_train = data['Y_train']
        Y_test = data['Y_test']

    print(X_train, X_test, Y_train, Y_test)


# load_npz()
# create_dataset_noncrop()
# create_dataset_crop_np()

image_data_generator_test()

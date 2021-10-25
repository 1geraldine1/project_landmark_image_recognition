import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from glob import glob

BASE_DIR = Path(__file__).resolve().parent
# 4class = landmark, 72class = image_collected
# data_dir = os.path.join(BASE_DIR, "landmark")
data_dir = os.path.join(BASE_DIR, "image_collected")
data_cropped_dir = os.path.join(BASE_DIR, "cropped_landmark")
test_data_dir = os.path.join(BASE_DIR, "Test")
json_dir = os.path.join(BASE_DIR, "landmark_json")
categories = os.listdir(data_dir)
num_classes = len(categories)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()


# tf.profiler.experimental.start('./log')

def model_cnn():
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=40,
                                                              width_shift_range=0.2,
                                                              height_shift_range=0.2,
                                                              shear_range=0.2,
                                                              zoom_range=0.2,
                                                              horizontal_flip=True,
                                                              fill_mode='nearest',
                                                              validation_split=0.2)

    feature_shape = (None, 128, 128, 3)
    label_shape = (None,)
    epoch = 50

    train_ds = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(
            data_dir, classes=categories,
            class_mode='sparse', subset='training', seed=123,
            target_size=(128, 128), batch_size=32
        ),
        output_signature=(
            tf.TensorSpec(shape=tf.TensorShape(feature_shape), dtype=tf.float16),
            tf.TensorSpec(shape=tf.TensorShape(label_shape), dtype=tf.float16)
        )
    )
    val_ds = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(
            data_dir, classes=categories,
            class_mode='sparse', subset='validation', seed=123,
            target_size=(128, 128), batch_size=32
        ),
        output_signature=(
            tf.TensorSpec(shape=tf.TensorShape(feature_shape), dtype=tf.float16),
            tf.TensorSpec(shape=tf.TensorShape(label_shape), dtype=tf.float16)
        )
    )

    train_ds = train_ds.prefetch(100 * 32)
    val_ds = val_ds.prefetch(100 * 32)

    with tf.device('gpu:0'):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.PReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.PReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.PReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes),
            tf.keras.layers.Softmax()
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        history = model.fit_generator(
            train_ds,
            epochs=epoch,
            # steps_per_epoch = train개수 / batch_size => 10259/32 => 올림하여 321
            steps_per_epoch=321,
            validation_data=val_ds,
            # validation_steps = valid개수 / batch_size => 2563/32 => 올림하여 81
            validation_steps=81
        )

    model.save('model01_noncrop.h5')

    plot_model(history, epoch)


def model_efficientnet():
    image_size = 600
    batch_size = 16

    ds = image_dataset_from_directory(data_dir,
                                      label_mode='categorical',
                                      class_names=categories,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      image_size=(image_size, image_size)
                                      )

    train_size = int(0.7 * len(list(ds)))
    test_size = int(0.15 * len(list(ds)))

    ds = ds.shuffle(buffer_size=len(list(ds)))
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    val_ds = test_ds.skip(test_size)
    test_ds = test_ds.take(test_size)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2, fill_mode='constant')
    ], name='data_augmentation')

    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    with strategy.scope():
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        x = preprocess_input(inputs)
        x = data_augmentation(x)

        IMG_SHAPE = (image_size, image_size) + (3,)
        base_model = tf.keras.applications.EfficientNetB7(input_shape=IMG_SHAPE,
                                                          input_tensor=x,
                                                          include_top=False,
                                                          weights='imagenet')

        base_model.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = tf.keras.layers.Dropout(top_dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    with strategy.scope():
        loss0, accuracy0 = model.evaluate(val_ds)
        print('initial loss : {:.2f}'.format(loss0))
        print('initial accuracy: {:.1f}%'.format(accuracy0 * 100))

    initial_epochs = 50

    filepath = './checkpoint/full_data_v3_initial.h5'
    checkpoint = ModelCheckpoint(filepath=filepath, mode='max', monitor='val_accuracy', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    tf.debugging.set_log_device_placement(True)
    with strategy.scope():
        history = model.fit(train_ds,
                            epochs=initial_epochs,
                            validation_data=val_ds,
                            callbacks=callbacks_list)

    # model performance check
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    # plt.show()

    # fine tuning - unfreeze model
    for layer in model.layers[-50:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # fine tuning - training
    fine_tune_epochs = 50
    total_epochs = initial_epochs + fine_tune_epochs

    filepath = './checkpoint/finetune_v3.h5'
    checkpoint = ModelCheckpoint(filepath=filepath, mode='max', monitor='val_accuracy', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    history_fine = model.fit(train_ds,
                             epochs=fine_tune_epochs,
                             validation_data=val_ds,
                             callbacks=callbacks_list)

    # model performance check2
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.5, 1])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # plt.show()

    # model.load_weights('./checkpoint/full_data_finetune.h5')

    # model test
    loss, accuracy = model.evaluate(test_ds)
    print('Test accuracy :', accuracy)

    model.save('efficientnetB7_full_data_train_done_v3')

    import pickle

    # Define the save names
    base_filepath = ''.join('./trained_model/efficientnetB7_full_data_trained_v3')
    json_filepath = base_filepath + '.json'
    weights_filepath = base_filepath + '.h5'
    pkl_filepath = base_filepath + '.pkl'

    # save model and weights
    model_json = model.to_json()

    with open(json_filepath, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_filepath)

    # save history
    with open(pkl_filepath, 'wb') as history_file:
        pickle.dump(history.history, history_file)
        # pickle.dump(history_file)


def plot_model(history, epoch):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epoch)

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


def model_efficientnet_reduce_time():
    files = tf.data.Dataset.list_files(str(data_dir + '\\*\\*'), shuffle=False)

    num_files = len([file for file in glob(str(data_dir + '\\*\\*'))])
    print(num_files)

    class_names = np.array(sorted(categories))
    print(class_names)

    image_size = 600
    batch_size = 16

    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)

        one_hot = parts[-2] == class_names

        return tf.argmax(tf.cast(one_hot, tf.int32))

    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)

        return tf.image.resize(img, [image_size, image_size])

    def process_TL(file_path):
        label = get_label(file_path)

        img = tf.io.read_file(file_path)
        img = decode_img(img)
        img = preprocess_input(img)
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE

    ds = files.interleave(lambda x: tf.data.Dataset.list_files(str(data_dir + '\\*\\*'), shuffle=True),
                          cycle_length=4).map(process_TL, num_parallel_calls=AUTOTUNE)

    train_size = int(0.8 * num_files)
    val_size = int(0.2 * num_files)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    train_ds = train_ds.repeat().batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2, fill_mode='constant')
    ], name='data_augmentation')

    def create_model():
        input_layer = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        x = data_augmentation(input_layer)
        base_model = tf.keras.applications.EfficientNetB7(input_tensor=x,
                                                          include_top=False,
                                                          weights='imagenet')

        base_model.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = tf.keras.layers.Dropout(top_dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(input_layer, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    model = create_model()
    model.summary()

    filepath = './checkpoint/full_data_v3_initial.h5'
    checkpoint = ModelCheckpoint(filepath=filepath, mode='max', monitor='val_accuracy', verbose=1, save_best_only=True,
                                 save_weights_only=False)

    class MyThresholdCallback(tf.keras.callbacks.Callback):
        def __init__(self, threshold):
            super(MyThresholdCallback, self).__init__()
            self.threshold = threshold

        def on_epoch_end(self, epoch, logs=None):
            val_acc = logs["val_accuracy"]
            if val_acc >= self.threshold:
                self.model.stop_training = True

    my_callback = MyThresholdCallback(threshold=0.99)

    history = model.fit(train_ds,
                        steps_per_epoch=int(train_size / batch_size),
                        validation_data=val_ds,
                        validation_steps=int(val_size / batch_size),
                        callbacks=[my_callback],
                        epochs=50)


model_efficientnet()

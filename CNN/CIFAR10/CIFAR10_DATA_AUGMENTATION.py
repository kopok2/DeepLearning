# coding=utf-8
"""CIFAR10 dataset classifier with data augmentation."""

import os

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True)

    IMG_CHANNELS = 3
    IMG_X_SIZE = 32
    IMG_Y_SIZE = 32

    BATCH_SIZE = 128
    EPOCHS = 35
    CLASSES = 10
    VERBOSE = 2
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = RMSprop()


    print("X_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")

    Y_train = np_utils.to_categorical(y_train, CLASSES)
    Y_test = np_utils.to_categorical(y_test, CLASSES)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    datagen.fit(X_train)

    model = Sequential()
    model.add(Conv2D(12, (3, 3), padding="same", input_shape=(IMG_X_SIZE, IMG_Y_SIZE, IMG_CHANNELS)))
    model.add(Activation("relu"))
    model.add(Conv2D(12, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(12, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(12, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CLASSES))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE), epochs=EPOCHS, verbose=VERBOSE,
                        steps_per_epoch=X_train.shape[0])
    score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])

    model_json = model.to_json()
    open("cifar10_cnn_model.json", "w").write(model_json)
    model.save_weights("cifar10_cnn_weights.h5", overwrite=True)
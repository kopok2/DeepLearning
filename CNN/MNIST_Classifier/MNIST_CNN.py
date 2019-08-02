# coding=utf-8
"""MNIST Dataset classifier using convolutional neural network (CNN)."""

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt


class CNNClassifier:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(20, 5, padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


if __name__ == "__main__":
    EPOCHS = 12
    BATCH_SIZE = 128
    VERBOSE = 2
    OPTIMIZER = Adam()
    VALIDATION_SPLIT = 0.2
    IMG_ROWS, IMG_COLS = 28, 28
    CLASSES = 10
    INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    K.set_image_dim_ordering("th")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    print("{0} training samples".format(X_train[0]))
    print("{0} test samples".format(X_test[0]))
    y_train = np_utils.to_categorical(y_train, CLASSES)
    y_test = np_utils.to_categorical(y_test, CLASSES)
    model = CNNClassifier.build(INPUT_SHAPE, CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    model.summary()
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    print("Test score: {0}".format(score[0]))
    print("Test accuracy: {0}".format(score[1]))
    print(history.history.keys())

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

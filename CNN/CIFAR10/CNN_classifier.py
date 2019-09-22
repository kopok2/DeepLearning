# coding=utf-8
"""CIFAR 100 Dataset CNN classifier."""

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam

if __name__ == '__main__':
    IMG_CHANNELS = 3
    IMG_X_SIZE = 32
    IMG_Y_SIZE = 32

    BATCH_SIZE = 128
    EPOCHS = 20
    CLASSES = 100
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = Adam()

    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    print("X_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")

    Y_train = np_utils.to_categorical(y_train, CLASSES)
    Y_test = np_utils.to_categorical(y_test, CLASSES)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    print("Creating model...")
    model = Sequential()
    # Convolutional part
    model.add(Conv2D(10, (3, 3), padding="same", input_shape=(IMG_X_SIZE, IMG_Y_SIZE, IMG_CHANNELS)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(50, kernel_size=5, padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Conv2D(10, kernel_size=5, padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Conv2D(10, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense part
    model.add(Flatten())
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CLASSES))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT,
              verbose=VERBOSE)
    score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])

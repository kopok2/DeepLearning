# coding=utf-8
"""MNIST dataset classifier using simple multilayer dense perceptron."""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras import regularizers
from keras.utils import np_utils

if __name__ == "__main__":
    EPOCHS_COUNT = 40
    BATCH_SIZE = 128
    VERBOSE_SETTING = 2
    OUT_CLASSES_COUNT = 10
    DROPOUT = 0.3

    OPTIMIZER = Adam()
    HIDDEN_LAYER_SIZE = 256
    VALIDATION_SPLIT_RATIO = 0.2
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    RESHAPED = 28 * 28

    # Transform data
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    print(X_train.shape[0], "training instances.")
    print(X_test.shape[0], "testing instances.")

    Y_train = np_utils.to_categorical(y_train, OUT_CLASSES_COUNT)
    Y_test = np_utils.to_categorical(y_test, OUT_CLASSES_COUNT)

    model = Sequential()
    model.add(Dense(HIDDEN_LAYER_SIZE, input_shape=(RESHAPED,)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(HIDDEN_LAYER_SIZE, kernel_regularizer=regularizers.l2()))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(HIDDEN_LAYER_SIZE, kernel_regularizer=regularizers.l2()))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(OUT_CLASSES_COUNT))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS_COUNT,
                        verbose=VERBOSE_SETTING, validation_split=VALIDATION_SPLIT_RATIO)

    score = model.evaluate(X_test, Y_test, verbose=VERBOSE_SETTING)
    print("Model score:", score[0])
    print("Model accuracy:", score[1])

    print("Saving model to json...")
    open("mnist_model.json", "w").write(model.to_json())

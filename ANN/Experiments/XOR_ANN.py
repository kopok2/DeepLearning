# coding=utf-8
"""XOR function emulation using artificial neural network."""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np


def v_xor(X):
    result = []
    for a in X:
        if a[0] and not a[1]:
            result.append(1)
        elif a[1] and not a[0]:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)


if __name__ == "__main__":
    X_train = np.column_stack((np.hstack((np.zeros(1000), np.ones(2000), np.zeros(1000))),
                         np.hstack((np.ones(2000), np.zeros(2000))))).reshape(4000, 2)
    y_train = np.hstack((np.ones(1000), np.zeros(1000), np.ones(1000), np.zeros(1000)))
    print(X_train, y_train)

    X_test = np.column_stack((np.random.randint(2, size=1000), np.random.randint(2, size=1000)))
    y_test = v_xor(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model = Sequential()
    model.add(Dense(64, input_shape=(2,)))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    model.summary()
    model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=10, epochs=50, verbose=2, validation_split=0.2)

    score = model.evaluate()
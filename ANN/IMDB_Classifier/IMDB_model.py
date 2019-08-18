# coding=utf-8
"""IMDB Movie reviews sentiment classification using multilayer neural network."""

import numpy as np
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
from keras.datasets import imdb
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
# Hyperparameters
EPOCHS = 25
VERBOSE = 2
OPTIMIZER = Adam()
BATCH_SIZE = 256


def construct_ANN_classifier(layers, width):
    model = Sequential()
    model.add(Dense(width, input_shape=(1000,)))
    model.add(Activation("relu"))
    for layer in range(layers):
        model.add(Dense(width))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation("softmax"))
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                          num_words=None,
                                                          skip_top=0,
                                                          maxlen=None,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)
    x_train = pad_sequences(x_train, maxlen=1000)
    x_test = pad_sequences(x_test, maxlen=1000)
    model = construct_ANN_classifier(2, 5000)
    model.summary()
    model.compile(optimizer=OPTIMIZER, loss="mean_squared_error", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=0.2)
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print(score)
    print(confusion_matrix(y_test, model.predict(x_test)))

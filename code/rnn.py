import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential


def get_model(Y_range):
    # Initialising the Model
    model = Sequential()
    # Adding layers
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(Y_range, activation='softmax'))
    # Compiling the model for training
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


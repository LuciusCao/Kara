from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import Activation, Dense


def build(input_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(16, return_sequences=True),
                            input_shape=(input_dim, 16)))
    model.add(Dense(8))
    model.add(Activation('linear'))
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    return model

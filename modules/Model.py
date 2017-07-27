from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import Activation, Dense
from seq2seq.models import Seq2Seq


def build_seq2seq(time_step, input_dim, hidden_dim, depth=1, batch_size=None):
    model = Seq2Seq(batch_input_shape=(batch_size, time_step, input_dim),
                    hidden_dim=hidden_dim,
                    output_length=time_step,
                    output_dim=input_dim,
                    depth=depth)
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    return model

def build_basic(time_step, input_dim):
    model = Sequential()
    model.add(LSTM(16, input_shape=(time_step, input_dim)))
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    return model

def build_td_basic(time_step, input_dim):
    model = Sequential()
    model.add(TimeDistributed(Dense(8, input_shape=(time_step, input_dim))))
    model.add(LSTM(8, return_sequences=True))
    model.add(TimeDistributed(Dense(8)))
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    return model

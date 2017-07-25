from modules.Preprocessor import Preprocessor
from modules.Loader import Loader
from modules.Writer import Writer
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import Activation, Dense
from keras.layers import Conv1D, MaxPooling1D
import os


if __name__ == '__main__':
    root_path = os.path.abspath('./dataset')
    preprocessor = Preprocessor(root_path)
    preprocessor.convert_all()
    loader = Loader(
        os.path.abspath('./dataset/wav/a_hisa - Town of Windmill.wav'),
        2048, 64
    )
    x, y, shape = loader.load_training_data()
    writer = Writer()
    # experiment code below
    model = Sequential()

    # model.compile(loss='mean_absolute_error', optimizer='rmsprop')

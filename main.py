from modules.Preprocessor import Preprocessor
from modules.Loader import Loader
from modules.Writer import Writer
from modules.Model import build_basic, build_td_basic, build_seq2seq
import os


if __name__ == '__main__':
    root_path = os.path.abspath('./dataset')
    preprocessor = Preprocessor(root_path)
    preprocessor.convert_all()
    loader = Loader(
        os.path.abspath('./dataset/wav/a_hisa - Town of Windmill.wav'),
        2048, 32
    )
    x, y, shape = loader.load_training_data()
    model = build_seq2seq(shape[1], shape[2], 128, depth=3)
    writer = Writer()

from modules.Preprocessor import Preprocessor
from modules.Loader import Loader
import os
import helpers as h

if __name__ == '__main__':
    root_path = os.path.abspath('./dataset')
    preprocessor = Preprocessor(root_path)
    preprocessor.convert_all()
    loader = Loader(
        os.path.abspath('./dataset/wav/a_hisa - Town of Windmill.wav'))

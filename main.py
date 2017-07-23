from modules import Preprocessor
import os

if __name__ == '__main__':
    root_path = os.path.abspath('./dataset')
    preprocessor = Preprocessor(root_path)
    preprocessor.convert_all()

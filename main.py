from modules.Preprocessor import Preprocessor
from modules.Loader import Loader
from modules.Writer import Writer
import os
import helpers as h

if __name__ == '__main__':
    root_path = os.path.abspath('./dataset')
    preprocessor = Preprocessor(root_path)
    preprocessor.convert_all()
    loader = Loader(
        os.path.abspath('./dataset/wav/a_hisa - Town of Windmill.wav'))
    writer = Writer()
    org_seq = writer.de_fourier_transform(loader.fourier_sequences)
    data_from_writer = writer.convert_seq_to_np_audio(org_seq)
    data_from_loader = writer.convert_seq_to_np_audio(loader.sequences)

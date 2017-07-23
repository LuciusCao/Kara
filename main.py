from modules.Preprocessor import Preprocessor
from modules.Loader import Loader
import os
import helpers as h

if __name__ == '__main__':
    root_path = os.path.abspath('./dataset')
    preprocessor = Preprocessor(root_path)
    preprocessor.convert_all()
    data, sample_rate = Loader.read_wav_to_np(os.path.abspath('./dataset/wav/a_hisa - Town of Windmill.wav'))
    seq_data = Loader.convert_np_audio_to_seq(data, sample_rate)
    fft_data = Loader.fourier_transform(seq_data)

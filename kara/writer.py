import scipy.io.wavfile as wav
import numpy as np


class Writer:
    def __init__(self, sample_rate=44100,):
        self.sample_rate = sample_rate

    def write_np_to_wav(self, data, filepath):
        original_data = data * 32767.0
        original_data.astype('int16')
        wav.write(filepath, self.sample_rate, original_data)
        return

    def convert_block_to_np_audio(self, block_data):
        data = np.concatenate(block_data)
        return data

    def de_fourier_transform(self, fourier_blocks):
        row, col = fourier_blocks.shape
        num_elems = col // 2
        original_block = np.zeros((row, num_elems))
        for i in range(row):
            block = fourier_blocks[i, :]
            real = block[:num_elems]
            imag = block[num_elems:]
            complex_block = real + 1.0j * imag
            de_fourier_block = np.fft.ifft(complex_block)
            original_block[i, :] = np.real(de_fourier_block)
        return original_block

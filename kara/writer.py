import scipy.io.wavfile as wav
import numpy as np


class Writer:
    def __init__(self, datablock, filepath, sample_rate):
        self.sample_rate = sample_rate
        self.datablock = datablock
        self.filepath = filepath

    def write_fft_to_wav(self):
        #  original_block = self._de_fourier_transform(self.datablock)
        data = self._convert_block_to_np_audio(self.datablock)
        self._write_np_to_wav(data, self.filepath)
        return

    def _write_np_to_wav(self, data, filepath):
        original_data = data * 32767.0
        original_data.astype('int16')
        wav.write(filepath, self.sample_rate, original_data)
        return

    def _convert_block_to_np_audio(self, block_data):
        data = np.concatenate(block_data)
        return data

    def _de_fourier_transform(self, fourier_blocks):
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

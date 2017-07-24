import scipy.io.wavfile as wav
import numpy as np


class Loader:
    def __init__(self, filepath, block_size):
        self.filepath = filepath
        self._block_size = block_size
        self.data, self.sample_rate = self._read_wav_to_np(self.filepath)
        self.blocks = self._convert_np_audio_to_block(self.data,
                                                       self._block_size)
        self.fourier_blocks = self._fourier_transform(self.blocks)

    def _read_wav_to_np(self, filename):
        data = wav.read(filename)
        np_arr = data[1].astype('float32') / 32767.0
        return np_arr, data[0]

    def _convert_np_audio_to_block(self, music_arr, block_size):
        num_blocks = len(music_arr) // block_size
        data = np.zeros((num_blocks, block_size))
        cur_pointer = 0
        while cur_pointer < num_blocks:
            block = music_arr[cur_pointer*block_size:(cur_pointer+1)*block_size]
            if block.shape[0] < block_size:
                padding = np.zeros((block_size - block.shape[0],))
                block = np.concatenate((block, padding))
            data[cur_pointer, :] = block
            cur_pointer += 1
        return data

    def _fourier_transform(self, data):
        # plausible need confirm
        row, col = data.shape
        fft_data = np.zeros((row, col*2))
        for i in range(row):
            fft_block = np.fft.fft(data[i, :])
            target_block = np.concatenate((np.real(fft_block),
                                           np.imag(fft_block)))
            fft_data[i] = target_block
        return fft_data

    def load_training_data(self, data_set):
        train_x = data_set
        train_y = train_x[1:]
        train_y.append(np.zeros(self._block_size))

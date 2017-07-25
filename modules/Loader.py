import scipy.io.wavfile as wav
import numpy as np


class Loader:
    def __init__(self, filepath, block_size, max_seq_len):
        self.filepath = filepath
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.raw_data, self.sample_rate = self._read_wav_to_np(self.filepath)

    def _read_wav_to_np(self, filename):
        data = wav.read(filename)
        np_arr = data[1].astype('float32') / 32767.0
        return np_arr, data[0]

    def _convert_np_audio_to_block(self, music_arr):
        num_blocks = len(music_arr) // self.block_size
        data = []
        cur_pointer = 0
        while cur_pointer < num_blocks:
            block = music_arr[cur_pointer*self.block_size:
                              (cur_pointer+1)*self.block_size]
            if block.shape[0] < self.block_size:
                padding = np.zeros((self.block_size - block.shape[0],))
                block = np.concatenate((block, padding))
            data.append(block)
            cur_pointer += 1
        return data

    def _fourier_transform(self, data):
        row = len(data)
        fft_data = []
        for i in range(row):
            fft_block = np.fft.fft(data[i])
            target_block = np.concatenate((np.real(fft_block),
                                           np.imag(fft_block)))
            fft_data.append(target_block)
        return fft_data

    def load_training_data(self):
        x = self._convert_np_audio_to_block(self.raw_data)
        y = x[1:]
        y.append(np.zeros(self.block_size))
        fft_x = self._fourier_transform(x)
        fft_y = self._fourier_transform(y)
        cur_seq = 0
        total_seq = len(x)
        train_x = []
        train_y = []
        while cur_seq + self.max_seq_len < total_seq:
            train_x.append(fft_x[cur_seq:cur_seq+self.max_seq_len])
            train_y.append(fft_y[cur_seq:cur_seq+self.max_seq_len])
            cur_seq += self.max_seq_len
        num_examples = len(train_x)
        num_dims_out = self.block_size * 2
        shape = (num_examples, self.max_seq_len, num_dims_out)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return train_x, train_y, shape
        # mean_x = np.mean(np.mean(train_x, axis=0), axis=0)
        # std_x = np.sqrt(
            # np.mean(np.mean((train_x - mean_x) ** 2, axis=0), axis=0))

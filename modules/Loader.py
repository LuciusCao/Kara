import scipy.io.wavfile as wav
import numpy as np
import os


class Loader:
    def __init__(self, target_dir, timestep, seq):
        self.target_dir = target_dir
        self.timestep = timestep
        self.seq = seq
        self.fft_seq = self.seq * 2

    def _read_wav_to_np(self, filename):
        data = wav.read(filename)
        np_arr = data[1].astype('float32') / 32767.0
        return np_arr

    def _convert_np_audio_to_block(self, music_arr):
        num_blocks = len(music_arr) // self.seq
        data = []
        cur_pointer = 0
        while cur_pointer < num_blocks:
            block = music_arr[cur_pointer*self.seq:
                              (cur_pointer+1)*self.seq]
            if block.shape[0] < self.seq:
                padding = np.zeros((self.seq - block.shape[0],))
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

    def _load_training_data(self, wav_file):
        np_arr = self._read_wav_to_np(wav_file)
        x = self._convert_np_audio_to_block(np_arr)
        y = x[1:]
        y.append(np.zeros(self.seq))
        fft_x = self._fourier_transform(x)
        fft_y = self._fourier_transform(y)
        cur_seq = 0
        total_seq = len(x)
        train_x = []
        train_y = []
        while cur_seq + self.timestep < total_seq:
            train_x.append(fft_x[cur_seq:cur_seq+self.timestep])
            train_y.append(fft_y[cur_seq:cur_seq+self.timestep])
            cur_seq += self.timestep
        return train_x, train_y

    def load_directory(self):
        file_list = os.listdir(self.target_dir)
        wav_list = [item for item in file_list if item.endswith('wav')]
        train_x = []
        train_y = []
        for wav_file in wav_list:
            wavfile_path = os.path.join(self.target_dir, wav_file)
            x, y = self._load_training_data(wavfile_path)
            train_x.extend(x)
            train_y.extend(y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        num_examples = len(train_x)
        shape = (num_examples, self.timestep, self.fft_seq)
        return train_x, train_y, shape

import scipy.io.wavfile as wav
import numpy as np
import os


class Loader:
    def __init__(self, input_dir, saved_data, timestep, seq):
        self.input_dir = input_dir
        self.saved_data = saved_data
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
        fft_data = []
        for block in data:
            fft_block = np.fft.fft(block)
            target_block = np.concatenate((np.real(fft_block),
                                           np.imag(fft_block)))
            fft_data.append(target_block)
        return fft_data

    def _load_one_wav(self, wav_file):
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

    def _save_data_to_file(self, x, y):
        m = len(x)
        np.save(self.saved_data['x'], x)
        np.save(self.saved_data['y'], y)
        np.save(self.saved_data['init_sample'], x[m//2:m//2+1, ])
        return

    def _load_data_from_file(self):
        return np.load(self.saved_data['x']), np.load(self.saved_data['y'])

    def load_directory(self):
        file_list = os.listdir(self.input_dir)
        wav_list = [item for item in file_list if item.endswith('wav')]
        train_x = []
        train_y = []
        for wav_file in wav_list:
            wavfile_path = os.path.join(self.input_dir, wav_file)
            x, y = self._load_one_wav(wavfile_path)
            train_x.extend(x)
            train_y.extend(y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        self._save_data_to_file(train_x, train_y)
        return train_x, train_y

    def load_training_data(self, force_reload=False):
        if force_reload is True:
            x, y = self.load_directory()
            print('force reloaded')
            return x, y
        try:
            x, y = self._load_data_from_file()
            print('load training data from npy file')
        except FileNotFoundError:
            x, y = self.load_directory()
            print('load raw audio from directory')
        return x, y

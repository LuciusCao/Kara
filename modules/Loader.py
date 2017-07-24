import scipy.io.wavfile as wav
import numpy as np


class Loader:
    def __init__(self, filepath, seq_length):
        self.filepath = filepath
        self._seq_length = seq_length
        self.data, self.sample_rate = self._read_wav_to_np(self.filepath)
        self.sequences = self._convert_np_audio_to_seq(self.data,
                                                       self._seq_length)
        self.fourier_sequences = self._fourier_transform(self.sequences)

    def _read_wav_to_np(self, filename):
        data = wav.read(filename)
        np_arr = data[1].astype('float32') / 32767.0
        return np_arr, data[0]

    def _convert_np_audio_to_seq(self, music_arr, seq_length):
        num_seqs = len(music_arr) // seq_length
        data = np.zeros((num_seqs, seq_length))
        cur_pointer = 0
        while cur_pointer < num_seqs:
            seq = music_arr[cur_pointer*seq_length:(cur_pointer+1)*seq_length]
            if seq.shape[0] < seq_length:
                padding = np.zeros((seq_length - seq.shape[0],))
                seq = np.concatenate((seq, padding))
            data[cur_pointer, :] = seq
            cur_pointer += 1
        return data

    def _fourier_transform(self, data):
        # plausible need confirm
        row, col = data.shape
        fft_data = np.zeros((row, col*2))
        for i in range(row):
            fft_seq = np.fft.fft(data[i, :])
            target_seq = np.concatenate((np.real(fft_seq), np.imag(fft_seq)))
            fft_data[i] = target_seq
        return fft_data

    def load_training_data(self, data_set):
        train_x = data_set
        train_y = train_x[1:]
        train_y.append(np.zeros(self._seq_length))

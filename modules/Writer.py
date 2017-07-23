import scipy.io.wavfile as wav
import numpy as np


class Writer:
    def __init__(self, sample_rate=44100,):
        self.sample_rate = sample_rate

    def write_np_to_wav(self, data, filepath):
        original_data = data * 32767.0
        original_data.astype('int16')
        wav.write(filepath, self.sample_rate, original_data)
        print('file %s has been written' % (filepath))
        return

    def convert_seq_to_np_audio(self, seq_data):
        data = np.concatenate(seq_data)
        return data

    def de_fourier_transform(self, fourier_sequences):
        row, col = fourier_sequences.shape
        num_elems = col // 2
        original_sequence = np.zeros((row, num_elems))
        for i in range(row):
            seq = fourier_sequences[i, :]
            real = seq[:num_elems]
            imag = seq[num_elems:]
            complex_seq = real + 1.0j * imag
            de_fourier_seq = np.fft.ifft(complex_seq)
            original_sequence[i, :] = de_fourier_seq
        return original_sequence

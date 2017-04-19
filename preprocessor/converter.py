import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
import wave

class Converter():
    def __init__(self, origin_path, tmp_path, target_path, sample_rate=44100):
        self.origin_path = os.path.abspath(origin_path)
        self.tmp_path = os.path.abspath(tmp_path)
        self.target_path = os.path.abspath(target_path)
        self._check_path()

        self.files = os.listdir(self.origin_path)
        self.targets = os.listdir(self.target_path)
        self._clean_files()

        self.sample_rate = sample_rate

    def _check_path(self):
        if not os.path.exists(self.origin_path):
            os.mkdir(self.origin_path)
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
        if not os.path.exists(self.target_path):
            os.mkdir(self.target_path)

    def _clean_files(self):
        self.files = [item[:-4] for item in self.files if item[-4:] == '.mp3']
        self.targets = [item[:-4] for item in self.targets if item[-4:] == '.wav']
        self.files = [item for item in self.files if item not in self.targets]
        if self.files:
            print('files:\n {}\n to be converted'.format(self.files))
        else:
            print('no files require to be converted')

    def mp3_to_wav(self, filename):
        origin_file = self.origin_path + '/' + filename + '.mp3'
        tmp_file = self.tmp_path + '/' + filename + '.mp3'
        target_file = self.target_path + '/' + filename + '.wav'

        kHz = str(self.sample_rate / 1000.0)
        cmd = 'lame -a -m m {0} {1}'.format(quote(origin_file), quote(tmp_file))
        os.system(cmd)
        cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(tmp_file), quote(target_file), kHz)
        os.system(cmd)
        return target_file

    def convert_directory(self):
        for item in self.files:
            target = self.mp3_to_wav(item)
            print('{} converted to wave'.format(target))


import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
import wave
import math

def path_check(mp3_path='./mp3', wav_path='./wav'):
    if not os.path.exists(mp3_path):
        os.makedirs(mp3_path)
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)

def mp3_to_wav(filename, rate):
    if not filename[:-4] == '.mp3':
        return '%s not an mp3 file'%(filename)
    original

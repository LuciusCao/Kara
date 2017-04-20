import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
import wave
import math

def config_path(root):
    path = {}
    path['mp3'] = os.path.abspath(root + '/mp3')
    path['tmp'] = os.path.abspath(root + '/tmp')
    path['wav'] = os.path.abspath(root + '/wav')
    return path


def calc_files_to_convert(path):
    files_from_mp3 = os.listdir(path['mp3'])
    files_from_wav = os.listdir(path['wav'])
    mp3_files = [item[:-4] for item in files_from_mp3 if item[-4:] == 'mp3']
    wav_files = [item[:-4] for item in files_from_wav if item[-4:] == 'wav']
    target_files = [item for item in mp3_files if item not in wav_files]
    return target_files

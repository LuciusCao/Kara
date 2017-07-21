import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
import wave
import math

def config_path(root):
    '''
    setup the necessary path for the project
    arguement
    - root: string, root directory of the project
    return
    - path: dictionary, that has folder names as keys
    and their abs path as values
    '''
    path = {}
    path['mp3'] = os.path.abspath(root + '/mp3')
    path['tmp'] = os.path.abspath(root + '/tmp')
    path['wav'] = os.path.abspath(root + '/wav')
    return path


def get_files_by_fmt(path, fmt, with_ext=False):
    '''
    get a file list from a folder by certain format
    argument
    - path: path dictionary with fmt as their key
    - fmt: string
    - with_ext: boolean, by fault is False, if set to True, return files
    will have ext.
    return
    - fmt_files: list of files
    '''
    fmt_len = len(fmt)
    ext_len = fmt_len + 1
    ext = '.%s'%(fmt)
    files_from_fmt = os.listdir(path[fmt])
    if with_ext:
        fmt_files = [item for item in files_from_fmt if item[-ext_len:] == ext]
    else:
        fmt_files = [item[:-ext_len] for item in files_from_fmt if item[-ext_len:] == ext]
    return fmt_files


def calc_files_to_convert(path):
    '''
    calculate what files need to be converted to avoid duplication
    arguement
    - path: path dictionary that holds folder names as keys
    and their abs path as values
    return
    - target_files: list of files
    '''
    mp3_files = get_files_by_fmt(path, 'mp3')
    wav_files = get_files_by_fmt(path, 'wav')
    target_files = [item for item in mp3_files if item not in wav_files]
    return target_files


def convert_mp3_to_wav(mp3_filename, path, sample_rate=44100):
    '''
    use lame cmd line tool to convert an mp3 to wav
    arguement
    - mpe_filename: string, can be either with ext or not, the fucntion will
    check and add ext on if not exists
    - path: dictionary
    - sample_rate: integer, kHz
    return
    - None
    '''
    if mp3_filename[-4:] == '.mp3':
        filename = mp3_filename[:-4]
    else:
        filename = mp3_filename
    mp3_file_path = path['mp3'] + '/' + filename + '.mp3'
    tmp_file_path = path['tmp'] + '/' + filename + '.mp3'
    wav_file_path = path['wav'] + '/' + filename + '.wav'

    cmd = 'lame -a -m m %s %s'%(quote(mp3_file_path), quote(tmp_file_path))
    os.system(cmd)
    cmd = 'lame --decode %s %s --resample %s'%(quote(tmp_file_path), quote(wav_file_path), str(sample_rate))
    os.system(cmd)
    return


def convert_all(target_list, path):
    '''
    convert target files to .wav and return the number of files converted
    arguement
    - target_list: list of path
    - path: dictionary
    return
    - i: integer that stands for how many files are converted
    '''
    i = 0
    for each_mp3 in target_list:
        convert_mp3_to_wav(each_mp3, path)
        i += 1
    return i


def read_wav_as_np(filename):
    '''
    read a wav file as an numpy array
    arguement
    - filename, path to wav file
    return
    - np_arr, numpy array of the wav file
    '''
    data = wav.read(filename)
    np_arr = data[1].astype('float32') / 32767.0  # Normalize 16-bit input to [-1, 1] range
    return np_arr, data[0]


def write_np_as_wav(X, sample_rate, filename):
    '''
    write a numpy array as a wav file
    arguement
    - X, np array
    - sample_rate, int use 44100 if the original wave file is encoded such way
    - filename, path to save the wav file
    return None
    '''
    Xnew = X * 32767.0
    Xnew = Xnew.astype('int16')
    wav.write(filename, sample_rate, Xnew)
    return


def convert_np_audio_to_sample_blocks(song_np, block_size):
    block_lists = []
    total_samples = song_np.shape[0]
    num_samples_so_far = 0
    while (num_samples_so_far < total_samples):
        block = song_np[num_samples_so_far:num_samples_so_far + block_size]
        if (block.shape[0] < block_size):
            padding = np.zeros(
                    (block_size - block.shape[0],))
            block = np.concatenate((block,padding))
        block_lists.append(block)
        num_samples_so_far += block_size
    return block_lists


def convert_sample_blocks_to_np_audio(blocks):
    song_np = np.concatenate(blocks)
    return song_np


def time_blocks_to_fft_blocks(blocks_time_domain, count):
    fft_blocks = []
    amplitude = []
    for block in blocks_time_domain:
        fft_block = np.fft.fft(block)
        new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
        fft_blocks.append(new_block)
    return fft_blocks


def fft_blocks_to_time_blocks(blocks_ft_domain):
    time_blocks = []
    for block in blocks_ft_domain:
        num_elems = block.shape[0] / 2
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        new_block = real_chunk + 1.0j * imag_chunk
        time_block = np.fft.ifft(new_block)
        time_blocks.append(time_block)
    return time_blocks


def convert_wav_files_to_nptensor(directory, block_size, max_seq_len, out_file, max_files=20, useTimeDomain=False):
    files = []

    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(directory + file)

    chunks_X = []
    chunks_Y = []

    num_files = len(files)
    if (num_files > max_files):
        num_files = max_files

    for file_idx in range(num_files):
        file = files[file_idx]

        print('Processing: ', (file_idx + 1), '/', num_files)
        print('Filename: ', file)

        X, Y = load_training_example(file, block_size, useTimeDomain=useTimeDomain)
        cur_seq = 0
        total_seq = len(X)
        print("X.shape", np.shape(X))
        print(total_seq)
        print(max_seq_len)
        while cur_seq + max_seq_len < total_seq:
            chunks_X.append(X[cur_seq:cur_seq + max_seq_len])
            chunks_Y.append(Y[cur_seq:cur_seq + max_seq_len])
            cur_seq += max_seq_len
    num_examples = len(chunks_X)
    num_dims_out = block_size * 2
    if (useTimeDomain):
        num_dims_out = block_size
    out_shape = (num_examples, max_seq_len, num_dims_out)
    x_data = np.zeros(out_shape)
    y_data = np.zeros(out_shape)
    for n in range(num_examples):
        for i in range(max_seq_len):
            x_data[n][i] = chunks_X[n][i]
            y_data[n][i] = chunks_Y[n][i]
        print('Saved example ', (n + 1), ' / ', num_examples)
    print('Flushing to disk...')
    mean_x = np.mean(np.mean(x_data, axis=0), axis=0)
    std_x = np.sqrt(
            np.mean(np.mean(np.abs(x_data - mean_x) ** 2, axis=0), axis=0))
    std_x = np.maximum(1.0e-8, std_x)

    np.save(out_file + '_mean', mean_x)
    np.save(out_file + '_var', std_x)
    np.save(out_file + '_x', x_data)
    np.save(out_file + '_y', y_data)
    print("x_data=", np.shape(x_data))
    print("y_data=", np.shape(y_data))
    inter_filename = out_file + '_x'
    convert_nptensor_to_wav_files_verify(x_data, num_examples, inter_filename, False)
    print('Done converting the input to the neural network to a WAV file')
    print('Done converting the WAV file to an np tensor to feed to the RNN ')


def convert_nptensor_to_wav_files_verify(tensor, indices, filename, useTimeDomain=False):
    num_seqs = tensor.shape[1]
    chunks = []
    for i in range(indices):
        for x in range(num_seqs):
            chunks.append(tensor[i][x])
    save_generated_example(filename + 'merged' + '.wav', chunks, useTimeDomain=useTimeDomain)


def convert_nptensor_to_wav_files(tensor, indices, filename, useTimeDomain=False):
    num_seqs = tensor.shape[1]
    for i in indices:
        chunks = []
        for x in range(num_seqs):
            chunks.append(tensor[i][x])
        save_generated_example(filename + str(i) + '.wav', chunks, useTimeDomain=useTimeDomain)


def load_training_example(filename, block_size=2048, useTimeDomain=False):
    data, bitrate = read_wav_as_np(filename)
    x_t = convert_np_audio_to_sample_blocks(data, block_size)
    y_t = x_t[1:]
    y_t.append(np.zeros(block_size))
    if useTimeDomain:
        return x_t, y_t

    X = time_blocks_to_fft_blocks(x_t, count=1)
    Y = time_blocks_to_fft_blocks(y_t, count=2)
    print(np.shape(X))
    return X, Y


def save_generated_example(filename, generated_sequence, useTimeDomain=False, sample_frequency=44100):
    if useTimeDomain:
        time_blocks = generated_sequence
    else:
        time_blocks = fft_blocks_to_time_blocks(generated_sequence)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    write_np_as_wav(song, sample_frequency, filename)
    return


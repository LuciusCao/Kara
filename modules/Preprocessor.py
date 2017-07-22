import os
from pipes import quote

class Preprocessor:
    '''
    Read dataset path and convert all mp3 files to wav files
    '''
    def __init__(self, root_path, sample_rate=44100):
        self.path = {
            'mp3': os.path.abspath(root_path+'/mp3'),
            'tmp': os.path.abspath(root_path+'/tmp'),
            'wav': os.path.abspath(root_path+'/wav')
        }
        self.sample_rate = sample_rate
        self.target_files = self.calc_files_to_convert()

    def _get_list_of_files(self, fmt):
        ext_len = len(fmt) + 1
        ext = '.%s'%(fmt)
        files_from_fmt = os.listdir(self.path[fmt])
        fmt_files = [item[:-ext_len] for item in files_from_fmt if item[-ext_len:] == ext ]
        return fmt_files

    def calc_files_to_convert(self):
        wav_files = self._get_list_of_files('wav')
        mp3_files = self._get_list_of_files('mp3')
        return [item for item in mp3_files if item not in wav_files]

    def _convert_mp3_to_wav(self, filename):
        mp3_file_path = '%s/%s.mp3'%(path['mp3'], filename)
        tmp_file_path = '%s/%s.mp3'%(path['tmp'], filename)
        wav_file_path = '%s/%s.wav'%(path['wav'], filename)
        cmd = 'lame -a -m m %s %s'%(quote(mp3_file_path), quote(tmp_file_path))
        os.system(cmd)
        cmd = 'lame --decode %s %s --resample %s'%(quote(tmp_file_path), quote(wav_file_path), str(self.sample_rate))
        os.system(cmd)
        return

    def convert_all(self):
        i = 0
        for each_mp3 in self.target_files:
            self._convert_mp3_to_wav(each_mp3)
            i += 1
        print (i, 'files have been converted')
        return i

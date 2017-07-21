import os

class Preprocessor:
    def __init__(self, root_path):
        self.path = {
            'mp3': os.path.abspath(root_path+'/mp3'),
            'tmp': os.path.abspath(root_path+'/tmp'),
            'wav': os.path.abspath(root_path+'/wav')
        }

    def get_list_of_file(self, fmt, with_ext=False):
        ext_len = len(fmt) + 1
        ext = '.%s'%(fmt)
        files_from_fmt = os.listdir(self.path[fmt])
        if with_ext:
            fmt_files = [item for item in files_from_fmt if item[-ext_len:] == ext]
        else:
            fmt_files = [item[:-ext_len] for item in files_from_fmt if itemp[-ext_len:] == ext ]
        return fmt_files

    def calc_files_to_convert(self):
        wav_files = self.get_list_of_file('wav')
        mp3_files = self.get_list_of_file('mp3')
        target_files = [item for item in mp3_files if item not in wav_files]
        return target_files


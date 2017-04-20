from helpers import *

if __name__ == '__main__':
    path = config_path('./dataset')
    target_files = calc_files_to_convert(path)
    if target_files:
        print('%d files to be converted'%(len(target_files)))
    else:
        print('no files to be converted')


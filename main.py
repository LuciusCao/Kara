from helpers import *
import numpy as np
import os

if __name__ == '__main__':
    path = config_path('./dataset')
    target_files = calc_files_to_convert(path)
    input('%d files to be converted, press ENTER to continue'%(len(target_files)))
    
    convert_all(target_files, path) 


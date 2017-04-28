import helpers
import numpy as np
import os

if __name__ == '__main__':
    path = helpers.config_path('./dataset')
    target_files = helpers.calc_files_to_convert(path)
    input('%d files to be converted, press ENTER to continue'%(len(target_files)))
    
    nb_converted = helpers.convert_all(target_files, path) 
    print('%d / %d files converted'%(nb_converted, len(target_files)))


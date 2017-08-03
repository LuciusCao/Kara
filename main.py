from modules.preprocessor import Preprocessor
from modules.loader import Loader
#  from modules.writer import Writer
#  from modules.model import build_basic, build_td_basic, build_seq2seq
from config import config
from parser import parser, parser_train, parser_generate, parser_prepare
import os


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
    elif args.mode == 'prepare':
        preprocessor = Preprocessor(config['dataset_root'])
        num_files, target_list = preprocessor.convert_all()
        print(num_files, 'has been converted')
    elif args.mode == 'train':
        loader = Loader(
            config['input_wav'],
            config['saved_data'],
            config['timestep'],
            config['seq_len'],
        )
        print('========loading data========')
        x, y = loader.load_training_data()
    elif args.mode == 'generate':
        print('generate mode')
    else:
        parser.print_help()
    #  print('========building models========')
    #  model = build_seq2seq(shape[1], shape[2], 64, depth=3)
    #  print('========training========')
    #  model.fit(x, y, batch_size=1, epochs=1)
    #  print('========saving model weights========')
    #  model.save_weights('trained_weights.h5')
    #  print('========Done!========')
    #  writer = Writer()

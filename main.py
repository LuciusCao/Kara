from modules.preprocessor import Preprocessor
from modules.loader import Loader
from modules.writer import Writer
from config import config
from parser import parser
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
        from modules.model import build_seq2seq

        loader = Loader(
            config['input_wav'],
            config['saved_data'],
            config['timestep'],
            config['seq_len'],
        )
        print('========loading data========')
        x, y = loader.load_training_data(force_reload=args.reload_dir)
        shape = x.shape

        print('========building models========')
        model = build_seq2seq(shape[1], shape[2], 32, depth=2)
        if args.rebuild is True:
            model.fit(x, y,
                      batch_size=config['batch_size'],
                      epochs=config['epochs'])
        else:
            try:
                model.load_weights(config['model_path'])
                print('model loaded from {}'.format(config['model_path']))
            except ValueError as e:
                print(e)
            except OSError as e:
                print(e)

            print('========training========')
            model.fit(x, y,
                        batch_size=config['batch_size'],
                        epochs=config['epochs'])

        print('========saving model weights========')
        model.save_weights('trained_weights.h5')
    elif args.mode == 'generate':
        print('generate mode')
    else:
        parser.print_help()
    #  print('========Done!========')
    #  writer = Writer()

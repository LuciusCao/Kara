from kara.writer import Writer
from config import config
from parser import parser
import os


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()

    elif args.mode == 'prepare':
        from kara.preprocessor import Preprocessor

        preprocessor = Preprocessor(config['dataset_root'])
        num_files, target_list = preprocessor.convert_all()
        print(num_files, 'has been converted')

    elif args.mode == 'train':
        from kara.loader import Loader
        from kara.model import build_seq2seq

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
        model = build_seq2seq(config['timestep'],
                              config['seq_len'] * 2,
                              32, depth=config['depth'])
        if args.rebuild is True:
            model.fit(x, y,
                      batch_size=config['batch_size'],
                      epochs=config['epochs'])
        else:
            try:
                model.load_weights(config['model_path'])
                print('model loaded from {}'.format(config['model_path']))
            except ValueError as e:
                print('Unexpected data shape. '
                      'Check you input data and your model')
            except OSError as e:
                print('Cannot find model weights. '
                      'Will train from scratch')

            print('========training========')
            model.fit(x, y,
                      batch_size=config['batch_size'],
                      epochs=config['epochs'])

        print('========saving model weights========')
        model.save_weights(config['model_path'])

    elif args.mode == 'generate':
        from kara.model import build_seq2seq

        model = build_seq2seq(config['timestep'],
                              config['seq_len'] * 2,
                              32, depth=config['depth'])
        try:
            model.load_weights(config['model_path'])
            print('model loaded from {}'.format(config['model_path']))
        except ValueError as e:
            raise ValueError
        except OSError as e:
            raise OSError
    else:
        parser.print_help()
    #  print('========Done!========')
    #  writer = Writer()

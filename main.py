from config import get_config
from parser import parser
import numpy as np
import os


if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(epochs=args.epochs)
    if args.mode is None:
        parser.print_help()

    elif args.mode == 'prepare':
        from kara.preprocessor import Preprocessor

        preprocessor = Preprocessor(config['dataset_root'],
                                    config['sample_rate'])
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
        min_max = np.load(config['saved_data']['min_max'])
        mini = min_max[0]
        maxi = min_max[1]
        norm_x = (x - mini) / (maxi - mini)
        norm_y = (x - mini) / (maxi - mini)

        print('========building models========')
        model = build_seq2seq(config['timestep'],
                              config['seq_len'],
                              32, config['loss'], depth=config['depth'])
        if args.rebuild is True:
            model.fit(norm_x, norm_y,
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
            model.fit(norm_x, norm_y,
                      batch_size=config['batch_size'],
                      epochs=config['epochs'])

        print('========saving model weights========')
        model.save_weights(config['model_path'])

    elif args.mode == 'generate':
        from kara.model import build_seq2seq
        from kara.writer import Writer

        min_max = np.load(config['saved_data']['min_max'])
        mini = min_max[0]
        maxi = min_max[1]

        sample_length = args.length
        sample_frequency = sample_length * config['sample_rate']
        num_iter = sample_frequency // config['seq_len']

        model = build_seq2seq(config['timestep'],
                              config['seq_len'],
                              32, config['loss'], depth=config['depth'])

        model.load_weights(config['model_path'])
        init_sample = np.load(config['saved_data']['init_sample'])
        print('model loaded from {}'.format(config['model_path']))

        generated_sample = []
        cur_iter = 0

        while cur_iter < num_iter:
            try:
                generated_sample.append(model.predict(generated_sample[-1]))
            except IndexError:
                generated_sample.append(model.predict(init_sample))
            cur_iter += 1

        generated_sample = np.array(generated_sample)
        shape = generated_sample.shape
        new_shape = (shape[0], shape[2], shape[3])
        generated_sample = generated_sample.reshape(new_shape)
        generated_sample = generated_sample * (maxi - mini) + mini
        generated_sample = np.concatenate(generated_sample)

        writer = Writer(generated_sample,
                        config['out_wav'],
                        config['sample_rate'])
        writer.write_fft_to_wav()

    else:
        parser.print_help()
    #  print('========Done!========')
    #  writer = Writer()

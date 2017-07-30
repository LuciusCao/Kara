#  from modules.Preprocessor import Preprocessor
#  from modules.Loader import Loader
#  from modules.Writer import Writer
#  from modules.Model import build_basic, build_td_basic, build_seq2seq
import os
import argparse


if __name__ == '__main__':
    root_path = os.path.abspath('./dataset')

    parser = argparse.ArgumentParser(prog='kara',
                                     description='An experiment that trains \
                                     models to learn to compose music',
                                     epilog='If you have any trouble feel free \
                                     to send a ticket on Github, or send email \
                                     to lucius.cao@gmail.com')
    subparsers = parser.add_subparsers(metavar='train generate')
    parser_train = subparsers.add_parser('train',
                                         description='Training module for the \
                                         model',
                                         help='train your model')
    parser_train.add_argument('--input', type=str,
                              metavar='dir',
                              help='specify your input audio directory')
    parser_generate = subparsers.add_parser('generate',
                                            help='generate results based on \
                                            your model')
    parser_generate.add_argument('--output', type=str, default='out.wav',
                                 metavar='path',
                                 help='specify your output directory, \
                                 default out.wav')
    parser_generate.add_argument('--length', type=int, default=15,
                                 metavar='length',
                                 help='length of output audio in seconds, \
                                 default 15 seconds')
    args = parser.parse_args()
    if args.__dict__ == {}:
        cmd = 'python main.py --help'
        os.system(cmd)
    else:
        print('args parsed')
    #  preprocessor = Preprocessor(root_path)
    #  preprocessor.convert_all()
    #  loader = Loader(
        #  os.path.abspath('./dataset/wav/a_hisa - Town of Windmill.wav'),
        #  2048, 32
    #  )
    #  print('========loading data========')
    #  x, y, shape = loader.load_training_data()
    #  print('========building models========')
    #  model = build_seq2seq(shape[1], shape[2], 64, depth=3)
    #  print('========training========')
    #  model.fit(x, y, batch_size=1, epochs=1)
    #  print('========saving model weights========')
    #  model.save_weights('trained_weights.h5')
    #  print('========Done!========')
    #  writer = Writer()

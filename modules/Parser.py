import argparse


parser = argparse.ArgumentParser(prog='kara',
                                 description='An experiment that trains \
                                 models to learn to compose music',
                                 epilog='If you have any trouble feel free \
                                 to send a ticket on Github, or send email \
                                 to lucius.cao@gmail.com')
subparsers = parser.add_subparsers(metavar='subcommand', dest='mode')
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

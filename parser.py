import argparse


parser = argparse.ArgumentParser(prog='kara',
                                 description='An experiment that trains \
                                 models to learn to compose music',
                                 epilog='If you have any trouble feel free \
                                 to send a ticket on Github, or send email \
                                 to lucius.cao@gmail.com')
subparsers = parser.add_subparsers(metavar='subcommand', dest='mode')
parser_prepare = subparsers.add_parser('prepare',
                                       help='run this command to prepare your \
                                       data')
parser_train = subparsers.add_parser('train',
                                     description='Training module for the \
                                     model',
                                     help='train your model')
parser_train.add_argument('--reload', action='store_true', dest='reload_dir',
                          help='force to reload all music from your dataset')
parser_train.add_argument('--rebuild', action='store_true', dest='rebuild',
                          help='force to rebuild and train the model from \
                          scratch')
parser_generate = subparsers.add_parser('generate',
                                        help='generate results based on \
                                        your model')
parser_generate.add_argument('--length', type=int, default=15,
                             metavar='length',
                             help='length of output audio in seconds, \
                             default 15 seconds')

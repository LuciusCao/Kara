from modules.preprocessor import Preprocessor
from modules.loader import Loader
#  from modules.writer import Writer
#  from modules.model import build_basic, build_td_basic, build_seq2seq
from config import config
from parser import parser, parser_train, parser_generate
import os


if __name__ == '__main__':
    project_root = config['project_root']
    dataset_path = os.path.join(project_root, 'dataset')

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
    elif args.mode == 'train':
        print('trainning mode')
    elif args.mode == 'generate':
        print('generate mode')
    else:
        parser.print_help()
    preprocessor = Preprocessor(dataset_path)
    preprocessor.convert_all()
    loader = Loader(
        os.path.abspath('./dataset/wav'),
        config['timestep'],
        config['seq_len'],
    )
    print('========loading data========')
    x, y, shape = loader.load_directory()
    #  print('========building models========')
    #  model = build_seq2seq(shape[1], shape[2], 64, depth=3)
    #  print('========training========')
    #  model.fit(x, y, batch_size=1, epochs=1)
    #  print('========saving model weights========')
    #  model.save_weights('trained_weights.h5')
    #  print('========Done!========')
    #  writer = Writer()

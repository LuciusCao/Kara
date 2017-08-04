import os


project_root = os.path.abspath('./')

dataset_root = os.path.join(project_root, 'dataset')
saved_data = os.path.join(dataset_root, 'saved_data')
input_wav = os.path.join(dataset_root, 'wav')

config = {
    'project_root': project_root,
    'dataset_root': dataset_root,
    'saved_data': {
        'x': saved_data + '_x.npy',
        'y': saved_data + '_y.npy',
    },
    'input_wav': input_wav,
    'out_wav': os.path.abspath('./output.wav'),
    'model': os.path.join(project_root, 'trained_weights.h5'),
    'timestep': 4,
    'seq_len': 2048,
    'epochs': 1,
    'batch_size': 32,
    'learning_rate': 0.001,
}

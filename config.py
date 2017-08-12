import os


def get_config(epochs=1, batch_size=32):
    project_root = os.path.abspath('./')

    dataset_root = os.path.join(project_root, 'dataset')
    model_path = os.path.join(project_root, 'trained_weights.h5')
    out_wav = os.path.join(project_root, './output.wav')

    saved_data = os.path.join(dataset_root, 'saved_data')
    init_sample = os.path.join(dataset_root, 'init_sample.npy')
    input_wav = os.path.join(dataset_root, 'wav')
    min_max = os.path.join(dataset_root, 'min_max.npy')

    seq_len = 44100
    fft_seq_len = seq_len * 2

    config = {
        'project_root': project_root,
        'dataset_root': dataset_root,
        'saved_data': {
            'x': saved_data + '_x.npy',
            'y': saved_data + '_y.npy',
            'init_sample': init_sample,
            'min_max': min_max,
        },
        'input_wav': input_wav,
        'out_wav': out_wav,
        'model_path': model_path,
        'sample_rate': 44100,
        'timestep': 4,
        'seq_len': seq_len,
        'fft_seq_len': fft_seq_len,
        'depth': 2,
        'epochs': epochs,
        'batch_size': batch_size,
        'loss': 'mean_absolute_error',
        'learning_rate': 0.001,
    }
    return config

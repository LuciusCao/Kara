import os


config = {
    'project_root': os.path.abspath('./'),
    'out_wav': os.path.abspath('./output.wav'),
    'timestep': 4,
    'seq_len': 2048,
    'epochs': 1,
    'batch_size': 32,
    'learning_rate': 0.001,
}

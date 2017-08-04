# Kara
This project is an experiment to generate audio from mp3 file via deep learning

## Usage
here is a brief guide on how to run the project on your own

### Setup Environment
This repo depends on a library called lame to convert mp3 file to wav file, before your get started make sure you have installed it correctly
**macOS**
`brew install lame`
pyenv or virtualenv are recommended and used by the author. Python 3.6.1 is used in this project.
`pip install requirement.txt`
make sure your mp3 file are tossed into the dataset/mp3 directory. Right now pure music is suggested

### Prepare for training
Before we start to train the model, mp3 file needs to be converted to wav
`python main.py prepare`
This command will help you setup the proper directory and convert mp3s to wavs

### Training
`python main.py train`
This command will try to search for existing .npy file to read, if there isn't one it will read your wav directory to generate training samples. You can change the behavior by adding `--reload` flag to always load from wav files directly
And it will use your trained weights for the model, if there isn't one it will train from scratch, and after training is completed, the model will be saved to 'saved_weights.h5'. You can change the behavior by adding `--rebuild`flag to force the model to train from scratch.
If you have changed the architecture of the model, the model will rebuild it self to train from scratch

## Known Issues
- the audio generated is with lots of noise (it has been found that if there isn't any transformation on the read wav file, there won't been quality loss in the generated file)
- The training model doesn't work at all, need a way to debug it. right now with 3 layers encoder and 3 layers decoder, and each layer has 128 units the loss there is about 1.16, which is not an ideal training loss at all!. After I add more song to the dataset/wav, the training loss goes down to 0.6. Nevertheless... the models works
- The way to deal with fourier transformed sequence should be evaluated

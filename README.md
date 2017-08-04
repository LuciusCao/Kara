# Kara
This project is an experiment to generate audio from mp3 file via deep learning

## Known Issues
- the audio generated is with lots of noise (it has been found that if there isn't any transformation on the read wav file, there won't been quality loss in the generated file)
- The training model doesn't work at all, need a way to debug it. right now with 3 layers encoder and 3 layers decoder, and each layer has 128 units the lost there is about 1.16, which is not an ideal training loss at all!. Nevertheless... the models works
- The way to deal with fourier transformed sequence should be evaluated

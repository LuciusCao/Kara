from preprocessor.converter import Converter

conv = Converter('./mp3', './tmp', './wav')
conv.convert_directory()

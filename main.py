from converter import converter

conv = converter.Converter('./mp3', './tmp', './wav')
conv.convert_directory()
